# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pandas.core.apply import com
import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class SPGTrainer(GRPOTrainer):
    """
    Sandwiched Policy Gradient (SPG) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation that leverages both an upper and 
    a lower bound of the true log-likelihood.

    Key features:
    - Separate log-likelihood estimation for positive and negative advantage traces
    - Block-wise masking for Monte Carlo estimation of log-likelihood
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_seq_logps = self._get_per_seq_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed], prompt_mask, completion_mask, reward_mask=inputs['reward_mask']) # num_iterations, batch_size
        
        # Check for NaN/inf in per_seq_logps
        if torch.isnan(per_seq_logps).any() or torch.isinf(per_seq_logps).any():
            print(f"WARNING: NaN/inf detected in per_seq_logps!")
            print(f"per_seq_logps: {per_seq_logps}")
            print(f"per_seq_logps.shape: {per_seq_logps.shape}")
            print(f"NaN count: {torch.isnan(per_seq_logps).sum()}")
            print(f"Inf count: {torch.isinf(per_seq_logps).sum()}")

        # Compute the loss
        advantages = inputs["advantages"]
        per_seq_loss = -advantages.unsqueeze(0) * per_seq_logps # [1, batch_size]
        completion_length = completion_mask.sum(dim=1).unsqueeze(0) # [1, batch_size]
        loss = (per_seq_loss * completion_length).sum() / completion_length.sum() # [1]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            raise NotImplementedError("KL divergence is not supported for SPG")

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        prompt_mask=None,
        # return_trajectory_samples=False,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()
            if prompt_mask is not None:
                # extend prompt_mask to the same shape as x, it originally has [bsz, prompt_length]
                prompt_mask = prompt_mask.bool()
                gen_mask = torch.ones(bs, gen_length, dtype=torch.bool).to(model.device)
                prompt_mask = torch.cat([prompt_mask, gen_mask], dim=1)

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_, attention_mask=prompt_mask).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x, attention_mask=prompt_mask).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index
            return x

    def forward_process(self, batch, prompt_index, mask_id, seed=None, completion_mask=None):#, trajectory_samples=None):
        set_seed(seed)
        forward_type = self.args.forward_type # "all", "random", "block_all", "block_random"
        # all: all gen_tokens are masked
        #    assert num_t == 1
        # random: randomly mask some gen_tokens
        #    - num_t: number of sampled timesteps
        #    - min_t: minimum timestep
        #    - max_t: maximum timestep
        # block_all: mask by blocks, all gen_tokens in a block are masked
        #    - num_t: number of sampled timesteps (sample from [0, ..., num_blocks-1], assert num_t <= num_blocks for now)
        # block_random: randomly mask some gen_tokens in a block
        #    - num_t: number of sampled timesteps (sample from [0, ..., num_blocks-1], assert num_t <= num_blocks for now)
        #    - min_t: minimum timestep
        #    - max_t: maximum timestep
        num_t = getattr(self.args, "num_t", 1)
        min_t = getattr(self.args, "min_t", 0)
        max_t = getattr(self.args, "max_t", 1)
        if forward_type == "all":
            assert num_t == 1
            b, l = batch.shape
            if self.args.use_mask_prompt:
                t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

                # Create a random matrix to decide whether each prompt token is masked
                random_matrix = torch.rand((b, l), device=batch.device)

                # For prompt tokens: mask if random_matrix < t_p
                # For completion tokens: always mask
                is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
                is_mask_completion = ~prompt_index  # all completion tokens are masked
                is_mask = is_mask_prompt | is_mask_completion
            else:
                is_mask_completion = ~prompt_index  # all completion tokens are masked
                is_mask = is_mask_completion

            # Create a noisy (masked) batch
            noisy_batch = torch.where(is_mask, mask_id, batch) # [b, l]
            noisy_batch = noisy_batch.unsqueeze(1) # [b, 1, l]
            block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)

        elif forward_type == "random":
            b, l = batch.shape
            gen_length = (l - prompt_index.sum()).item()
            completion_length = completion_mask.sum(-1)
            is_mask = torch.zeros((b, num_t, gen_length), dtype=torch.bool, device=batch.device)
            for i in range(b):
                start_mask_num = max(int(completion_length[i] * min_t), 1)
                end_mask_num = min(int(completion_length[i] * max_t), completion_length[i])
                assert start_mask_num <= end_mask_num
                mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, num_t), device=batch.device) # [1, num_t]
                # randomly select mask_num tokens to mask for each sequence
                indices = torch.arange(gen_length, device=batch.device).repeat(1, num_t, 1) # [1, num_t, gen_length]
                is_mask[[i], :, :] = indices < mask_num.unsqueeze(2) # [1, num_t, gen_length]
                for j in range(num_t):
                    is_mask[i, j, :completion_length[i]] = is_mask[i, j, :completion_length[i]][torch.randperm(completion_length[i])]
            is_mask = torch.cat((torch.zeros(b, num_t, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=2) # [b, num_t, l]
            completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_index.sum(), dtype=torch.bool, device=batch.device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)
            if self.args.use_mask_prompt:
                t_p = torch.ones(b, num_t, device=batch.device) * self.args.p_mask_prompt
                random_matrix = torch.rand((b, num_t, l), device=batch.device)
                is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(2))
                is_mask = (is_mask_prompt | is_mask) | ~completion_mask_append
            else:
                is_mask = is_mask | ~completion_mask_append # mask all tokens after the first EOS token
            noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1)) # [b, num_t, l]
            block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
        elif forward_type == "block_all":
            b, l = batch.shape
            gen_length = (l - prompt_index.sum()).item()
            block_length = self.args.block_length
            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            completion_num_blocks = (completion_mask.sum(-1)-1)//block_length+1
            assert num_t <= num_blocks
                
            indices = torch.arange(num_blocks, device=batch.device).repeat(b, 1) # [b, num_blocks]
            for i in range(b):
                indices[i] = indices[i][torch.randperm(num_blocks)] % completion_num_blocks[i]
            mask_block_idx = indices[:, :num_t]
            is_mask = torch.zeros((b, num_t, l), dtype=torch.bool, device=batch.device)
            block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
            for i in range(b):
                for j in range(num_t):
                    is_mask[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length:] = True
                    if mask_block_idx[i, j] < num_blocks - 1:
                        block_mask[i, j, -(num_blocks - mask_block_idx[i, j] - 1) * block_length:] = False
            if self.args.use_mask_prompt:
                t_p = torch.ones(b, num_t, device=batch.device) * self.args.p_mask_prompt
                random_matrix = torch.rand((b, num_t, l), device=batch.device)
                is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
                is_mask = is_mask_prompt | is_mask
            else:
                is_mask = is_mask
            noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1)) # [b, num_t, l]
            
        elif forward_type == "block_random":
            b, l = batch.shape
            gen_length = (l - prompt_index.sum()).item()
            block_length = self.args.block_length
            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            completion_num_blocks = (completion_mask.sum(-1)-1)//block_length+1
            assert num_t <= num_blocks
            indices = torch.arange(num_blocks, device=batch.device).repeat(b, 1) # [b, num_blocks]
            for i in range(b):
                indices[i] = indices[i][torch.randperm(num_blocks)] % completion_num_blocks[i]
            mask_block_idx = indices[:, :num_t]
            is_mask = torch.zeros((b, num_t, l), dtype=torch.bool, device=batch.device)
            block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
            for i in range(b):
                for j in range(num_t):
                    is_mask[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length:] = True
                    if mask_block_idx[i, j] < num_blocks - 1:
                        block_mask[i, j, -(num_blocks - mask_block_idx[i, j] - 1) * block_length:] = False
            completion_length = completion_mask.sum(-1)
            is_mask_following = torch.ones((b, num_t, l), dtype=torch.bool, device=batch.device)
            for i in range(b):
                for j in range(num_t):
                    mask_length = min(block_length, completion_length[i] - block_length * mask_block_idx[i, j])
                    assert mask_length > 0
                    start_mask_num = max(int(mask_length * min_t), 1)
                    end_mask_num = min(int(mask_length * max_t), mask_length)
                    assert start_mask_num <= end_mask_num
                    mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, 1), device=batch.device) # [1, 1]
                    # randomly select mask_num tokens to mask for each sequence
                    indices = torch.arange(block_length, device=batch.device).repeat(1, 1, 1) # [1, 1, block_length]
                    is_mask_next = indices < mask_num.unsqueeze(2) # [1, 1, block_length]
                    if mask_block_idx[i, j] == num_blocks - 1 and mask_length == block_length:
                        is_mask_following[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length:] = is_mask_next[0, 0][torch.randperm(block_length)]
                    else:
                        is_mask_following[i, j, -(num_blocks - mask_block_idx[i, j]) * block_length: -(num_blocks - mask_block_idx[i, j]) * block_length + mask_length] = is_mask_next[0, 0, :mask_length][torch.randperm(mask_length)]
            completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_index.sum(), dtype=torch.bool, device=batch.device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)
            if self.args.use_mask_prompt:
                t_p = torch.ones(b, num_t, device=batch.device) * self.args.p_mask_prompt
                random_matrix = torch.rand((b, num_t, l), device=batch.device)
                is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
                is_mask = is_mask_prompt | (is_mask & is_mask_following) | ~completion_mask_append
            else:
                is_mask = (is_mask & is_mask_following) | ~completion_mask_append
            noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1)) # [b, num_t, l]

        return noisy_batch, block_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id, prompt_mask=None):
        if len(batch.shape) == 3:
            multisample = True
            bsz, num_t, l = batch.shape
            batch = batch.view(-1, l)
            prompt_len = prompt_mask.shape[-1]
            prompt_mask = prompt_mask.unsqueeze(1).repeat(1, num_t, 1).view(-1, prompt_len)

        if prompt_mask is not None:
            prompt_mask = prompt_mask.bool()
            assert batch.shape[0] == prompt_mask.shape[0], f"batch.shape: {batch.shape}, prompt_mask.shape: {prompt_mask.shape}"
            prompt_mask = torch.cat([prompt_mask, torch.ones(batch.shape[0], batch.shape[1] - prompt_mask.shape[1], dtype=torch.bool, device=batch.device)], dim=1)
        
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])
            prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=1)
            
        input = batch
        logits = model(input, attention_mask=prompt_mask).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        if multisample:
            logits = logits.view(bsz, num_t, l, -1)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _get_per_seq_logps(self, model, input_ids, logits_to_keep, mask_seeds, prompt_mask=None, completion_mask=None, reward_mask=None):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        all_block_masks = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, block_mask = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed, completion_mask=completion_mask
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)
            all_block_masks.append(block_mask)
        all_block_masks = torch.cat(all_block_masks, dim=0) # [num_iterations * batch_size, num_t, seq_len]
        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, num_t, seq_len]
        perturb_mask = perturbed_seq == self.args.mask_id
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        if prompt_mask is not None:
            prompt_mask = torch.cat([prompt_mask]*num_iterations, dim=0)

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id, prompt_mask
        )  # [num_iterations * batch_size, num_t, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, num_t, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        perturb_mask = perturb_mask[:, :, -logits_to_keep:]
        all_block_masks = all_block_masks[:, :, -logits_to_keep:]

        completion_targets = completion_targets.unsqueeze(1).repeat(1, self.args.num_t, 1)
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        prob = F.softmax(flat_logits, dim=-1).gather(dim=-1, index=flat_targets.unsqueeze(-1))
        
        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, self.args.num_t, logits_to_keep)
        completion_probs = prob.view(num_iterations * batch_size, self.args.num_t, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, self.args.num_t, logits_to_keep)
        per_token_probs = completion_probs.view(num_iterations, batch_size, self.args.num_t, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        per_token_probs = per_token_probs.to(torch.float32)
        assert completion_mask is not None

        # all_ref_per_token_logps: num_iterations, batch_size, logits_to_keep
        # perturb_mask: num_iterations*batch_size, logits_to_keep
        # completion_mask: batch_size, logits_to_keep
        completion_mask_expanded = completion_mask.unsqueeze(0).unsqueeze(2).expand(num_iterations, -1, self.args.num_t, -1)
        perturb_mask_expanded = perturb_mask.view(num_iterations, batch_size, self.args.num_t, logits_to_keep)
        block_mask_expanded = all_block_masks.view(num_iterations, batch_size, self.args.num_t, logits_to_keep)
        assert per_token_logps.shape == (num_iterations, batch_size, self.args.num_t, logits_to_keep), f"per_token_logps.shape: {per_token_logps.shape}"
        assert completion_mask_expanded.shape == (num_iterations, batch_size, self.args.num_t, logits_to_keep), f"completion_mask_expanded.shape: {completion_mask_expanded.shape}"
        assert perturb_mask_expanded.shape == (num_iterations, batch_size, self.args.num_t, logits_to_keep), f"perturb_mask_expanded.shape: {perturb_mask_expanded.shape}"
        assert block_mask_expanded.shape == (num_iterations, batch_size, self.args.num_t, logits_to_keep), f"block_mask_expanded.shape: {block_mask_expanded.shape}"
        
        # For perturbed tokens, we should weight over instances (sequence-wise)
        # num_iterations, batch_size, num_t
        
        # Check for zero denominators before division
        denominator = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3)
        if (denominator == 0).any():
            print(f"WARNING: Zero denominator detected in per_seq_logps calculation!")
            print(f"denominator shape: {denominator.shape}")
            print(f"Zero count: {(denominator == 0).sum()}")
            print(f"completion_mask_expanded shape: {completion_mask_expanded.shape}")
            print(f"perturb_mask_expanded shape: {perturb_mask_expanded.shape}")
            print(f"completion_mask_expanded: {completion_mask_expanded}")
            print(f"perturb_mask_expanded: {perturb_mask_expanded}")
            # Add small epsilon to avoid division by zero
            denominator = torch.clamp(denominator, min=1e-8)
        
        per_seq_logps = (per_token_logps * completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / denominator
        if self.args.logp_estimation == 'eubo' or self.args.logp_estimation == 'mix':
            empirical_t = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / completion_mask_expanded.sum(dim=3)
            empirical_t_expanded = empirical_t.unsqueeze(3).expand(-1, -1, -1, completion_mask_expanded.size(-1))
            per_token_avg_ps = per_token_probs.pow(self.args.eubo_beta) * perturb_mask_expanded * completion_mask_expanded / empirical_t_expanded.clamp(min=1e-8)
            per_token_avg_ps = per_token_avg_ps.mean(dim=2) # [num_iterations, batch_size, logits_to_keep]
            # set all zero values to 1
            per_token_avg_ps_dezero = per_token_avg_ps.clone()
            per_token_avg_ps_dezero[per_token_avg_ps_dezero == 0] = 1
            loss_mask = (per_token_avg_ps > 0).bool()
        
        reward_mask_expanded = reward_mask.unsqueeze(0).expand(num_iterations, -1)
        per_seq_logps_positive = per_seq_logps.mean(dim=2) # [num_iterations, batch_size]
        if self.args.logp_estimation == 'eubo':
            per_seq_logps_negative = (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / self.args.eubo_beta
        elif self.args.logp_estimation == 'mix':
            per_seq_logps_negative = self.args.mix_weight * (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / self.args.eubo_beta + (1-self.args.mix_weight) * per_seq_logps.mean(dim=2)
        elif self.args.logp_estimation == 'elbo':
            per_seq_logps_negative = per_seq_logps.mean(dim=2)
        elif self.args.logp_estimation == 'zero':
            per_seq_logps_negative = torch.zeros_like(per_seq_logps_positive)
        else:
            raise ValueError(f"logp_estimation: {self.args.logp_estimation} is not supported")

        per_seq_logps = reward_mask_expanded.float() * per_seq_logps_positive + (~reward_mask_expanded).float() * per_seq_logps_negative
        assert per_seq_logps.shape == (num_iterations, batch_size), f"per_seq_logps.shape: {per_seq_logps.shape}"

        return per_seq_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.
                batch_prompt_completion_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                    prompt_mask=batch_prompt_mask,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token (but keep the first EOS token!)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

       
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        reward_mask = (advantages[process_slice] > 0).bool()
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )

            if self.beta == 0.0:
                all_ref_per_seq_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_seq_logps = self._get_per_seq_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, prompt_mask, completion_mask, reward_mask#, trajectory_samples
                    )
                    all_ref_per_seq_logps = ref_per_seq_logps

        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_seq_logps": all_ref_per_seq_logps, # num_iterations, batch_size
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
            "reward_mask": reward_mask,
        }
