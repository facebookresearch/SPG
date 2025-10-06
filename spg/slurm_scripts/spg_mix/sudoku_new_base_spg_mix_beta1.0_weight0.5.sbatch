#!/bin/bash
#SBATCH --output=../../slurm/%A/slurm.out
#SBATCH --job-name=sudoku-new-mix
#SBATCH --time=120:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --qos=storygen_high
#SBATCH --account=storygen
#SBATCH --cpus-per-task=50

export LOGDIR=../../logs
mkdir -p $LOGDIR
echo $LOGDIR

source activate spg
# Generate a random port number between 10000 and 65535
RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"

SAVE_DIR=/fsx-checkpoints/

DATASET="sudoku_new"
FEW_SHOT=3
RUN_NAME=${DATASET}_${FEW_SHOT}shot_base_spg_mix_beta1.0_weight0.5
MODEL_PATH=${SAVE_DIR}/hf_models/LLaDA-8B-Instruct
NUM_ITER=4
srun --cpu-bind=none --output ${LOGDIR}/spg_mix_%j.out \
    accelerate launch \
        --config_file ../accelerate_genai_a100.yaml \
        --main_process_port $RANDOM_PORT ../../diffu_grpo_train.py \
        --config ../train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir ${SAVE_DIR}/spg/$RUN_NAME \
        --trainer spg \
        --forward_type block_random \
        --num_t 2 \
        --min_t 0 \
        --max_t 1 \
        --num_generations 6 \
        --per_device_train_batch_size 3 \
        --gradient_accumulation_steps 4 \
        --beta 0.0 \
        --logp_estimation mix \
        --mix_weight 0.5 \
        --eubo_beta 1.0 \
        --temperature 0.3 \
        --few_shot $FEW_SHOT \
        --max_prompt_length 1500