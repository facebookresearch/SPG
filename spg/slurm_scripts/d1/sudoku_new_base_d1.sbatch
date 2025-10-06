#!/bin/bash
#SBATCH --output=../../slurm/%A/slurm.out
#SBATCH --job-name=sudoku-new-d1
#SBATCH --time=120:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --qos=storygen_high
#SBATCH --account=storygen
#SBATCH --cpus-per-task=50

export LOGDIR=../../logs
mkdir -p $LOGDIR
echo $LOGDIR

source activate spg

SAVE_DIR=/fsx-checkpoints/

NUM_ITER=4
FEW_SHOT=3
DATASET="sudoku_new"
RUN_NAME=${DATASET}_${FEW_SHOT}shot_base_d1
MODEL_PATH=${SAVE_DIR}/hf_models/LLaDA-8B-Instruct

srun --output ${LOGDIR}/d1_%j.out \
    accelerate launch \
        --config_file ../accelerate_genai_a100.yaml \
        --main_process_port 12346 ../../diffu_grpo_train.py \
        --config ../train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --trainer diffu_grpo \
        --num_generations 6 \
        --per_device_train_batch_size 3 \
        --gradient_accumulation_steps 4 \
        --output_dir ${SAVE_DIR}/spg/$RUN_NAME \
        --temperature 0.3 \
        --few_shot $FEW_SHOT \
        --max_prompt_length 1500