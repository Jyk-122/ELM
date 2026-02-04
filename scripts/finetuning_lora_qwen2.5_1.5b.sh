#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
DATASET_PATH=../datasets
OUTPUT_PATH=./output


torchrun --nproc_per_node 8 --master_port 9600 finetuning_lora.py \
    --model_save_name Qwen2.5-1.5B-Instruct-Lora \
    --hf_model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --dataset_name metamathqa \
    --data_type instruction \
    --prompt_type qwen \
    --max_seq_len 2048 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --batch_size 1 \
    --accum_iter 1 \
    --epochs 3 \
    --warmup_epochs 0.3 \
    --save_freq 1 \
    --lr 2e-4 \
    --min_lr 2e-6 \
    --clip_grad 0.1 \
    --weight_decay 0.02 \
    --output_dir $OUTPUT_PATH \
    # --unittest 100