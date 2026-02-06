#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

PROJECT_PATH=/data1/jiangyikun/ELMs/Deq
HF_MODEL_PATH=/data1/jiangyikun/models/Qwen2.5-1.5B-Instruct
LORA_PATH=/data1/jiangyikun/ELMs/Lora/output
MASTER_PORT=9811

torchrun --nproc_per_node 1 --master_port $MASTER_PORT $PROJECT_PATH/eval.py \
    --dataset_name math \
    --dataset_path $PROJECT_PATH/../datasets/math \
    --hf_model_path $HF_MODEL_PATH \
    --lora_model_path $LORA_PATH/math-code/Qwen2.5-1.5B-Instruct-Lora_2025-11-19_01:40/ckpt/checkpoint-2.pth \
    --finetune_ckpt_path $PROJECT_PATH/output/math-code/Qwen2.5-1.5B-Instruct-ELM-Prune8-start2_2025-11-19_08:19/ckpt/checkpoint-3.pth \
    --model_args_path $PROJECT_PATH/output/math-code/Qwen2.5-1.5B-Instruct-ELM-Prune8-start2_2025-11-19_08:19/model_args.json \
    --max_seq_len 2048 \
    --max_gen_len 2048 \
    --max_batch_size 1 \
    --temperature 0.0 \
    --fixed_point_solver "simple" \
    --fixed_point_max_iters 8 \
    --fixed_point_ee_threshold 0.0 \