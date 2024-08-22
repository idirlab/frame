#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

TASK_NAME="fn1.7"  # or fn1.7
OUTPUT_DIR="models/model_""$TASK_NAME"
DATA_DIR="data/$TASK_NAME"

python3 main.py \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--max_choice 15 \
--max_seq_length 300 \
--encode_type ludef_fndef \
--per_gpu_eval_batch_size 4 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--overwrite_output \
--overwrite_cache
