#!/bin/bash
pkill -f /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py

# cd llama_factory dir
cd /apdcephfs_qy3/share_301372554/share_info/ruihanyang/LLaMA-Factory
echo "Current directory $(pwd)"
# setup CUDA deivses

# setup CUDA deivses
export CUDA_VISIBLE_DEVICES=0,1,2,3

llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/models/uncertain-llama3-8b-sft-cutoff-2" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset uncertain-dpo-llama3-20000 \
    --cutoff_len 1024 \
    --learning_rate 1e-05 \
    --num_train_epochs 3.0 \
    --max_samples 20000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/models/uncertain-llama3-8b-1213-dpo-cutoff-2-bs64-ds20000-epoch3-lr1e-5 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target "q_proj","v_proj" \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 2


num_gpus=$(nvidia-smi --list-gpus | wc -l)

utilization=${1:-90}

for (( i=0; i<num_gpus; i++ )); do
    echo "occupying gpu $i"
    python3 /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py \
        --gpu_id=$i \
        --utilization=$utilization &
done