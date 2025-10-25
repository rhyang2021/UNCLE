#!/bin/bash
pkill -f /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py

# cd llama_factory dir
cd /apdcephfs_qy3/share_301372554/share_info/ruihanyang/LLaMA-Factory 
echo "Current directory $(pwd)"
# setup CUDA deivses
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/apdcephfs_qy3/share_301372554/share_info/ruihanyang/AgentGym

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset sci-critic-sft-mix \
    --cutoff_len 4096 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --max_samples 170000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir /apdcephfs_qy3/share_301372554/share_info/ruihanyang/data/critic-llama3-8b-sft-mix-0104-lr1e5-bs256-epoch3 \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --val_size 0.05 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 1



num_gpus=$(nvidia-smi --list-gpus | wc -l)


utilization=${1:-90}


for (( i=0; i<num_gpus; i++ )); do
    echo "occupying gpu $i"
    python3 /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py \
        --gpu_id=$i \
        --utilization=$utilization &
done


