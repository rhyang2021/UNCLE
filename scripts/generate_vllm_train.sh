#!/bin/bash
pkill -f /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py
BASE_DIR=$(cd .. && pwd)

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models"
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAMES=("mistral-7b")
DATASET_NAMES=("bio")
# METHODS=("zero" "unc-zero" "unc-few" "pair-few" "self-refine" "sft-ablation-re" "sft-cutoff-2" "dpo-ablation-re" "dpo-cutoff-2-ds20000-epoch3")
METHODS=("zero")
# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            
            echo "Processing model: $MODEL_NAME"
            echo "Processing dataset: $DATASET_NAME"
            echo "Processing method: $METHOD"

            FILE_TO_CHECK="$BASE_DIR/sft_data/1205/${DATASET_NAME}/${MODEL_NAME}_${METHOD}_answers.jsonl"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                echo "Save to $FILE_TO_CHECK."
                python ../src/gen_vllm.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --split train \
                    --parallel_size 4 \
                    --input_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/data" \
                    --output_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data/1205"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1
            
            FILE_TO_CHECK="$BASE_DIR/sft_data/1205/${DATASET_NAME}/${MODEL_NAME}_${METHOD}_samples.jsonl"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                echo "Save to $FILE_TO_CHECK."
                python ../src/gen_vllm_sampling.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --parallel_size 4 \
                    --split train \
                    --input_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/data" \
                    --output_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data/1205"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1

        done
    done
done

num_gpus=$(nvidia-smi --list-gpus | wc -l)


utilization=${1:-90}


for (( i=0; i<num_gpus; i++ )); do
    echo "occupying gpu $i"
    python3 /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py \
        --gpu_id=$i \
        --utilization=$utilization &
done