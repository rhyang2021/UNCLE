#!/bin/bash
pkill -f /apdcephfs_qy3/share_301372554/share_info/ruihanyang/occupy_gpu.py
BASE_DIR=$(cd .. && pwd)

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONFIDENCE_TYPE=generative
export CONFIDENCE_METHOD=binary

MODEL_NAMES=("mistral-7b")
# DATASET_NAMES=("bio" "longfact" "wild")
# METHODS=("unc-zero" "unc-few" "pair-few" "sft" "self-refine" "sft-cutoff-2" "sft-ablation-re" "dpo-cutoff-2-ds20000-epoch3" "dpo-ablation-re")
# exec > ../logs/generate_responses_0826_2.log 2>&1

DATASET_NAMES=("bio" "wild")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        
        echo "Processing model: $MODEL_NAME"
        echo "Processing dataset: $DATASET_NAME"

        FILE_TO_CHECK="$BASE_DIR/sft_data/1205/${DATASET_NAME}/${MODEL_NAME}_${CONFIDENCE_TYPE}_confidence_${CONFIDENCE_METHOD}.jsonl"
        if [ ! -f "$FILE_TO_CHECK" ]; then
            python ../src/cal_uncertainty.py \
                    --dataset $DATASET_NAME \
                    --model_name $MODEL_NAME \
                    --confidence_type $CONFIDENCE_TYPE \
                    --confidence_method $CONFIDENCE_METHOD 
        else
            echo "File $FILE_TO_CHECK already exists. Skipping the command."
        fi
        sleep 2
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