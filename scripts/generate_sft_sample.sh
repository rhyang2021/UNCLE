#!/bin/bash
BASE_DIR=$(cd .. && pwd)


export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models"


MODEL_NAMES=("mistral-7b")
DATASET_NAMES=("bio" "longfact" "wild")
METHODS=("sft-cutoff-2")
THRE=("2")
# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            
            echo "Processing model: $MODEL_NAME"
            echo "Processing dataset: $DATASET_NAME"
            echo "Processing method: $METHOD"
            FILE_TO_CHECK="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data/${DATASET_NAME}/${MODEL_NAME}_train_${METHOD}.json"
            if [ ! -f "$FILE_TO_CHECK" ]; then
                echo "Save to $FILE_TO_CHECK."
                python ${BASE_DIR}/src/gen_sft_samples.py \
                    --dataset $DATASET_NAME \
                    --model_id $MODEL_NAME \
                    --method $METHOD \
                    --threshold $THRE \
                    --input_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data/1205" \
                    --output_dir "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data"
            else
                echo "File $FILE_TO_CHECK already exists. Skipping the command."
            fi
            sleep 1
        done
    done
done
