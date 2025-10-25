#!/bin/bash
export OPENAI_API_KEY=""
export AZURE_OPENAI_ENDPOINT=""
export AZURE_OPENAI_API_KEY=""

BASE_DIR=$(cd .. && pwd)

DATASETS=("companies" "diseases" "movies" "planets" "bios")
# MODEL_NAMES=("llama3-8b" "mistral-7b" "qwen2-7b" "llama3-70b" "mistral-8x7b" "qwen2-72b" "gpt-4o")
MODEL_NAMES=("deepseek-chat")
# METHODS=("repeat1_unc-zero_long-short" "repeat1_zero_knowledge_eval")
METHODS=("repeat1_zero_long-short")

# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "Processing model: $MODEL_NAME"
            echo "Processing method: $METHOD"

            FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_${METHOD}_facts_veracity.jsonl"
            echo $FILE_TO_CHECK
            # if [ ! -f "$FILE_TO_CHECK" ]; then
            python $BASE_DIR/src/factchecker.py \
                    --model_name $MODEL_NAME \
                    --dataset $DATASET \
                    --method $METHOD
            # else
                # echo "File $FILE_TO_CHECK already exists. Skipping the command."
            # fi
            sleep 1
        done 
    done
done