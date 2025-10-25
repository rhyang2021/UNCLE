#!/bin/bash
pkill -f /apdcephfs_cq10/share_1567347/share_info/ruihanyang/occupy_gpu.py
BASE_DIR=$(cd .. && pwd)
export AZURE_OPENAI_ENDPOINT=""
export AZURE_OPENAI_API_KEY=""
export OPENAI_API_KEY=""

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/apdcephfs_cq11/share_1567347/share_info/rhyang/huggingface_models"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASETS=("companies" "diseases" "movies" "planets" "bios")
# MODEL_NAMES=("llama3-8b" "mistral-7b" "qwen2-7b" "llama3-70b" "mistral-8x7b" "qwen2-72b")
# MODEL_NAMES=("gpt-4o")
MODEL_NAMES=("gpt-4-1106-preview" "gpt-3.5-turbo-1106")
REPEATS=("1")
METHODS=("zero")
# exec > ../logs/generate_responses_0826_2.log 2>&1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for REPEAT in "${REPEATS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                echo "Processing model: $MODEL_NAME"
                echo "Processing dataset: $DATASET"

                FILE_TO_CHECK="$BASE_DIR/results/${DATASET}/${MODEL_NAME}_repeat${REPEAT}_${METHOD}_knowledge_eval_answers.jsonl"
                if [ ! -f "$FILE_TO_CHECK" ]; then
                    echo "Save to $FILE_TO_CHECK."
                    python ../src/gen_knowledge_eval_openai.py \
                        --model_id $MODEL_NAME \
                        --repeat $REPEAT \
                        --parallel_size 4 \
                        --method $METHOD \
                        --dataset $DATASET \
                        --input_dir "/apdcephfs_cq11/share_1567347/share_info/rhyang/constrained-logu/data" \
                        --output_dir "/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup/results"
                else
                    echo "File $FILE_TO_CHECK already exists. Skipping the command."
                fi
                sleep 1
            done
        done
    done
done

num_gpus=$(nvidia-smi --list-gpus | wc -l)


utilization=${1:-90}


for (( i=0; i<num_gpus; i++ )); do
    echo "occupying gpu $i"
    python3 /apdcephfs_cq10/share_1567347/share_info/ruihanyang/occupy_gpu.py \
        --gpu_id=$i \
        --utilization=$utilization &
done