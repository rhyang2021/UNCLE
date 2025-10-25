import os
import json
import argparse
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup")
from luq_vllm import LUQ_vllm
from dis_vllm import DIS_vllm
import numpy as np
from tqdm import tqdm
import time


def setup_environment():
    os.environ['HF_HOME'] = '/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models'
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_TOKEN"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
        return data

        
def main(args):
    setup_environment()

    if args.confidence_type == "generative":
        assert args.confidence_method in ["binary", "multiclass"]
    elif args.confidence_type == "discriminative":
        assert args.confidence_method in ["single", "context", "rating"]
    else:
        raise ValueError(f"Unsupported confidence type: {args.confidence_type}")

    atomic_file_path = f"../sft_data/1205/{args.dataset}/{args.model_name}_zero_atomic_facts.jsonl"

    atomic_facts = read_jsonl_file(atomic_file_path)

    samples_file_path = f"../sft_data/1205/{args.dataset}/{args.model_name}_zero_samples.jsonl"

    samples = read_jsonl_file(samples_file_path)
    # Sometimes we only use part of the data in fact checking (e.g. 500)
    samples = samples[:len(atomic_facts)]

    if args.debug:
        print("Debug mode.")
        atomic_facts = atomic_facts[:2]
        samples = samples[:2]

    save_file = f"../sft_data/1205/{args.dataset}/{args.model_name}_{args.confidence_type}_confidence_{args.confidence_method}.jsonl"

    if os.path.exists(save_file) and not args.overwrite:
        print(f"File {save_file} already exists.")
        exit()
    
    # Initialize the uncertainty calculator
    if args.confidence_type == "generative":
        luq_vllm = LUQ_vllm(nli_model="llama3-8b-instruct", method=args.confidence_method, gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)

    if args.confidence_type == "discriminative":
        dis_vllm = DIS_vllm(model=args.model_name, method=args.confidence_method, gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)

    results_to_save = []
    for item, item_samples in tqdm(zip(atomic_facts, samples), total=len(atomic_facts)):
        # First calculate the generative uncertainty
        try:
            assert item["prompt"] == item_samples['prompt']
        except AssertionError:
            print(f"Assertion failed: \n {item['prompt']} \n {item_samples['prompt']}")
        prompt = item["prompt"]
        answer = item["answer"]
        atomic_response = item["atomic_facts"]
        samples = item_samples['responses']

        if args.confidence_type == "generative":

            confidence_scores, raw_scores = luq_vllm.predict(
                sentences=atomic_response,              
                sampled_passages=samples,
            )
        elif args.confidence_type == "discriminative":
            confidence_scores, raw_scores = dis_vllm.predict(
                context=prompt,
                targets=atomic_response,
            )

        item["confidence_scores"] = confidence_scores.tolist()
        item["raw_scores"] = raw_scores.tolist()
        results_to_save.append(item)
        
        time.sleep(2)
    # Save as jsonl
    with open(save_file, "w") as f:
        for item in results_to_save:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files and generate uncertainty charts.")
    parser.add_argument('--confidence_type', choices=['generative', 'discriminative'], help='Type of confidence to calculate.')
    parser.add_argument('--confidence_method', help='Generative method to use.')
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--dataset', choices=['bio', 'longfact', 'wild'], help='Dataset to use.')
    parser.add_argument('--cuda_devices', type=str, default="0,1,2,3", help='CUDA devices to use.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory utilization.')
    parser.add_argument('--debug', action='store_true', help='Debug mode (no saving results).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    main(args)