import os
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pdb
import sys
from llm_base import vllm_Agent
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup")
from utils import read_jsonl


def load_data(file_name):
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID for LLM")
    parser.add_argument("--dataset", type=str, default="bios", help="reasoning method")
    parser.add_argument("--method", type=str, default="zero", help="reasoning method")
    parser.add_argument("--parallel_size", type=int, default=2, help="number of GPUs")
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_cq11/share_1567347/share_info/rhyang/constrained-logu/data")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument('--repeat', type=int, default=10, help="number of sampling")
    
    
    args = parser.parse_args()
    
    if 'llama3-8b' in args.model_id:
        model_name = f"/apdcephfs_cq11/share_1567347/share_info/llm_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    elif "llama3-70b" in args.model_id:
        model_name = f"/apdcephfs_cq11/share_1567347/share_info/llm_models/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
    elif "mistral-7b" in args.model_id:
        model_name = f"/apdcephfs_cq10/share_1567347/share_info/ruihanyang/huggingface_models/Mistral-7B-Instruct-v0.2"
    elif "mistral-8x7b" in args.model_id:
        model_name = f"/apdcephfs_cq11/share_1567347/share_info/llm_models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1" 
    elif "qwen2-7b" in args.model_id:
        model_name = f"/apdcephfs_cq10/share_1567347/share_info/llm_models/Qwen2-7B-Instruct"   
    elif "qwen2-72b" in args.model_id:
        model_name = f"/apdcephfs_cq11/share_1567347/share_info/llm_models/models--Qwen--Qwen2-72B-Instruct/snapshots/c867f763ef53f2ea9d9b31ee8501273dedd391eb"
    
    with open(f"{args.input_dir}/all_domains/constrained_{args.dataset}.json") as f:
        data = json.load(f)
    data = data[:120]
    
    temperature = 1 if args.repeat > 0 else 0
    agent = vllm_Agent(
        model_id=model_name,
        parallel_size=args.parallel_size, 
        temperature=temperature
    )
    
    output_dir = f'{args.output_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{args.model_id}_repeat{args.repeat}_{args.method}_knowledge_eval_answers.jsonl'
    
    if os.path.exists(output_file):
        processed_items = load_data(output_file)
        processed_prompts = {item["prompt"] for item in processed_items}
        data = [item for item in data if item["prompt"] not in processed_prompts]
    
    for item in tqdm(data, desc="entity"):

        individual_prompts = [i["question"] for i in item['individual_qa']]
        _short_answers = []
        for _ in range(args.repeat):
            short_answers = []
            for question in individual_prompts:
                if "method" == "unc-zero":
                    question = question + " You should express uncertainty for any question you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
                answer = agent.generate(prompt=question)
                short_answers.append(answer)
            _short_answers.append(short_answers)
        
        with open(output_file, 'a') as f:
            f.write(json.dumps({"entity": item['entity'],
                                'prompt': item['prompt'],
                                'individual_qa': item['individual_qa'],
                                'properties': [i["description"] for i in item["individual_qa"]],
                                'gold_answers': [i["answer"] for i in item["individual_qa"]],
                                'individual_answers': _short_answers,
                                'answers': [],
                                }) + '\n')
        