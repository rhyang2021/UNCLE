import os
import pdb
import sys
from llm_base import llm_azure, llm_gpt
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
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/constrained-logu/data")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument('--repeat', type=int, default=10, help="number of sampling")
    
    
    args = parser.parse_args()
    

    with open(f"{args.input_dir}/all_domains/constrained_{args.dataset}.json") as f:
        data = json.load(f)
    data = data[:120]
    
    temperature = 1 if args.repeat > 0 else 0

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
                answer = llm_gpt(prompt=question, model=args.model_id)
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
        