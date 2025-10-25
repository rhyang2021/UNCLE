import os
import pdb
import sys
from llm_base import llm_azure, llm_gpt, llm_deepseek, llm_claude
sys.path.append("/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup")
from utils import read_jsonl

unc_instruction = f"You should express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."

prompt_template = ["In a paragraph", "In 100 words", "In 150 words", "In 200 words", "In 250 words", "In 300 words"]

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
    parser.add_argument("--model_id", type=str, default="gpt-4o", help="Model ID for LLM")
    parser.add_argument("--method", type=str, default="zero", help="reasoning method")
    parser.add_argument("--dataset", type=str, default='bios', help="number of GPUs")
    parser.add_argument("--parallel_size", type=int, default=2, help="number of GPUs")
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/constrained-logu/data")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument('--repeat', type=int, default=10, help="number of sampling")
    
    
    args = parser.parse_args()
    
    with open(f"{args.input_dir}/all_domains/constrained_{args.dataset}.json") as f:
        data = json.load(f)
    data = data[:120]
    
    temperature = 0.7 if args.repeat > 0 else 0
    
    output_dir = f'{args.output_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{args.model_id}_repeat{args.repeat}_{args.method}_long-short_answers.jsonl'
    
    if os.path.exists(output_file):
        processed_items = load_data(output_file)
        processed_prompts = {item["prompt"] for item in processed_items}
        data = [item for item in data if item["prompt"] not in processed_prompts]
    
    for item in tqdm(data, desc="entity"):
        _long_answers = []
        long_prompt = item["prompt"]
        
        for _ in range(args.repeat):
            user_prompt = long_prompt+unc_instruction if "method" == "unc-zero" else long_prompt
            if args.model_id == "gpt-4o":
                long_answer = llm_azure(prompt=user_prompt)
            elif args.model_id == "deepseek-chat":
                long_answer = llm_deepseek(prompt=user_prompt, model="deepseek-chat", temperature=temperature)
            elif args.model_id.startswith("claude"):
                long_answer = llm_claude(prompt=user_prompt, temperature=temperature)
            else:
                long_answer = llm_gpt(prompt=user_prompt, model=args.model_id)
            _long_answers.append(long_answer)
        
        individual_prompts = [i["question"] for i in item['individual_qa']]
        _short_answers = []
        for _ in range(args.repeat):
            short_answers = []
            for question in individual_prompts:
                if "method" == "unc-zero":
                    question = question + " You should express uncertainty for any question you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
                if args.model_id == "gpt-4o":
                    answer = llm_azure(prompt=question)
                elif args.model_id == "deepseek-chat":
                    answer = llm_deepseek(prompt=question, model="deepseek-chat", temperature=temperature)
                elif args.model_id.startswith("claude"):
                    answer = llm_claude(prompt=question, temperature=temperature)
                else:
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
                                'answers': _long_answers
                                }) + '\n')
        