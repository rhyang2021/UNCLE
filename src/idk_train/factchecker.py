import os
import json
from tqdm import tqdm
import argparse
from openai import OpenAI
import time
import pdb
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup")
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/src")
from prompt_base import INSTUCT_FACTCHECK_SHORT
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer

# Cost per million tokens for gpt-4-1106-preview model
INPUT_COST_PER_MILLION = 5.00  # USD
OUTPUT_COST_PER_MILLION = 15.00  # USD


def estimate_cost(input_tokens, output_tokens):
    print("Cost for gpt-4o:")
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    return total_cost

def estimate_overall_cost(data, average_output_length=200):
    tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
    total_input_tokens = sum(len(tokenizer.encode(''.join(item["generated_answer"]))) for item in data)
    total_output_tokens = total_input_tokens*1.5
    total_cost = estimate_cost(total_input_tokens, total_output_tokens)
    return total_cost

@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(10))
def llm_azure(prompt: str):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01"
        )
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # model = "deployment_name".
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

def factchecker(args, generation_file_name):
    def load_data(file_name):
        with open(file_name, "r") as f:
            return json.load(f)

    def save_data(file_name, data):
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def process_answers(question, generate_answer_string, ground_truth_string):
        veracity_labels = []
        try:
            completion = llm_azure(INSTUCT_FACTCHECK_SHORT.format(question=question, passage=ground_truth_string, 
                                                                  atomic_facts_string=generate_answer_string))
            atomic_responses = [x.strip() for x in completion.split("### ") if '$$' in x]
            veracity_labels = [response.split("$$")[1] for response in atomic_responses]
        except Exception as e:
            print(f"Error: {e}. Skipping...")
            veracity_labels = []
        return veracity_labels

    data = load_data(generation_file_name)

    if args.debug:
        data = data[:2]

    estimated_cost = estimate_overall_cost(data)
    print(f"Estimated overall cost: ${estimated_cost:.6f}")

    output_file_name = f"/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_datasets/{args.model_name}/triviaqa_train_tp1.0_10responses_with_em_labels.json"
    
    if os.path.exists(output_file_name):
        processed_items = load_data(output_file_name)
        processed_prompts = {item["question_id"] for item in processed_items}
        data = [item for item in data if item["question_id"] not in processed_prompts]
    
    print(f"Continue from {len(data)} items")
    
    new_data = []
    for item in tqdm(data, desc=f"{args.model_name}"):
        generated_answers = item["generated_answer"]
        answer_ground_truth = item['answer_ground_truth']
        generated_answer_string = "### " + "\n### ".join(generated_answers) + "\n\n"
        ground_truth_string = "### " + "\n### ".join(answer_ground_truth) + "\n\n"

        veracity_labels = process_answers(item["question"], generated_answer_string, ground_truth_string)
        if not len(veracity_labels) == len(generated_answers):
            continue
        outputs = []
        for veracity_label, answer in zip(veracity_labels, generated_answers):
            outputs.append({
                "generated_answer": answer,
                "True_or_False": "True" if veracity_label == "S" else "False"
            })
        new_item = {
            "question_id": item['question_id'],
            "question": item["question"],
            "answer_ground_truth": item["answer_ground_truth"],
            "generated_answer": outputs
        }
        new_data.append(new_item)
        save_data(output_file_name, new_data)


def main(args):
    generation_file_name = f"/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_datasets/{args.model_name}/triviaqa_train_tp1.0_10_responses.json"
    factchecker(args, generation_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b", help="Model name")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    args = parser.parse_args()
    main(args)
