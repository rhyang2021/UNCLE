import os
import json
from tqdm import tqdm
import argparse
from openai import OpenAI
import time
import pdb
from prompt_base import INSTUCT_FACTCHECK_LONG, INSTUCT_FACTCHECK_SHORT
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
    total_input_tokens = sum(len(tokenizer.encode(''.join(item["answers"]))) for item in data)
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

def llm_gpt(prompt: str, model: str):
    """Get completion from the GPT model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages =[{"role": "user", "content": prompt}],
                temperature=0,
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
            return [json.loads(line) for line in f]

    def save_data(file_name, data):
        with open(file_name, "a") as f:
            f.write(json.dumps(data) + "\n")

    def process_long_answers(answers, qa_pairs):    
        veracity_labels = []
        for answer in tqdm(answers, desc="repeat"):
            try:
                completion = llm_gpt(prompt=INSTUCT_FACTCHECK_LONG.format(paragraph=answer, qa_pairs=qa_pairs), model="gpt-4o")
                # completion = llm_azure(prompt=INSTUCT_FACTCHECK_LONG.format(paragraph=answer, qa_pairs=qa_pairs))
                atomic_responses = [x.strip() for x in completion.split("### ") if x]
                gpt_labels = [response.split("$")[1] for response in atomic_responses]
                assert len(atomic_facts) == len(gpt_labels)
            except Exception as e:
                print(f"Error: {e}. Skipping...")
                gpt_labels = []
            veracity_labels.append(gpt_labels)
        return veracity_labels
    
    def process_short_answers(atomic_facts_strings):
        veracity_labels = []
        for atomic_facts_string in tqdm(atomic_facts_strings, desc="repeat"):
            try:
                # completion = llm_azure(prompt=INSTUCT_FACTCHECK_SHORT.format(atomic_facts_string=atomic_facts_string))
                completion = llm_gpt(prompt=INSTUCT_FACTCHECK_SHORT.format(atomic_facts_string=atomic_facts_string),
                                    model="gpt-4o")
                atomic_responses = [x.strip() for x in completion.split("### ") if x]
                gpt_labels = [response.split("$")[1] for response in atomic_responses]
                assert len(atomic_facts) == len(gpt_labels)
            except Exception as e:
                print(f"Error: {e}. Skipping...")
                gpt_labels = []
            veracity_labels.append(gpt_labels)
        return veracity_labels

    data = load_data(generation_file_name)

    if args.debug:
        data = data[:2]

    estimated_cost = estimate_overall_cost(data)
    print(f"Estimated overall cost: ${estimated_cost:.6f}")

    output_file_name = f"../results/{args.dataset}/{args.model_name}_{args.method}_facts_veracity.jsonl"
    
    if os.path.exists(output_file_name):
        processed_items = load_data(output_file_name)
        processed_prompts = {item["prompt"] for item in processed_items}
        data = [item for item in data if item["prompt"] not in processed_prompts]
    
    print(f"Continue from {len(data)} items")
    
    new_data = []
    for item in tqdm(data, desc=f"{args.model_name} {args.method}"):
        individual_qa = item["individual_qa"]
        individual_questions = [i["question"] for i in individual_qa]
        individual_gold_answers = [i["answer"] for i in individual_qa]
        answers = item["answers"]
        atomic_facts = [f"Question: {question} Gold answer: {', '.join(gold_answer)}." 
                        for question, gold_answer in zip(individual_questions, individual_gold_answers)]
        qa_pairs = "### " + "\n### ".join(atomic_facts) + "\n\n"

        veracity_labels = process_long_answers(answers, qa_pairs)
        item.update({"veracity_labels": veracity_labels})
        
        individual_answers = item.get('individual_answers', [])
        if individual_answers:
            atomic_facts_strings = []
            for answers in individual_answers:   
                if len(answers) != len(individual_questions):
                    print("Mismatch in number of answers and questions. Skipping...")
                    continue  
                atomic_facts = [f"Question: {question} Model answer: {model_answer} Gold answer: {', '.join(gold_answer)}." 
                        for question, model_answer, gold_answer in zip(individual_questions, answers, individual_gold_answers)]
                atomic_facts_strings.append("### " + "\n### ".join(atomic_facts) + "\n\n")
            individual_veracity_labels = process_short_answers(atomic_facts_strings)
            item.update({"individual_veracity_labels": individual_veracity_labels})
        new_data.append(item)
        save_data(output_file_name, item)


def main(args):
    generation_file_name = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup/results/{args.dataset}/{args.model_name}_{args.method}_answers.jsonl"
    factchecker(args, generation_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b", help="Model name")
    parser.add_argument("--dataset", type=str, default="bios", help="Model name")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    parser.add_argument('--method', type=str, default="zero")
    args = parser.parse_args()
    main(args)
