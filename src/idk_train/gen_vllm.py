import os
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup")

class Agent(object):
    def __init__(self,
                model_id="/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct",
                temperature=1.0,
                num_generations=1,
                top_p=0.9,
                max_tokens=512,
                repetition_penalty=1,
                parallel_size=2,
                ):
        
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.num_generations = num_generations
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.parallel_size = parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, 
                       tensor_parallel_size=parallel_size)
        self.sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True
            )
        
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        response = self.llm.generate(text, self.sampling_params)
        return response[0].outputs[0].text

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID for LLM")
    parser.add_argument("--parallel_size", type=int, default=8, help="number of GPUs")
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_data")
    parser.add_argument("--output_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_data")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    
    with open(f"{args.input_dir}/triviaqa_{args.split}.json", "r") as f:
        data = json.load(f)
    
    if 'llama' in args.model_id:
        model_name = "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    elif 'mistral' in args.model_id:
        model_name = "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"

    agent = Agent(
        model_id=model_name,
        parallel_size=args.parallel_size
    )
    
    outputs = []
    for item in tqdm(data):
        prompt = item["question"]
        answers = []
        for _ in range(10):
            answer = agent.generate(prompt=prompt)
            answers.append(answer)
        new_item = {
            "question_id": item["question_id"],
            "question": prompt,
            "answer_ground_truth": item["answer_ground_truth"],
            "generated_answer": answers
        }
        outputs.append(new_item)
        
        output_dir = f'{args.output_dir}/{args.model_id}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f'{output_dir}/triviaqa_{args.split}_tp1.0_10_responses.jsonl'
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
    