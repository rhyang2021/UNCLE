from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from peft import get_peft_model,PeftModel
import torch
import json
from tqdm import tqdm
import argparse
import random
import pdb
random.seed(42)
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/long_uncertainty_express")


max_gpu_memory = 39
start_id = 0
num_gpus = 2

# lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-0917-filter-bs16-epoch5-lr5e-5/checkpoint-4000"
# bs_model = '/apdcephfs_qy3/share_733425/timhuang/huggingface_models/llama3-8b-instruct'
# lora_path = '/apdcephfs_qy3/share_301372554/share_info/ruihanyang/data/critic-llama3-8b-sft-1225-lr1e5-bs128-epoch3/checkpoint-3900'
bs_model = '/apdcephfs_cq10/share_1567347/share_info/ruihanyang/LongCoT/models/cot_llama3-8b_dpo_prefix_lr1e5_bs32_epoch3_full_0508'

kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{bs_model}/offload"}
kwargs.update({
    "device_map": "auto",
    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
})
print(kwargs)

# template = f"<s>[INST] {{}} [/INST]"
template = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

instruction_pool={
    "bio": f"Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "wild": f"Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "longfact": f"Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
}

tokenizer = AutoTokenizer.from_pretrained(bs_model, trust_remote_code=True)
tokenizer.padding_side = 'left'

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("/apdcephfs_cq10/share_1567347/share_info/ruihanyang/LongCoT/models/cot_llama3-8b_dpo_prefix_lr1e5_bs32_epoch3_full_0508", low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
# model = PeftModel.from_pretrained(model, lora_path)

# output_file = f"/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_data/test_answers.json"

# with open(f"/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/idk_data/triviaqa_train.json", "r") as f:
    # data = json.load(f)

# data = random.sample(data, 10000)
# prompts = [item['question'] for item in data]
prompts = ["What is the capital of France?", "Tell me a bio of Albert Einstein", "What is the largest mammal in the world?"]
messages = [template.format(prompt) for prompt in prompts]
# print(messages) 

inputs = tokenizer(messages,
                   return_tensors="pt",
                   add_special_tokens=False,
                   padding=True)

inputs.input_ids = inputs.input_ids.to(model.device)

batch_size = 64
input_ids = [inputs.input_ids[i:i+batch_size] for i in range(0, len(inputs.input_ids), batch_size)]
attention_masks = [inputs.attention_mask[i:i+batch_size] for i in range(0, len(inputs.attention_mask), batch_size)]

answers = []
model.eval()
with torch.no_grad():
    for input_id, attention_mask in tqdm(zip(input_ids, attention_masks), desc="Generating"):

        outputs = model.generate(input_id,
                                attention_mask=attention_mask, 
                                max_length=512,
                                top_p=0.9,
                                temperature=1,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                stopping_criteria=StoppingCriteriaList(),
                                # return_dict_in_generate=True, 
                                # output_scores=True
                                )
        
        # sequences = outputs.sequences
        # scores = outputs.scores
        for id, output in zip(input_id, outputs):
            answers.append(tokenizer.decode(output[len(id):], skip_special_tokens=True).strip())

assert(len(answers)==len(prompts))

outputs = []
for prompt, answer in zip(prompts, answers):
    print(answer)
    new_item = {
        # "question_id": item["question_id"],
        "question": prompt,
        # "answer_ground_truth": item["answer_ground_truth"],
        "generated_answer": answer
    }
    outputs.append(new_item)

with open(output_file, 'w') as f:
    json.dump(outputs, f, indent=4)
'''
with open(output_file, 'w') as f:
    for original_prompt, answer in zip(prompts, answers):
        print(original_prompt, answer)
        f.write(json.dumps({"topic": "",
                            "prompt": original_prompt,
                            "answer": answer}) + '\n')
'''