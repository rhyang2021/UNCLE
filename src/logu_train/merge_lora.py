from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
import torch
import random
from peft import get_peft_model,PeftModel
    
random.seed(42)
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup")
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/src")


max_gpu_memory = 39
start_id = 0
num_gpus = 4
bs_model = "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
lora_path = "/apdcephfs_qy3/share_301372554/share_info/ruihanyang/GenRM/models/actor-llama3-8b-sft-mix-0113-lr2e5-bs32-epoch3/checkpoint-2370"

kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{bs_model}/offload"}
kwargs.update({
    "device_map": "auto",
    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
})
print(kwargs)

tokenizer = AutoTokenizer.from_pretrained(bs_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
     
model = AutoModelForCausalLM.from_pretrained(bs_model, 
                                             low_cpu_mem_usage=True, 
                                             trust_remote_code=True, 
                                             **kwargs)
# load lora weight
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
print(model)
model.save_pretrained("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/data/actor-llama3-8b-sft-mix-0113")
tokenizer.save_pretrained("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/data/actor-llama3-8b-sft-mix-0113")


'''
mistral:
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-mistral-7b-0904-bs16-epoch10-lr5e5/checkpoint-800"
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-mistral-7b-0915-cutoff-bs16-epoch3-lr5e-5/checkpoint-2400"
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-mistral-7b-0917-filter-bs16-epoch5-lr5e-5/checkpoint-4000"
llama3:
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-0905-bs16-epoch10-lr5e5/checkpoint-2230"
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-0915-cutoff-bs16-epoch3-lr5e-5/checkpoint-2532"
lora_path = "/apdcephfs_qy3/share_733425/timhuang/rhyang/long_uncertainty_express/models/uncertain-llama3-8b-0917-filter-bs16-epoch5-lr5e-5/checkpoint-4000"
'''