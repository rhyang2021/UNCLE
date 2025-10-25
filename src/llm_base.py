import os
import time
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from openai import AzureOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import json
import pdb
import requests
#  from vllm import LLM, SamplingParams

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


class vllm_Agent(object):
    def __init__(self,
                model_id="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/huggingface_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a",
                temperature=0.7,
                num_generations=1,
                top_p=0.9,
                max_tokens=512,
                parallel_size=2,
                ):
        
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.num_generations = num_generations
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.parallel_size = parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, 
                       tensor_parallel_size=parallel_size,
                       # gpu_memory_utilization=0.8,
                       dtype="half")
        self.sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # stop_token_ids=[self.tokenizer.eos_token_id],
            stop_token_ids=[],
            skip_special_tokens=True
            )
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        response = self.llm.generate(text, self.sampling_params)
        return response[0].outputs[0].text


def llm_gpt(prompt: str, model: str):
    """Get completion from the GPT model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

ROLE_DICT_ANTHROPIC = {'user': 'user', 'model': 'assistant'}
def send_conversation_request(api_key, conversation, model='claude-3-5-sonnet@20240620', temperature=0.2):
    """发送多轮对话请求，包含温度参数"""
    url = "http://9.208.232.38:8080/dialogue"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    if 'claude' in model:
        formatted_conversation = [
            {"role": ROLE_DICT_ANTHROPIC[entry["role"]], "content": [{"type": "text", "text": entry["parts"]["text"]}]}
            for entry in conversation
        ]
    else:
        formatted_conversation = [
            {"role": ROLE_DICT_GEMINI[entry["role"]], "parts": {"text": entry["parts"]["text"]}}
            for entry in conversation
        ]
    data = {
        "conversation": formatted_conversation,
        "model": model,
        "temperature": temperature
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()


def llm_claude(prompt, temperature):
    api_key = "ruotianma_key"
    temperature_value = temperature
    conversation_history=[({"role": "user","parts": {"text": prompt}})]
    result_multi = send_conversation_request(api_key, conversation_history, 'claude-3-5-sonnet@20240620', temperature=temperature_value)
    
    try:
        data = json.loads(result_multi['result']) 
        return data["content"][0]["text"]
    except:
        return ""
    
def llm_deepseek(prompt: str, model: str, temperature: float = 0.0):
    """Get completion from the GPT model."""
    client = OpenAI(api_key="sk-b134cd42fd8a4ae68ffb272a9e27f558", 
                    base_url="https://api.deepseek.com")
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature)
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None

