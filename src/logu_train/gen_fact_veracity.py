import os
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup")
from prompt_base import BIO_GEN_TEMPLATE, WILD_GEN_TEMPLATE, BIO_BIAS_GEN_TEMPLATE, INSTRUCT_REFINE_UNCERTAIN
from utils import read_jsonl


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    import numpy as np
    
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID for LLM")
    parser.add_argument("--parallel_size", type=int, default=8, help="number of GPUs")
    parser.add_argument("--dataset", type=str, default="bio", help="dataset")
    parser.add_argument("--input_dir", type=str, default="../sft_data/1205")
    parser.add_argument("--output_dir", type=str, default="../sft_data/1205")
    parser.add_argument('--method', type=str, default="zero")
    parser.add_argument('--threshold', type=int, default=10)
    args = parser.parse_args()
    
    with open(f"{args.input_dir}/{args.dataset}/{args.model_id}_generative_confidence_binary.jsonl") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    confidence_scores = []
    for line in data:
        confidence_scores.extend(line["confidence_scores"])

    score_threshold = np.percentile(confidence_scores, args.threshold)
    print(score_threshold)
    for line in data:
        assert(len(line["confidence_scores"]) == len(line["atomic_facts"]))
        atomic_facts_veracity = []
        for atomic_fact, confidence_score in zip(line["atomic_facts"], line["confidence_scores"]):
            if confidence_score < score_threshold:
                atomic_facts_veracity.append("NS")
            else:
                atomic_facts_veracity.append("S")
        
        with open(f"{args.output_dir}/{args.dataset}/{args.model_id}_{args.method}_atomic_facts_veracity.jsonl", "a") as f:
            f.write(json.dumps({"topic": line["topic"],
                                "prompt": line["prompt"],
                                "answer": line["answer"],
                                "atomic_facts": line["atomic_facts"],
                                "raw_atomic_facts": line["raw_atomic_facts"],
                                "confidence_scores": line["confidence_scores"],
                                "raw_scores": line["raw_scores"],
                                "atomic_facts_veracity": atomic_facts_veracity
                                }) + '\n')
        