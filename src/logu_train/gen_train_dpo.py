import pandas as pd
import pdb
pd.options.mode.chained_assignment = None
from transformers import AutoTokenizer

instruction_pool={
    "bio": f"Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "wild": f"Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "longfact": f"Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
}


def get_template(dataset, label):
    
    output = []
    for data in dataset:
        for prompt in data["prompt"]:
            for refine_answer in data['refine_answer']:
                output.append(
                    {"instruction": prompt,
                     "input": "",
                    "chosen": refine_answer,
                    "rejected": data['answer']
                    })
            for amateur_answer in data['amateur_answer']:
                output.append(
                    {"instruction": prompt,
                     "input": "",
                    "chosen": data['answer'],
                    "rejected": amateur_answer
                    } 
                )
            for refine_answer in data['refine_answer']:
                for amateur_answer in data['amateur_answer']:
                    output.append(
                        {"instruction": prompt,
                        "input": "",
                        "chosen": refine_answer,
                        "rejected": amateur_answer
                        } 
                    )

    return output


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="generate closed set fasts for each entity in bio dataset.")
    parser.add_argument("--model_id", type=str, default="llama3-8b", help="Model ID for generating")
    parser.add_argument("--method", type=str, default="dpo-cutoff-2", help="Model ID for generating")
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data", help="Model ID for generating")
    parser.add_argument("--output_dir", type=str, default="/apdcephfs_qy3/share_301372554/share_info/ruihanyang/LoGU-followup/sft_data", help="Model ID for generating")
    args = parser.parse_args()

    final_data = []
    for label in ['bio', 'wild', 'longfact']:
        with open(f'{args.input_dir}/{label}/{args.model_id}_train_{args.method}.json') as f:
            dataset = json.load(f)
        
        final_data += get_template(dataset, label)
        print(len(final_data))

    sample_data = random.sample(final_data, 20000)
    print(len(sample_data))
    with open(f"{args.output_dir}/uncertain_{args.method}_{args.model_id}-sample20000.json", 'w') as f:
        json.dump(sample_data, f, indent=4)