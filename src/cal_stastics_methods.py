import numpy as np
import json
import pandas as pd
from sklearn.metrics import confusion_matrix




if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup/results")
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--method_1", type=str, default="repeat1_idk-dpo")
    parser.add_argument("--method_2", type=str, default="repeat1_mix-dpo")
    args = parser.parse_args()

    veracity_1 = []
    veracity_2 = []
    for dataset in ['companies','diseases', 'movies', 'planets']:
        with open(f"{args.input_dir}/{dataset}/{args.model_name}_{args.method_1}_long-short_facts_veracity.jsonl") as f:
            results_1 = []
            for line in f:
                results_1.append(json.loads(line))

        with open(f"{args.input_dir}/{dataset}/{args.model_name}_{args.method_2}_long-short_facts_veracity.jsonl") as f:
            results_2 = []
            for line in f:
                results_2.append(json.loads(line))

        for item_1, item_2 in zip(results_1, results_2):
            if len(item_1['veracity_labels'][0]) == len(item_2['veracity_labels'][0]):
                veracity_1.extend(item_1['veracity_labels'][0])
                veracity_2.extend(item_2['veracity_labels'][0])


    # 创建 DataFrame
    df = pd.DataFrame({'A': veracity_1, 'B': veracity_2})

    # 计算混淆矩阵
    cm = confusion_matrix(df['A'], df['B'], labels=['S', 'NS', "UNC"])

    # 将混淆矩阵转换为 DataFrame
    cm_df = pd.DataFrame(cm, index=['S', 'NS', "UNC"], columns=['S', 'NS', "UNC"])
    cm_percentage = (cm / cm.sum().sum() * 100).round(2)
    cm_df = pd.DataFrame(cm_percentage, index=['S', 'NS', "UNC"], columns=['S', 'NS', "UNC"])
    print(cm_df)