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
    parser.add_argument("--method", type=str, default="repeat1_unc-zero")
    args = parser.parse_args()

    long_known = []
    short_known = []
    long_veracity, short_veracity = [], []

    for dataset in ['bios']:
        with open(f"{args.input_dir}/bios/{args.model_name}_{args.method}_long-short_facts_veracity.jsonl") as f:
            results = []
            for line in f:
                results.append(json.loads(line))


        for item in results:
            long_known.append(item['veracity_labels'][0].count("S")/len(item["veracity_labels"][0]) if item["veracity_labels"][0] else 0)
            short_known.append(item['individual_veracity_labels'][0].count("S")/len(item["individual_veracity_labels"][0]) if item["individual_veracity_labels"][0] else 0)
            
        print(np.mean(long_known), np.mean(short_known))

        for item in results:
            if len(item['veracity_labels'][0]) == len(item['individual_veracity_labels'][0]):
                long_veracity.extend(item['veracity_labels'][0])
                short_veracity.extend(item['individual_veracity_labels'][0])


    # 创建 DataFrame
    df = pd.DataFrame({'A': short_veracity, 'B': long_veracity})

    # 计算混淆矩阵
    cm = confusion_matrix(df['A'], df['B'], labels=['S', 'NS', "UNC"])

    # 将混淆矩阵转换为 DataFrame
    cm_df = pd.DataFrame(cm, index=['S', 'NS', "UNC"], columns=['S', 'NS', "UNC"])
    # 将混淆矩阵转换为百分数并创建 DataFrame
    cm_percentage = (cm / cm.sum().sum() * 100).round(2)
    cm_df = pd.DataFrame(cm_percentage, index=['S', 'NS', "UNC"], columns=['S', 'NS', "UNC"])
    c_c = cm_df['S']['S']+cm_df['NS']['NS']+cm_df['S']['NS'] + cm_df['NS']['S']
    unc_unc = cm_df['UNC']['UNC']
    print(cm_df)
    print(c_c)
    print(unc_unc)