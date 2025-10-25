import numpy as np
import json
import pandas as pd
from sklearn.metrics import confusion_matrix

def get_veracity_labels(data):
    transposed_data = list(zip(*data))
    result = []

    for column in transposed_data:
        if column.count("S") == len(column):
            result.append("S")
        else:
            result.append("NS")
    return result

if __name__ == "__main__":
    import argparse
    import pdb

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/apdcephfs_cq11/share_1567347/share_info/rhyang/LoGU-followup/results")
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--dataset", type=str, default="bios")
    parser.add_argument("--method", type=str, default="repeat1_pair-few")
    args = parser.parse_args()
    
    knowledge_veracity_labels, veracity_labels = [], []
    for dataset in ['planets']:
        with open(f"{args.input_dir}/{dataset}/{args.model_name}_repeat1_zero_knowledge_eval_facts_veracity.jsonl") as f:
            knowledge_eval_results = []
            for line in f:
                knowledge_eval_results.append(json.loads(line))

        with open(f"{args.input_dir}/{dataset}/{args.model_name}_{args.method}_long-short_facts_veracity.jsonl") as f:
            long_short_results = []
            for line in f:
                long_short_results.append(json.loads(line))
        print(len(knowledge_eval_results), len(long_short_results))
        assert(len(knowledge_eval_results) == len(long_short_results))
        
        for knowledge_item, long_short_item in zip(knowledge_eval_results, long_short_results):
            long_short_veracity = long_short_item['veracity_labels'][0]
            knowledge_item_veracity = knowledge_item['individual_veracity_labels']
            if len(long_short_veracity) != len(knowledge_item_veracity[0]):
                continue
            # knowledge_item_veracity_labels = get_veracity_labels(knowledge_item_veracity)
            knowledge_item_veracity_labels = knowledge_item_veracity[0]
            if len(knowledge_item_veracity_labels) != len(long_short_veracity):
                continue
            knowledge_veracity_labels.extend(knowledge_item_veracity_labels)
            veracity_labels.extend(long_short_veracity)

    # 创建 DataFrame
    df = pd.DataFrame({'Knowledge Veracity': knowledge_veracity_labels, 'Veracity': veracity_labels})

    # 计算混淆矩阵
    cm = confusion_matrix(df['Knowledge Veracity'], df['Veracity'], labels=['S', 'NS', 'UNC'])

    # 将混淆矩阵转换为 DataFrame
    cm_df = pd.DataFrame(cm, index=['S', 'NS', 'UNC'], columns=['S', 'NS', 'UNC'])
    print(cm_df)
    column_sums = cm_df.sum(axis=0)
    raw_sums = cm_df.sum(axis=1)
    ACC = (column_sums['S'])/(column_sums['S']+column_sums['NS'])
    UR1 = cm_df['UNC']['NS']/raw_sums['NS']
    UR2 = cm_df['S']['S']/raw_sums['S']
    Uncle_score = (cm_df['UNC']['NS'] + cm_df['S']['S'])/(raw_sums['NS']+raw_sums['S'])
    UR = 2*UR1*UR2/(UR1+UR2)
    UP = cm_df['UNC']['NS']/column_sums['UNC']
    print(cm_df['UNC']['NS'])
    print(column_sums['UNC'])
    F1 = 2*UR*UP/(UR+UP)
    print("ACC: ", ACC)
    print("UA: ", UP)
    print("UUR: ", UR1)
    print("KCR: ", UR2)
    print("EA: ", Uncle_score)