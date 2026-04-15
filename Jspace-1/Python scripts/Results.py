#Results script: Our scores compared to links.csv scores and goldstandards.

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Model Definition For the Jspace
class JSpaceMapper(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=1, keepdim=True) + 1e-8)

def load_embeddings(folder_path):
    mapping_df = pd.read_csv(os.path.join(folder_path, 'entity_to_idx.csv'))
    id_to_idx = {str(row.iloc[1]).split('/')[-1].replace('ex:', ''): int(row.iloc[0]) for _, row in mapping_df.iterrows()}

    weights = torch.load(os.path.join(folder_path, 'model.pt'), map_location='cpu')
    if isinstance(weights, dict):
        weights = weights.get('ent_emb.weight') or next(v for v in weights.values() if torch.is_tensor(v))
    return id_to_idx, weights

def evaluate():
    # Load data
    map1, w1 = load_embeddings('dice_embeddings_1')
    map2, w2 = load_embeddings('dice_embeddings_2')

    links = pd.read_csv('links.csv', header=None, names=['src_id', 'tgt_id', 'links_score'])
    gold_standards = pd.read_csv('gold_standard.csv', header=None, names=['src_id', 'tgt_id'])


    gold_set = set(zip(gold_standards['src_id'].astype(str), gold_standards['tgt_id'].astype(str)))

    # Load Model
    model = JSpaceMapper(w1.shape[1])
    model.load_state_dict(torch.load('jspace_projector.pth'))
    model.eval()

    with torch.no_grad():
        proj1 = model(w1)
        proj2 = model(w2)

    results = []

    for _, row in links.iterrows():
        src, tgt = str(row['src_id']), str(row['tgt_id'])
        links_score = float(row['links_score'])

        # Check if pair exists in our mappings
        if src in map1 and tgt in map2:
            idx1, idx2 = map1[src], map2[tgt]


            dist = torch.norm(proj1[idx1] - proj2[idx2]).item()
            jspace_score = 1 - (dist / 2.0)

            in_gold = (src, tgt) in gold_set

            results.append({
                'src_id': src,
                'tgt_id': tgt,
                'links_score': links_score,
                'jspace_score': jspace_score,
                'in_gold_standard': in_gold
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)



    #7th March - ROC curve:
    # Convert true/false to 1/0 for sklearn
    y_true = results_df['in_gold_standard'].astype(int)

    # Calculate AUC (Area Under the Curve)
    auc_links = roc_auc_score(y_true, results_df['links_score'])
    auc_jspace = roc_auc_score(y_true, results_df['jspace_score'])

    # Calculate Curve Points
    fpr_links, tpr_links, _ = roc_curve(y_true, results_df['links_score'])
    fpr_jspace, tpr_jspace, _ = roc_curve(y_true, results_df['jspace_score'])

    # Save the plot as an image
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_links, tpr_links, label=f'Baseline (links.csv) AUC = {auc_links:.3f}', color='blue')
    plt.plot(fpr_jspace, tpr_jspace, label=f'JSpace Model AUC = {auc_jspace:.3f}', color='red')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: JSpace vs Baseline')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

    # Print AUC to the prompt
    print("\n--- ROC / AUC ANALYSIS ---")
    print(f"Baseline AUC: {auc_links:.3f}")
    print(f"JSpace AUC:   {auc_jspace:.3f}")
    print("ROC curve graph saved to 'roc_curve.png'.")
    # --------------------------------


    # Calculate Metrics (> 0.5 threshold) as an experinment to see what will happen with more cleaner data
    total_evaluated = len(results_df)

    links_positives = results_df[results_df['links_score'] > 0.5]
    links_true_positives = links_positives[links_positives['in_gold_standard'] == True]

    jspace_positives = results_df[results_df['jspace_score'] > 0.5] 


    jspace_true_positives = jspace_positives[jspace_positives['in_gold_standard'] == True]

    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Total pairs evaluated from links.csv: {total_evaluated}")
    print("-" * 40)
    print("ORIGINAL LINKS.CSV SCORES:")
    print(f"Matches proposed (>0.5 score): {len(links_positives)}")
    print(f"Actually Correct (True Positives): {len(links_true_positives)}")
    print("-" * 40)
    print("OUR JSPACE SCORES:")
    print(f"Matches proposed (>0.5 score): {len(jspace_positives)}")
    print(f"Actually Correct (True Positives): {len(jspace_true_positives)}")
    print("--------------------------------")
    print("Detailed breakdown saved to 'results.csv'.")

if __name__ == "__main__":
    evaluate()
