import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

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
    mapping_path = os.path.join(folder_path, 'entity_to_idx.csv')
    weights_path = os.path.join(folder_path, 'model.pt')
    mapping_df = pd.read_csv(mapping_path)


    #idx_to_entity = {int(row.iloc[0]): str(row.iloc[1]).split('/')[-1] for _, row in mapping_df.iterrows()}
    idx_to_entity = {int(row.iloc[0]): str(row.iloc[1]).split('/')[-1].replace('ex:', '') for _, row in mapping_df.iterrows()}

    weights = torch.load(weights_path, map_location='cpu')
    if isinstance(weights, dict):
        weights = weights.get('ent_emb.weight') or next(v for v in weights.values() if torch.is_tensor(v))

    return idx_to_entity, weights

def run_inference():
    print("--- Loading Data and Trained Projector ---")
    id_map1, weights1 = load_embeddings('dice_embeddings_1')
    id_map2, weights2 = load_embeddings('dice_embeddings_2')

    model = JSpaceMapper(weights1.shape[1])
    model.load_state_dict(torch.load('jspace_projector.pth'))
    model.eval()


    print("--- Projecting all entities into JSpace ---")
    with torch.no_grad():

        projected1 = model(weights1).numpy()

        import torch.nn.functional as F
        projected2 = F.normalize(weights2, p=2, dim=1).numpy()





    print("--- Calculating Validity Scores (Distance-based) ---")
    # Use NearestNeighbors to find the closest match in the target KG
    # 9 March - cosine - similarity score. double check what did we used to calculate the scores. our output should be based on similarity. that they express similarity.
    # 9 March - Euclidea(distance) is inversly proportional to simillarity..
    # in results.py, then this score inverted  (1 - distance) to create the final jspace_score that goes from 0 to 1 and is similarity score which is then inserted into GraphCR.
    nbrs = NearestNeighbors(n_neighbors=1, metric='cosine') # Also, I ran it with euclidean
    nbrs.fit(projected2)
    distances, indices = nbrs.kneighbors(projected1)

    results = []
    for i in range(len(projected1)):
        results.append({
            'src_id': id_map1[i],
            'matched_tgt_id': id_map2[indices[i][0]],
            'jspace_distance': float(distances[i][0]) # Lower = More likely to be a valid sameAs link
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv('predicted_links.csv', index=False)
    print(f"Success: Predicted links saved to predicted_links.csv.")

if __name__ == "__main__":
    run_inference()
