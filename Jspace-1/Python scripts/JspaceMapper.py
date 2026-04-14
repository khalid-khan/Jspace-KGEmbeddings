# Generates jspace with FULL implementation of Li
# 2. FULL IMPLEMENTATION: Distance Preservation Loss (L_i)
# Ensure that the neighbors in JSpace have the same distance as in DICE
# Select a random subset for L_i to keep memory low if dataset is huge


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------------------------
# 1.JSpace Mapping Function (Phi*)
# ------------------------------------------------------------------------------
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

        # Identity Initialization because of comment from ## 16th Feb 2026
        # To make sure the code uses a copy machine in the start rather then randomly scrambling the vectors for epoch 0.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Start near identity to preserve initial DICEE structure
                nn.init.eye_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        z = self.net(x)

        return z / (z.norm(dim=1, keepdim=True) + 1e-8)

def load_dice_data(folder_path):
    mapping_path = os.path.join(folder_path, 'entity_to_idx.csv')
    weights_path = os.path.join(folder_path, 'model.pt')

    mapping_df = pd.read_csv(mapping_path)
    # Extract IDs (handling potential URL formats)
    #id_to_idx = {str(row.iloc[1]).split('/')[-1]: int(row.iloc[0]) for _, row in mapping_df.iterrows()}
    id_to_idx = {str(row.iloc[1]).split('/')[-1].replace('ex:', ''): int(row.iloc[0]) for _, row in mapping_df.iterrows()}


    weights = torch.load(weights_path, map_location='cpu')
    if isinstance(weights, dict):
        weights = weights.get('ent_emb.weight') or next(v for v in weights.values() if torch.is_tensor(v))

    return id_to_idx, weights

# ------------------------------------------------------------------------------
# 2. FULL IMPLEMENTATION: PRE-CALCULATING NEIGHBORHOOD STRUCTURE
# ------------------------------------------------------------------------------
def train():
    #alpha = 0.4        # Weight for Structural Preservation (per the)
    alpha=0.01  # Higher values for alph makes it harder for the model to move away, in other words it makes higher preservation.
    ### try some different alpha values


    k_neighbors = 10   # The 'k' defined in KnearestNeighbr
    batch_size = 512

    # Load Data
    map1, w1 = load_dice_data('dice_embeddings_1')
    map2, w2 = load_dice_data('dice_embeddings_2')
    links = pd.read_csv('links.csv', header=None)

    # 1. Align linked entities
    pairs = []
    for _, row in links.iterrows():
        s, t = str(row[0]), str(row[1])
        score = float(row[2]) # Get the confidence score

        # ONLY train on highly confident pairs  # 9 March: we can flip the value in diagram
        if s in map1 and t in map2 and score >= 0.50: # 9 March: a good reason is needed to make chagnes for better graph
            pairs.append((map1[s], map2[t]))

    src_idx = torch.tensor([p[0] for p in pairs])
    tgt_idx = torch.tensor([p[1] for p in pairs])

    print(f"Pre-calculating {k_neighbors}-Nearest Neighbors for Structural Preservation...")
    # 2. FULL IMPLEMENTATION: Compute KNN in the original space (w1)
    # This captures the "Local Topology" we aim to preserve.
    #knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(w1[src_idx].numpy())
    #distances, indices = knn.kneighbors(w1[src_idx].numpy())


    # Normalize w1 BEFORE computing neighbors.
    # This ensures "original distances" are on the unit sphere (0 to 2),
    # matching the scale of the model's output.
    w1_norm = F.normalize(w1, p=2, dim=1)
    ### Why do we need this at all
    ####This could be why the the epoch 0 shows differences(not zero). Could be counterproductive
    ####Why even int he epoch 0 the Li loss is not zero(it should hav ebeen zero as nothing else has been done yet and the values are just added in the jspace)

    #/actually Epoch 0 is NOT zero because the model starts as a randome number generator and not as a perfect copy machine.
    #/ at epoch 0 the model takes that map, scrambles it, rotates it randomly, and projects it onto a random sphere.
    #/ the goal of training is to unscramble the map going along, That is why the loss drops from 0.22 to 0.04 accordingly

    ## 16th Feb 2026 :see if there is related work that ahelp us understand what should we start with copy or random or something else.
    ## 16th Feb 2026 :effectiveness, the jspace that works i the next steps...
    ## implemented accordingly in the __init__ method above, with comment 'Identity Initialization '


    # Compute KNN on the normalized vectors instead of raw w1
    #knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(w1_norm[src_idx].numpy())
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(w1_norm.numpy()) #/ removed the part src_idx, so that it can find its true neighbours in entire Graph not just the pairs. Bug: Make sure it is taken care of 'anchor = model(w1[src_idx])'
    distances, indices = knn.kneighbors(w1_norm[src_idx].numpy())  #/ We still only calculate distances FOR the anchors (to save time/memory)
    #### the a, b in Ljoint comes from links.csv(potiential pairs)
    #### the a, b in the Li comes from the original graphs(so all the a's and b's, not just the ones which are pairs.) tHE b IS ONE OF THE K-NEARST NEIGHBOURS OF a.
    ### we should make sure that the NEARESTNEIGHBOURS CALCULATED FOR ALL THE A's and B's.






    # Convert to tensors for the loss function
    orig_neighbor_indices = torch.from_numpy(indices[:, 1:]) # Exclude self
    orig_neighbor_dists = torch.from_numpy(distances[:, 1:])

    model = JSpaceMapper(w1.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # The--- Losses
    criterion_joint = nn.TripletMarginLoss(margin=0.5)
    criterion_dist = nn.MSELoss()

    print("Starting Training...")

    for epoch in range(301):
        model.train()
        optimizer.zero_grad()

        # 1. Joint Alignment Loss (L_joint)
        #anchor = model(w1[src_idx])
        #positive = w2[tgt_idx] # Fixed target space

        # 1. Joint Alignment Loss (L_joint)
        #anchor = model(w1[src_idx])
        projected_w1 = model(w1)       # Bug: Take all the neighbours(entire graph) into account
        anchor = projected_w1[src_idx] # Bug: Take all the neighbours(entire graph) into account







        # Normalize target embeddings so they are on the same scale as the model output
        #w2_norm = torch.nn.functional.normalize(w2, p=2, dim=1)
        #positive = w2_norm[tgt_idx]

        positive = model(w2[tgt_idx]) # We now move both graphs rather then keeping one fixed because of ## 16th Feb 2026 : the model moves vectors which could be froom any graph.. both pairs are moved..

        # Hard Negative Mining (shuffling targets)
        #neg_idx = tgt_idx[torch.randperm(len(tgt_idx))]
        #negative = w2[neg_idx]

        # Hard Negative Mining
        neg_idx = tgt_idx[torch.randperm(len(tgt_idx))]
        #negative = w2_norm[neg_idx]
        negative = model(w2[neg_idx]) # We now move both graphs rather then keeping one fixed because of ## 16th Feb 2026 : the model moves vectors which could be froom any graph.. both pairs are moved..



        l_joint = criterion_joint(anchor, positive, negative)

        # 2. FULL IMPLEMENTATION: Distance Preservation Loss (L_i)
        # Ensure that the neighbors in JSpace have the same distance as in DICE
        # Select a random subset for L_i to keep memory low if dataset is huge
        subset = torch.randint(0, len(src_idx), (128,))



        # Neighbors of subset in JSpace
        #current_neighbor_embeds = anchor[orig_neighbor_indices[subset]] # Shape [128, k, dim]
        #current_anchor_embeds = anchor[subset].unsqueeze(1)            # Shape [128, 1, dim]



        # Calculate neighbors for ALL anchors at once by removing the subset.. because of ## 16th Feb 2026 : rather thean reandom 128 nodes, just take nodes all the nodes accordignly..
        #current_neighbor_embeds = anchor[orig_neighbor_indices] # Shape: [Total_Anchors, k, dim]


        current_neighbor_embeds = projected_w1[orig_neighbor_indices]  # Bug: Take all the neighbours(entire graph) into account
        current_anchor_embeds = anchor.unsqueeze(1)             # Shape: [Total_Anchors, 1, dim]








        # Calculate current distances to neighbors in JSpace
        jspace_dists = torch.norm(current_anchor_embeds - current_neighbor_embeds, dim=2)


        # Compare to original DICE distances #\scales every vector in w1 between 0-1
        #l_i = criterion_dist(jspace_dists, orig_neighbor_dists[subset].to(torch.float32))
        l_i = criterion_dist(jspace_dists, orig_neighbor_dists.to(torch.float32)) #Not using the subset now because of this comment ## 16th Feb 2026 : rather thean reandom 128 nodes, just take nodes all the nodes accordignly..
        #### Normalize it with number of nodes in the graph Gi*k

        #/calculating the error for a random sample of 128 nodes(subset = torch.randint(0, len(src_idx), (128,))).
        #/basically, Sum of 128 errors divivided by 128 (via MSELoss)
        #/ otherwise, we might get the Sum of 128 errors divivided 27,000 a very smaller number eg e.g., 0.0002)
        ## 16th Feb 2026 : rather thean reandom 128 nodes, just take nodes all the nodes accordignly.. implemented in the line of code above

        # Total Combined Loss
        #Remember the changes you made the linie of code
        ###lossTotal = (alpha * l_i * 1/number of graphs(2)) + ((1 - alpha) * l_joint * 1/number Owlsamea links)
        ### We can normalize the li part by 1/number graphs to normalize.
        ### We should also use the nomalization on l_joint 1/Owlsameas links


        #// the code only calculates L_i for one graph (KG1). AND then PROJECTs from KG1 to KG2. Since we aren't calculating L_i for KG2 separately and adding them together, there is no need for 1/number of graphs(2).
        #//Because we dont WANT TO MOVE BOTH KG1 AND KG2. to keep things stable we move kg1 and keep kg2 fixed.
        ## 16th Feb 2026 : the model moves vectors which could be froom any graph.. both pairs are moved..


        loss = (alpha * l_i) + ((1 - alpha) * l_joint)
        #loss = (alpha * (l_i * 10)) + ((1 - alpha) * l_joint)


        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3} | Loss: {loss.item():.4f} | L_i: {l_i.item():.4f} | L_joint: {l_joint.item():.4f}")

    torch.save(model.state_dict(), 'jspace_projector.pth')
    print("Full The----compliant model saved.")

if __name__ == "__main__":
    train()