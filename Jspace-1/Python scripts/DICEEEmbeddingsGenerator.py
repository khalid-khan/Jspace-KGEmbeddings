# SCRIPT NAME: DICE Knowledge Graph Embedding Engine (Independent Mode)
# AUTHOR: Khalid Khan
# ------------------------------------------------------------------------------
# 1. READS: 1_vertices...ttl AND 2_vertices...ttl separately.
# 2. EXTRACTS: Triples for each file into distinct training sets.
# 3. TRAINS: Two separate DICE 'Keci' models (Phi_1, Phi_2).
# 4. OUTPUTS: Saves results to 'dice_embeddings_1' and 'dice_embeddings_2'.

import pandas as pd
import re
import os
from dicee.executer import Execute
from dicee.config import Namespace

# --- CONFIGURATION -
FILES_CONFIG = [
    {
        "input": "1_vertices_output of 73 batches with category hierarchy and literals removed names removed.ttl",
        "output_dir": "dice_embeddings_1",
        "temp_data_dir": "kg_data_1"
    },
    {
        "input": "2_vertices_output of 73 batches with category hierarchy and literals removed names removed.ttl",
        "output_dir": "dice_embeddings_2",
        "temp_data_dir": "kg_data_2"
    }
]

def extract_triples(input_path):
    print(f"Extracting triples from {input_path}...")
    triples = []
    current_subject = None

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@prefix'):
                continue

            subject_match = re.match(r'^<([^>]+)>', line)
            if subject_match:
                current_subject = subject_match.group(1)

            po_match = re.search(r'([\w\d]+:[\w\d]+)\s+(.+?)\s*[;.]$', line)

            if current_subject and po_match:
                predicate = po_match.group(1)
                obj = po_match.group(2).strip()

                # Clean object
                obj = obj.strip('"').replace(' ', '_')
                clean_sub = current_subject.replace(' ', '_')
                clean_pred = predicate.replace(' ', '_')

                triples.append([clean_sub, clean_pred, obj])

    return triples

def run_dicee_training(triples, data_dir, output_dir):
    # 1. Prepare Data Directory
    os.makedirs(data_dir, exist_ok=True)

    df = pd.DataFrame(triples)
    df = df.dropna()

    # Consistent use of 'train_path' to fix NameError
    train_path = os.path.join(data_dir, 'train.txt')
    df.to_csv(train_path, sep=' ', index=False, header=False)

    print(f"Training data saved to {train_path} ({len(df)} triples)")

    # 2. Configure DICEE
    args = Namespace()
    args.model = 'Keci' #Currently using Keci
    args.dataset_dir = data_dir
    args.path_to_store_single_run = output_dir
    args.num_epochs = 50
    args.embedding_dim = 128
    args.batch_size = 512 #So far has worked nicely
    args.scoring_technique = "KvsAll" #Lets try others techiniques as well if time premits.

    # 3. Train
    print(f"Starting DICEE training -> {output_dir}...")
    try:
        Execute(args).start()
        print(f"Success! Model saved to {output_dir}")
    except Exception as e:
        print(f"Training failed: {e}")

# --- MAIN EXECUTION ---
for config in FILES_CONFIG:
    print(f"\n--- Processing {config['input']} ---")
    extracted_triples = extract_triples(config['input'])

    if extracted_triples:
        run_dicee_training(extracted_triples, config['temp_data_dir'], config['output_dir'])
    else:
        print("Skipping training (no triples found).")

print("\nAll tasks complete.")
print("Next Step: Run JSpace alignment script.")