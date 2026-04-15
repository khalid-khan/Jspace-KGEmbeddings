#Same script as above with totalbudget as 1000 initialtraining still 100 increment budget 50
# in method prep_Data line 77 relates to the input threshold mentioned in the paper in section 6.2.2


"""
=============================================================================
MASTER EVALUATION SCRIPT (GRAPHCR vs JSPACE)
=============================================================================
Purpose: 
This script evaluates the performance of the Baseline matching algorithm vs. 
the JSpace matching algorithm using the GraphCR Active Learning framework.

Differences from earlier versions:
1. STRICT ENVIRONMENT STERILIZATION: Automatically deletes old GraphCR folders 
   to prevent leftover files (like '1_vertices') from crashing the GraphCR parser.
2. STRICT ORPHAN FILTERING: Instead of assigning '-1' clusters, this script 
   strictly drops any edges from results.csv that point to entities missing 
   from the current gold standard subset.
3. NEWLINE PARSER FIX: Replaces standard pandas .to_csv() with a custom 
   save_clean_csv() function. This strips trailing newlines to prevent 
   GraphCR's 'IndexError: list index out of range' crash.
   4. is being used with a subset of Gold_Standards which gold_Standard records related
   to 1_vertices and 2_vertices only.
=============================================================================
"""

import os
import shutil
import pandas as pd
import sys
import matplotlib
import traceback
matplotlib.use('Agg')

sys.path.append('/content/graphCR')
from graphCR.evaluation import al_famer

# 1. SETUP & STRICT FOLDER STERILIZATION
path_base = '/content/graphCR/DS_baseline/threshold_0.0'
path_jspc = '/content/graphCR/DS_jspace/threshold_0.0'

# get rid the folders to ensure absolute cleanliness
if os.path.exists('/content/graphCR/DS_baseline'):
    shutil.rmtree('/content/graphCR/DS_baseline')
if os.path.exists('/content/graphCR/DS_jspace'):
    shutil.rmtree('/content/graphCR/DS_jspace')

os.makedirs(path_base, exist_ok=True)
os.makedirs(path_jspc, exist_ok=True)

# Load the clean files
src1_ids = set(pd.read_csv('/content/1_vertices', sep=';', header=None, usecols=[0])[0].astype(str))
results_df = pd.read_csv('/content/results.csv')
gold_raw = pd.read_csv('/content/gold_standard.csv', header=None)

# 2. DATA PREPration
gold_recs = [{'id': str(row[0]), 'cluster': idx} for idx, row in gold_raw.iterrows()] + \
            [{'id': str(row[1]), 'cluster': idx} for idx, row in gold_raw.iterrows()]
gold_df = pd.DataFrame(gold_recs).drop_duplicates('id')
valid_ids = set(gold_df['id'])


results_df = results_df[results_df['src_id'].isin(valid_ids) & results_df['tgt_id'].isin(valid_ids)]

all_ids = pd.concat([results_df['src_id'], results_df['tgt_id']]).unique()
v_df = pd.DataFrame({'id': all_ids.astype(str)}).merge(gold_df, on='id', how='left')
v_df['source'] = v_df['id'].apply(lambda x: 'Source_1' if x in src1_ids else 'Source_2')

# Safe CSV writer to bypass the  newline bug
def save_clean_csv(df, path):
    csv_str = df.to_csv(index=False, header=False, sep=';')
    with open(path, 'w') as f:
        f.write(csv_str.strip())

# 3. FORMAT DATA
def prep_data(folder, score_col):
    filtered_df = results_df[results_df[score_col] >= 0.1] #Input Threshold in section 6.2.2 in the paper
    
    edges_df = pd.DataFrame({'t':'e', 's1':'Source_1', 'i1':filtered_df['src_id'], 'i2':filtered_df['tgt_id'], 's2':'Source_2', 'sc':filtered_df[score_col]})
    save_clean_csv(edges_df, os.path.join(folder, 'edges.csv'))
    
    gold_std_df = v_df[['id', 'cluster']]
    save_clean_csv(gold_std_df, os.path.join(folder, 'gold_standard.csv'))
    
    vertices_df = pd.DataFrame({'id':v_df['id'], 'l':'rec', 's':v_df['source'], 'gt':v_df['cluster']})
    save_clean_csv(vertices_df, os.path.join(folder, 'vertices.csv'))
    
    meta_df = pd.DataFrame({'v':['v','v'], 's':['Source_1','Source_2'], 'm':['gtId:string','gtId:string']})
    save_clean_csv(meta_df, os.path.join(folder, 'metadata.csv'))

prep_data(path_base, 'links_score')
prep_data(path_jspc, 'jspace_score')

# 4. RUN & READ SAFELY
summary = []
for name, folder in [("Baseline", path_base), ("JSpace", path_jspc)]:
    print(f"\n🚀 Running {name}...")
    out_csv = f"{name}_results.tsv"

    try:
        al_famer.evaluate(
            folder, is_edge_wise=True, use_gpt=0,
            model_name="none", api_key="none",
            initial_training=100, increment_budget=50, 
            selection_strategy="bootstrap_comb",  
            total_budget=1000,  
            output=out_csv
        )
    except Exception as e:
        print(f"\n[CRITICAL CRASH IN {name}]")
        traceback.print_exc()

    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv, sep='\t', header=None)
        last_row = df.iloc[-1]
        summary.append({
            "Approach": name,
            "Precision": round(float(last_row.iloc[15]), 4),
            "Recall": round(float(last_row.iloc[16]), 4),
            "F1-Score": round(float(last_row.iloc[17]), 4)
        })

print("\n" + "="*50 + "\n FINAL METRICS\n" + "="*50)
print(pd.DataFrame(summary).to_markdown(index=False) if summary else "No data extracted.")


# =============================================================================
# HISTORICAL NOTES & OBSERVATIONS
# =============================================================================
# 24-March - We need to clarify what we mean clean vs uncleaned clusters(IMPORTANT).. 
# Make sure the gold_Standards is clean(have records from 1_vertices and 2_vertices).. 
# gold(55:1) look into how there are 55 exactly same cameras.
# 1732 VS 1714(JSPACE) WHERE DID THE 10 PRODUCTS GO..
# tHE TABLE SHOULD HAVE TWO ROWS ALSO FOR GraphCR.
# These are immediate questions and must prepare for...

# 1st April:
# two rows are now added.
# uncleand clusters: The graph before GraphCR does any active learning.
# uncleaned clusters: Counter({2: 116, 3: 59, 4: 43, 5: 27, 6: 15, 7: 11} This means before the active learnning there were 116 clusters of 2 cameras and so on..
# cleaned clusters: the graph after the Random Forest AI has evaluated the edges and "cut/separatee" the bad ones to repair the clusters.
# As I understand it: the complete_ratio and cluster co-efficient proves better results for jspace, here:
# feature importance used by random forest['pagerank', 'closeness', 'cluster_coefficient', 'betweenness', 'normal_link_ratio', 'strong_link_ratio', 'sim', 'bridges', 'betweenness', 'complete_ratio']
# [0.06406802  0.03120366     0.25305053           0.07330872       0.07010709                  0.     0.12026446   0.01037234        0.00271402       0.37491117]
# JSpace Precision: 72.5% (When JSpace guessed a match, it was right almost 3/4
# recal being 0.000014: because we ran the AI 1,000 iterations to search a dataset of over 13,000 entities in the gold_St
