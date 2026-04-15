
import pandas as pd

def evaluate():
    print("--- Evaluating Alignment Performance ---")

    try:
        ground_truth = pd.read_csv('links.csv', header=None, names=['src_id', 'tgt_id', 'score']) #For potential pairs(source id and target Id) only..
        predictions = pd.read_csv('predicted_links.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    def clean_id(x): return str(x).split('/')[-1].replace('ex:', '')

    predictions['src_clean'] = predictions['src_id'].apply(clean_id)
    predictions['tgt_clean'] = predictions['matched_tgt_id'].apply(clean_id)
    ground_truth['src_clean'] = ground_truth['src_id'].apply(clean_id)
    ground_truth['tgt_clean'] = ground_truth['tgt_id'].apply(clean_id)

    merged = pd.merge(
        predictions,
        ground_truth,
        left_on='src_clean',
        right_on='src_clean',
        how='inner'
    )

    correct = merged[merged['tgt_clean_x'] == merged['tgt_clean_y']]

    hits_at_1 = len(correct)
    total_test_cases = len(ground_truth['src_clean'].unique())
    accuracy = (hits_at_1 / total_test_cases) * 100 if total_test_cases > 0 else 0

    print(f"Total Reference Links: {len(ground_truth)}")
    print(f"Correct Predictions (Hits@1): {hits_at_1}")
    print(f"Accuracy against the links.csv: {accuracy:.2f}%")

    if not correct.empty:
        print("\n--- Sample Correct Matches ---")
        print(correct[['src_id_x', 'matched_tgt_id']].head())

if __name__ == "__main__":
    evaluate()
