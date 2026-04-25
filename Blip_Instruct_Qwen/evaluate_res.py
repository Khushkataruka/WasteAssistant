import os
import pandas as pd
from tqdm import tqdm
import csv

# Evaluation Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_calc

# === Helper Function for Precision and F1 ===
def calculate_token_metrics(reference, prediction):
    """Calculates token-level Precision, Recall, and F1-Score."""
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0 # Both are empty, perfect match

        if not pred_tokens:
            return 0.0, 0.0, 0.0 # Prediction is empty but reference is not

            true_positives = len(ref_tokens.intersection(pred_tokens))

            # Precision = TP / (TP + FP) = TP / len(pred_tokens)
            precision = true_positives / len(pred_tokens)

            # Recall = TP / (TP + FN) = TP / len(ref_tokens)
            recall = true_positives / len(ref_tokens) if ref_tokens else 0.0

            # F1 Score
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

                return precision, recall, f1

# === File Paths ===
predictions_csv = "frame_vqa_results.csv"
ground_truth_csv = "video_ground.csv"
output_csv = "full_evaluation_results.csv"

# === 1. Load and Merge Data using Pandas ===
print(" Loading and merging data...")
try:
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)
except FileNotFoundError as e:
    print(f" Error: {e}. Please ensure both CSV files are in the correct directory.")
    exit()

# --- Refined Merging Logic ---
# Standardize column names for clarity
pred_df.rename(columns={'prediction': 'model_prediction', 'frame': 'image_filename'}, inplace=True)
gt_df.rename(columns={'answer': 'ground_truth'}, inplace=True)

# Create a common 'image_filename' key in the ground truth dataframe from the full path
gt_df['image_filename'] = gt_df['image_path'].apply(os.path.basename)

# Merge the two dataframes on the common keys: 'image_filename' and 'question'
merged_df = pd.merge(pred_df, gt_df, on=['image_filename', 'question'])
print(f" Found {len(merged_df)} matching question-answer pairs to evaluate.")


# === 2. Run Evaluation ===
results = []
total_bleu, total_bert_f1, total_precision, total_f1 = 0, 0, 0, 0
smoothie = SmoothingFunction().method4

for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Calculating Scores"):
    gt = row['ground_truth']
    pred = row['model_prediction']

    # --- Metric Calculations ---
    # a) BLEU Score
    bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)

    # b) BERTScore
    P_bert, R_bert, F1_bert = bert_score_calc([pred], [gt], lang="en", rescale_with_baseline=True)
    bert_f1 = F1_bert.mean().item()

    # c) Token-level Precision and F1
    precision, _, f1 = calculate_token_metrics(gt, pred)

    # Accumulate totals
    total_bleu += bleu
    total_bert_f1 += bert_f1
    total_precision += precision
    total_f1 += f1

    # Store individual results
    results.append({
    'image_filename': row['image_filename'],
    'question': row['question'],
    'ground_truth': gt,
    'prediction': pred,
    'precision': round(precision, 4),
    'f1_score': round(f1, 4),
    'bleu_score': round(bleu, 4),
    'bert_score': round(bert_f1, 4)
    })

# === 3. Final Scores and Output ===
num_samples = len(merged_df)
if num_samples > 0:
    avg_precision = total_precision / num_samples
    avg_f1 = total_f1 / num_samples
    avg_bleu = total_bleu / num_samples
    avg_bert = total_bert_f1 / num_samples

    print("\n" + "="*30)
    print(" FINAL EVALUATION RESULTS ")
    print("="*30)
    print(f"Token Precision: {avg_precision:.4f}")
    print(f"Token F1-Score: {avg_f1:.4f}")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"BERTScore F1: {avg_bert:.4f}")
    print("="*30)

    # Save detailed results to a new CSV file
    with open(output_csv, "w", newline="", encoding='utf-8') as f:
        # Corrected fieldnames to match the keys in the 'results' dictionary
        writer = csv.DictWriter(f, fieldnames=['image_filename', 'question', 'ground_truth', 'prediction', 'precision', 'f1_score', 'bleu_score', 'bert_score'])
        writer.writeheader()
        writer.writerows(results)

        print(f" Detailed results saved to: {output_csv}")

else:
    print(" No matching data found to evaluate. Please check that the image filenames and questions align in your CSV files.")
