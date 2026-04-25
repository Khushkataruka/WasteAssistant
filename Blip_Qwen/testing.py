import os
import json
import random

# === Root dataset directory (one level up from this script) ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BlipProcessor, BlipForQuestionAnswering
from bert_score import score as bert_score # Optional semantic metric

# === Paths ===
model_path = "blip_model_all_layers"
test_json = os.path.join(ROOT_DIR, "split_jsons", "test.json")
output_csv = "vqa_eval_results.csv"

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Load Model and Processor ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)
processor = BlipProcessor.from_pretrained(model_path)

# === Load Test Data ===
def load_test_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

test_data = load_test_data(test_json)
print(f" Loaded {len(test_data)} test samples")

# === Evaluation Setup ===
model.eval()
smooth = SmoothingFunction().method4
total_bleu = 0
total_bert = 0
failures = 0
results = []

# === Run Evaluation ===
for sample in tqdm(test_data, desc="Evaluating VQA"):
    image_path = sample["image_path"]
    question = sample["question"]
    answer = sample["answer"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f" Could not open image {image_path}: {e}")
        failures += 1
        continue

        # Process inputs
        inputs = processor(image, question, return_tensors="pt").to(device)

        # Generate prediction
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=32)
            prediction = processor.decode(output_ids[0], skip_special_tokens=True)

            # Token-level BLEU score
            ref_tokens = answer.split()
            pred_tokens = prediction.split()
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            total_bleu += bleu

            # Semantic similarity (BERTScore F1)
            P, R, F1 = bert_score([prediction], [answer], lang="en", rescale_with_baseline=True)
            bert_f1 = F1[0].item()
            total_bert += bert_f1

            # Store result
            results.append({
            "image_path": image_path,
            "question": question,
            "ground_truth": answer,
            "prediction": prediction,
            "bleu_score": round(bleu, 4),
            "bert_score": round(bert_f1, 4)
            })

            # Optional: Print live
            print(f"\nQ: {question}\nGT: {answer}\nPred: {prediction}\nBLEU: {bleu:.4f} | BERT: {bert_f1:.4f}\n")

# === Final Scores ===
num_samples = len(test_data) - failures
avg_bleu = total_bleu / num_samples if num_samples > 0 else 0
avg_bert = total_bert / num_samples if num_samples > 0 else 0
print(f"\n Average BLEU Score: {avg_bleu:.4f}")
print(f"Average BERTScore F1: {avg_bert:.4f}")
print(f"Failures: {failures}")

# === Save Results to CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "question", "ground_truth", "prediction", "bleu_score", "bert_score"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to: {output_csv}")
