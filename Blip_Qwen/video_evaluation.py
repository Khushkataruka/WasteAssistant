import os
import torch
import csv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BlipProcessor, BlipForQuestionAnswering


# === Configuration ===
FRAME_DIR = "output_frames/Frames" # Directory where your 10 captured frames are stored
ANNOTATED_DIR = "annotated_frames" # Directory to save frames with VQA output
OUTPUT_CSV = "frame_vqa_results.csv" # CSV file to store all results

# Corrected questions from your dataset to properly assess the model
QUESTIONS = [
"Rate the hazard level of this item on a scale of 1-10 where 1 is safe and 10 is extremely hazardous",
"Which colored bin should be used to dispose of this waste according to SWM Rules?,",
"What materials is this waste object primarily made of?",
"Which category does this waste belong to according to Indian Solid Waste Management Rules 2016?",
"What kind of waste is visible in the image?"
]

# --- Model Setup ---
model_path = "blip_model_all_layers" # Ensure this path is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

print(f"Using device: {device}")

# === Load model & processor ===
try:
    model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)
    processor = BlipProcessor.from_pretrained(model_path)
    model.eval()
    print(" Model and processor loaded successfully.")
except Exception as e:
    print(f" Error loading model from '{model_path}'. Please check the path. Error: {e}")
    exit()


# === Evaluation Setup ===
smooth = SmoothingFunction().method4
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    print("Arial font not found, using default font.")
    font = ImageFont.load_default()

results = []

def annotate_image(image_path, question, answer, output_path):
    """Annotates an image with the question and the model's predicted answer."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Create a semi-transparent background for the text
    text = f"Q: {question}\nA: {answer}"

    # Use textbbox to get the size of the text block
    try:
        # textbbox is preferred for modern Pillow versions
        text_box = draw.textbbox((10, 10), text, font=font)
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)
        text_box = (10, 10, 10 + text_width, 10 + text_height)

        draw.rectangle([(5, 5), (text_box[2] + 15, text_box[3] + 15)], fill=(0, 0, 0, 180))
        draw.text((10, 10), text, fill="white", font=font)

        image.save(output_path)

# === Process Each Frame with Each Question ===
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not frame_files:
    print(f" No image frames found in the '{FRAME_DIR}' directory. Please add your captured frames.")
    exit()

# Setup the main progress bar for frames
for frame_file in tqdm(frame_files, desc=" Processing Frames"):
    frame_path = os.path.join(FRAME_DIR, frame_file)
    try:
        image = Image.open(frame_path).convert("RGB")
    except Exception as e:
        print(f" Skipping {frame_file}: Could not open image. Error: {e}")
        continue

        # Ask all questions for the current frame
        for i, question in enumerate(QUESTIONS):
            inputs = processor(image, question, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=50) # Increased max_length for descriptive answers
                prediction = processor.decode(output_ids[0], skip_special_tokens=True).strip()

                # For demonstration, we don't have a ground truth. 
                # The scores here are just for structural consistency.
                # In a real evaluation, you would compare `prediction` to a `ground_truth` answer.
                bleu_score = 0.0 
                bert_f1 = 0.0

                results.append({
                "frame": frame_file,
                "question": question,
                "prediction": prediction,
                "bleu_score": "N/A",
                "bert_score": "N/A"
                })

                # Annotate and save the image for each question
                # The filename will indicate which question was asked (e.g., frame-1_q2.jpg)
                base_name, ext = os.path.splitext(frame_file)
                annotated_filename = f"{base_name}_q{i+1}{ext}"
                annotated_path = os.path.join(ANNOTATED_DIR, annotated_filename)
                annotate_image(frame_path, question, prediction, annotated_path)

# === Save Results to CSV ===
if results:
    print("\n" + "="*30)
    print(" VQA Processing Complete.")
    print("="*30)

    with open(OUTPUT_CSV, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "question", "prediction", "bleu_score", "bert_score"])
        writer.writeheader()
        writer.writerows(results)

        print(f" VQA results for {len(frame_files)} frames and {len(QUESTIONS)} questions saved to: {OUTPUT_CSV}")
        print(f" Annotated frames saved in: {ANNOTATED_DIR}/")
else:
    print("\n No results were generated.")

