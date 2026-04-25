import os
import torch
import csv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# === Configuration ===
FRAME_DIR = "output_frames/Frames" # Directory where your 10 captured frames are stored
ANNOTATED_DIR = "annotated_frames" # Directory to save frames with VQA output
OUTPUT_CSV = "frame_vqa_results.csv" # CSV file to store all results

# Corrected questions from your dataset to properly assess the model
QUESTIONS = [
"Which category does this waste belong to according to Indian Solid Waste Management Rules 2016?",
"Which colored bin should be used to dispose of this waste according to SWM Rules?",
"Rate the hazard level of this item on a scale of 1-10 where 1 is safe and 10 is extremely hazardous.",
"What environmental impact may result from improper disposal of this waste?",
"What kind of waste is visible in the image?"
]

# --- Model Setup ---
model_path = "instructor_blip_freeze_unfreeze" # Ensure this path is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

print(f"Using device: {device}")

# === Load model & processor ===
try:
    # Load model with bfloat16 for stability and auto device mapping
    model = InstructBlipForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    )
    # Load processor
    processor = InstructBlipProcessor.from_pretrained(model_path)
    # Sync processor config to model if needed
    processor.num_query_tokens = model.config.num_query_tokens
    model.eval()
    print(" Model and processor loaded successfully.")
except Exception as e:
    print(f" Error loading model from '{model_path}'. Please check the path. Error: {e}")
    exit()


# === Evaluation Setup ===
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
            # Format prompt similar to training
            prompt = f"Question: {question} Answer:"

            # Process inputs
            inputs = processor(
            images=[image],
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True
            ).to(device, torch.bfloat16) # Move to device and cast to bfloat16

            # Generate prediction with mixed precision
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                    **inputs,
                    max_length=50 # Increased max_length
                    )

                    # Decode prediction
                    prediction = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

                    # Remove the prompt from the generated text if it's included
                    if prediction.startswith(prompt):
                        prediction = prediction[len(prompt):].strip()

                        results.append({
                        "frame": frame_file,
                        "question": question,
                        "prediction": prediction
                        })

                        # Annotate and save the image for each question
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
        writer = csv.DictWriter(f, fieldnames=["frame", "question", "prediction"])
        writer.writeheader()
        writer.writerows(results)

        print(f" VQA results for {len(frame_files)} frames and {len(QUESTIONS)} questions saved to: {OUTPUT_CSV}")
        print(f" Annotated frames saved in: {ANNOTATED_DIR}/")
else:
    print("\n No results were generated.")
