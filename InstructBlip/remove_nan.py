import os
import time
import base64
import re
from PIL import Image
from io import BytesIO
import ollama
import pandas as pd

# Config
csv_path = "vqa_generated.csv"
model_name = "llava"
temperature = 0.4

# SWM Chart Text (for prompt context)
swm_chart = """
Waste Classification Based on India's SWM Rules, 2016:

1. Biodegradable Waste:
    - Examples: Food scraps, vegetable peels, flowers, eggshells
    - Bin Color: Green
    - Disposal: Compost at home or give to municipality

2. Dry Waste (Non-Biodegradable):
    - Examples: Paper, cardboard, plastic bottles, glass, clean metal cans
    - Bin Color: Blue
    - Disposal: Clean and dry before binning; recyclable

3. Sanitary Waste:
    - Examples: Diapers, sanitary pads, bandages, condoms
    - Bin Color: White/Yellow
    - Disposal: Wrap in paper, mark with red cross, hand over separately

4. Domestic Hazardous Waste:
    - Examples: Batteries, chemicals, CFL bulbs, expired medicine
    - Bin Color: Red Bag (Separate Collection)
    - Disposal: Give to ULB or authorized collectors

5. E-Waste:
    - Examples: Phones, wires, laptops, chargers
    - Bin: Separate box or drop-off
    - Disposal: Do not mix; use authorized e-waste collection

6. Inert Waste:
    - Examples: Debris, tiles, broken bricks
    - Handling: Not collected door-to-door; inform ULB for pickup

7. Composite Waste:
    - Examples: Chips packets, cups with plastic lining
    - Bin: Blue (Dry Waste)
    - Disposal: Not recyclable; may go to landfill or EPR-linked systems
"""

# Helper: Clean answer to remove special characters
def clean_answer(ans):
    return re.sub(r"[^\w\s\.-]", "", ans).strip()

# Helper: Encode image to base64
def image_to_base64(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format='JPEG')
            return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Failed to encode image {path}: {e}")
        return None

# Load the dataset
df = pd.read_csv(csv_path)
updated_rows = 0

# Process rows where answer == "N/A"
for idx, row in df.iterrows():
    if row['answer'] != "N/A":
        continue

    image_path = row['image_path']
    question = row['question']

    # Build prompt with just this one question
    prompt_template = f"""You are a waste classification expert trained on India's Solid Waste Management Rules (2016).

Use the image and the SWM chart below to answer the following visual question:
- Only answer using visual and regulatory cues.
- Follow Indian municipal standards.
- Answer briefly (50 words max).
- Avoid special characters in the answer.

{swm_chart}

1. {question}
"""

    print(f"\nProcessing: {image_path} | Question: {question}")
    img_b64 = image_to_base64(image_path)
    if not img_b64:
        print(f"Skipping image due to encoding issue: {image_path}")
        continue

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt_template,
                "images": [img_b64]
            }],
            options={"temperature": temperature}
        )

        # Get and clean response
        raw = response["message"]["content"].strip()

        if not raw or raw.lower() in ["n/a", "none", ""]:
            print("No valid answer returned, keeping N/A.")
            continue  # Skip updating

        # Handle format like "1. Answer" or just "Answer"
        if "." in raw:
            parts = raw.split(".", 1)
            if parts[0].strip().isdigit():
                raw = parts[1]

        cleaned = clean_answer(raw)
        if cleaned:
            df.at[idx, 'answer'] = cleaned
            updated_rows += 1
            print(f"Answer updated: {cleaned}")
        else:
            print("Cleaned answer still empty, skipping.")

        time.sleep(1)

    except Exception as e:
        print(f"Error with {image_path}: {e}")

# Save updated CSV
df.to_csv(csv_path, index=False)
print(f"\nCompleted. Total rows updated: {updated_rows}")
