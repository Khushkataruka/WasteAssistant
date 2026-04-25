import os
import csv
import time
import base64
import re
from PIL import Image
from io import BytesIO
import ollama

# Config
dataset_root = "dataset"  # folder path with subfolders for each category (e.g., dataset/plastic/)
output_csv = "vqa_generated.csv"
model_name = "llava"  # change if using a different model version
temperature = 0.4

# VQA Questions aligned with SWM Rules
questions = [
    "What type of waste is represented in the image?",
    "Which category does this waste belong to according to Indian Solid Waste Management Rules 2016?",
    "Which colored bin should be used to dispose of this waste?",
    "Is this waste biodegradable, non-biodegradable, or hazardous?",
    "Is this item recyclable through municipal systems in India?",
    "What steps should be taken before disposing of this waste?",
    "Does this item require any special handling or segregation?",
    "What potential environmental impact may result from improper disposal?",
    "Can this waste be composted at the household level?",
    "Is this item considered a violation of the SWM Rules?",
    "How hazardous is this item on a scale of 1 to 10?",
    "Can this item be reused or upcycled practically?",
    "What is the estimated time it would take for this waste to decompose in landfill?",
    "What visible features indicate its original purpose or usage?",
    "Which agency or authority is typically responsible for collecting this waste type?"
]

# Text summary of the Indian SWM 2016 chart
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

# Full Prompt Template
combined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
prompt_template = f"""You are a waste classification expert trained on India's Solid Waste Management Rules (2016).

Use the image and the SWM chart below to answer the following 15 visual questions:
- Only answer using visual and regulatory cues.
- Follow Indian municipal standards.
- Answer each point briefly (50 words max).
- Avoid special characters in the answer.

{swm_chart}

{combined_questions}
"""

# Encode image to base64
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

# Clean special characters
def clean_answer(ans):
    return re.sub(r"[^\w\s\.-]", "", ans).strip()

# Create CSV if it doesn't exist
file_exists = os.path.exists(output_csv)
with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists or os.stat(output_csv).st_size == 0:
        writer.writerow(['image_path', 'label', 'question', 'answer'])

    # Loop through dataset folders
    for category in sorted(os.listdir(dataset_root)):
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\nCategory: {category}")
        images = [entry.path for entry in os.scandir(category_path)
                  if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_path in images:
            print(f"Processing: {img_path}")
            img_b64 = image_to_base64(img_path)
            if not img_b64:
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

                raw_answer = response["message"]["content"].strip()
                answer_lines = raw_answer.split("\n")

                for i, question in enumerate(questions):
                    try:
                        line = answer_lines[i].strip()
                        ans = line.split(".", 1)[-1].strip() if '.' in line else line
                        ans = clean_answer(ans)
                    except:
                        ans = "N/A"
                    writer.writerow([img_path, category, question, ans])

                time.sleep(1)  # Prevent overload

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
