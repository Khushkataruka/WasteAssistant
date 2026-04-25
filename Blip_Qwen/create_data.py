

import os
import csv
import time
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

# ================================
# CONFIGURATION
# ================================
class Config:
    DATASET_ROOT = "dataset"
    OUTPUT_CSV = "vqa_generated.csv"

    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

    # Load token from .env file at root
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, ".env"))
    TOKEN = os.environ.get("HF_TOKEN")

    MAX_NEW_TOKENS = 400
    TEMPERATURE = 0.4
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    MAX_PIXELS = 1024 * 28 * 28

# ================================
# QUESTIONS AND PROMPT
# ================================
QUESTIONS = [
    "Rate the hazard level of this item on a scale of 1-10 where 1 is safe and 10 is extremely hazardous",
    "Which colored bin should be used to dispose of this waste according to SWM Rules?,",
    "What materials is this waste object primarily made of?",
    "Which category does this waste belong to according to Indian Solid Waste Management Rules 2016?",
    "What kind of waste is visible in the image?"
]

COMPREHENSIVE_PROMPT = f"""
You are an expert waste management consultant specializing in India's Solid Waste Management Rules 2016. 

Analyze the image carefully and answer each question in three to four sentences.

IMPORTANT: Answer in this exact format:
Q1: [Your complete sentence answer]
Q2: [Your complete sentence answer]
...and so on for all questions.

Make your answers conversational and practical, like: "This plastic bottle should go in the blue bin."

KNOWLEDGE BASE:
- Dry Waste (Blue Bin): Recyclables like paper, plastic, metal, glass
- Wet Waste (Green Bin): Biodegradable items like food scraps, garden waste  
- Sanitary Waste (Red Bin): Diapers, sanitary pads, medical waste
- Hazardous Waste (Black Bin): Batteries, chemicals, medicines, CFLs
- E-waste: Electronics, phones, computers, circuit boards
- Composite Waste: Multi-material items like tetra packs, shoes

QUESTIONS:
{QUESTIONS}
"""

# ================================
# MODEL CLASS
# ================================
class WasteClassifier:
    def __init__(self):
        print(f"Loading {Config.MODEL_NAME}...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            token=Config.TOKEN,
        )

        self.processor = AutoProcessor.from_pretrained(
            Config.MODEL_NAME,
            token=Config.TOKEN,
            max_pixels=Config.MAX_PIXELS
        )

        print("Model loaded successfully")

    def analyze_image(self, image_path: str) -> dict:
        """Analyze image and return structured results"""

        for attempt in range(Config.MAX_RETRIES):
            try:
                # Prepare input
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": COMPREHENSIVE_PROMPT},
                        {"type": "image", "image": image_path}  # pass path instead of PIL.Image
                    ]
                }]

                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

                # Processor with vision info
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)

                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=Config.MAX_NEW_TOKENS,
                        temperature=Config.TEMPERATURE,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                # Extract response
                generated = outputs[0][inputs.input_ids.shape[-1]:]
                response = self.processor.decode(generated, skip_special_tokens=True).strip()

                # Parse answers
                answers = self.parse_qa_response(response)

                if len(answers) >= 10:  # At least 10 valid answers
                    return {
                        'success': True,
                        'attempt': attempt + 1,
                        'raw_response': response,
                        'answers': answers
                    }
                else:
                    print(f"Only got {len(answers)} answers, retrying...")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)

        return {
            'success': False,
            'attempt': Config.MAX_RETRIES,
            'answers': [f"FAILED_Q{i+1}" for i in range(15)]
        }

    def parse_qa_response(self, response: str) -> list[str]:
        """Parse Q1:, Q2: format responses"""
        answers = []

        # Split by lines and look for Q1:, Q2: patterns
        lines = response.split('\n')
        current_answer = ""

        for line in lines:
            line = line.strip()

            # Check if line starts with Q followed by number and colon
            match = re.match(r'^Q(\d+):\s*(.*)$', line, re.IGNORECASE)
            if match:
                # Save previous answer if exists
                if current_answer:
                    answers.append(current_answer.strip())

                # Start new answer
                current_answer = match.group(2)
            elif current_answer and line:
                # Continue current answer on new line
                current_answer += " " + line

        # Add the last answer
        if current_answer:
            answers.append(current_answer.strip())

        # Ensure we have exactly 15 answers
        while len(answers) < 15:
            answers.append(f"MISSING_ANSWER_Q{len(answers)+1}")

        return answers[:15]  # Take only first 15

# ================================
# DATASET PROCESSING
# ================================
def collect_images(dataset_root: str):
    """Collect all valid images from dataset"""
    image_files = []

    if not os.path.exists(dataset_root):
        print(f"Dataset directory '{dataset_root}' not found")
        return []

    for category in sorted(os.listdir(dataset_root)):
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                full_path = os.path.join(category_path, filename)
                try:
                    # Validate image
                    with Image.open(full_path) as img:
                        img.verify()
                    image_files.append((full_path, category))
                except:
                    print(f"Skipping invalid image: {full_path}")

    return image_files

def save_results(results):
    """Save results to CSV"""
    with open(Config.OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class', 'question', 'answer', 'success', 'attempts'])

        for result in results:
            for i, (question, answer) in enumerate(zip(QUESTIONS, result['answers'])):
                writer.writerow([
                    result['image_path'],
                    result['class'],
                    question,
                    answer,
                    result['success'],
                    result.get('attempt', 0)
                ])

    print(f"Results saved to {Config.OUTPUT_CSV}")

def main():
    """Main processing function"""
    # Initialize classifier
    classifier = WasteClassifier()

    # Collect images
    image_files = collect_images(Config.DATASET_ROOT)
    if not image_files:
        print("No valid images found!")
        return

    print(f"Found {len(image_files)} images across {len(set(cat for _, cat in image_files))} categories")

    # Process images
    results = []
    stats = {'total': len(image_files), 'successful': 0, 'failed': 0}

    for image_path, category in tqdm(image_files, desc="Analyzing images"):
        try:
            result = classifier.analyze_image(image_path)

            # Add metadata
            result['image_path'] = image_path
            result['class'] = category

            results.append(result)

            if result['success']:
                stats['successful'] += 1
                print(f"{os.path.basename(image_path)} - Success in {result['attempt']} attempts")
            else:
                stats['failed'] += 1
                print(f"{os.path.basename(image_path)} - Failed after {result['attempt']} attempts")

        except Exception as e:
            stats['failed'] += 1
            print(f"Critical error processing {image_path}: {e}")

            results.append({
                'image_path': image_path,
                'class': category,
                'success': False,
                'attempt': 0,
                'answers': [f"ERROR_Q{i+1}: {str(e)}" for i in range(15)]
            })

    # Save results
    save_results(results)

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    if stats['total'] > 0:
        print(f"Success rate: {stats['successful']/stats['total']*100:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    try:
        print("Starting Waste Classification Analysis...")
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
