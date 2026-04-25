import os
import csv
import time
import base64
import re
from PIL import Image
from io import BytesIO
import ollama
import logging

# Enhanced Config
dataset_root = "dataset"
output_csv = "vqa_generated.csv"
model_name = "llava"
temperature = 0.3  # Lower temperature for more consistent, factual responses
max_retries = 5
retry_delay = 2  # seconds between retries

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced VQA Questions aligned with SWM Rules and BLIP training
questions = [
    "What type of waste is represented in the image?",
    "Which category does this waste belong to according to Indian Solid Waste Management Rules 2016?",
    "Which colored bin should be used to dispose of this waste according to SWM Rules?",
    "Is this waste biodegradable, non-biodegradable, or hazardous?",
    "Is this item recyclable through municipal systems in India?",
    "What preprocessing steps should be taken before disposing of this waste?",
    "Does this item require special handling or segregation according to SWM guidelines?",
    "What environmental impact may result from improper disposal of this waste?",
    "Can this waste be composted at household level under SWM Rules?",
    "Does improper disposal of this item violate SWM Rules 2016?",
    "Rate the hazard level of this item on a scale of 1-10 where 1 is safe and 10 is extremely hazardous",
    "Can this item be practically reused or upcycled before disposal?",
    "What is the estimated decomposition time for this waste in landfill conditions?",
    "What visible material properties indicate the waste classification?",
    "Which municipal authority or collection system handles this waste type in India?"
]

# ENHANCED SWM Rules 2016 Knowledge Base with EXPLICIT COMPOSTING RULES
swm_knowledge_base = """
COMPREHENSIVE WASTE CLASSIFICATION - INDIA SWM RULES 2016:

ITEMS THAT CAN BE COMPOSTED (HOME COMPOSTING ALLOWED):
Food waste: vegetable peels, fruit peels, fruit rinds, leftover cooked food, rice, dal, curry
Kitchen scraps: tea leaves, coffee grounds, eggshells, onion peels, potato peels
Garden waste: fallen leaves, flower petals, small twigs, grass clippings, plant trimmings
Natural organic materials: cotton cloth pieces, jute materials, coconut fiber, wood shavings
Food-soiled paper: napkins used for food, tissues with food residue only

ITEMS THAT CANNOT BE COMPOSTED (NEVER COMPOSTABLE):
ALL METALS: aluminum cans, steel cans, tin cans, bottle caps, foil, any metal containers
ALL PLASTICS: plastic bottles, containers, bags, wrappers, synthetic materials, PET bottles
ALL GLASS: glass bottles, jars, containers, broken glass pieces
PAPER & CARDBOARD: newspapers, magazines, books, cardboard boxes, packaging (SWM 2016 classifies as non-biodegradable)
SYNTHETIC TEXTILES: polyester clothes, nylon, synthetic fabrics, shoes
ELECTRONICS: any electronic items, batteries, wires, components
CHEMICALS: cleaning products, paints, pesticides, medicines
SANITARY WASTE: diapers, sanitary pads, masks, bandages
PROCESSED MATERIALS: laminated items, multi-layer packaging, tetra packs

1. BIODEGRADABLE/WET WASTE (GREEN BIN):
- Food waste: vegetable peels, fruit rinds, cooked food, tea leaves, coffee grounds
- Garden waste: leaves, flowers, small twigs, grass clippings
- Natural organic materials: cotton cloth, jute, wood shavings, eggshells
- Food-soiled paper: napkins, tissues with food residue
- Decomposition: 2 weeks to 6 months
- Collection: Daily door-to-door pickup
- Processing: Home composting (pit composting, vermicomposting, aerobic composters), municipal composting, biogas generation
- COMPOSTING: CAN BE COMPOSTED - Turn regularly, maintain 50-60% moisture, carbon:nitrogen ratio 25-30:1, ready in 3-6 months

2. DRY/NON-BIODEGRADABLE WASTE (BLUE BIN):
- Paper products: newspapers, magazines, books, office paper, cardboard, packaging paper
- Plastics: bottles (PET), containers, bags (clean and dry), packaging films
- Metals: aluminum cans, steel containers, foil, bottle caps, tin cans, metal containers
- Glass: bottles, jars, containers
- Textiles: synthetic clothes, polyester, nylon fabrics
- Decomposition: 20-1000+ years depending on material
- Preprocessing: Clean, dry, remove labels/caps where possible
- Collection: Every alternate day
- Processing: Recycling, material recovery facilities
- COMPOSTING: CANNOT BE COMPOSTED - All items in this category are non-compostable
- NOTE: ALL paper, cardboard, metals, plastics, glass go in DRY WASTE and CANNOT BE COMPOSTED

3. SANITARY/REJECT WASTE (WHITE/YELLOW BAG):
- Personal hygiene: diapers, sanitary pads, tampons
- Medical waste: bandages, cotton swabs, masks, used tissues
- Human waste contact items: soiled tissues, personal care items
- Pet waste: cat litter, pet hygiene products
- Preprocessing: Wrap in newspaper, mark with red cross
- Collection: Separate handover to waste collector
- Processing: Incineration, secure landfilling
- COMPOSTING: CANNOT BE COMPOSTED - These items are never compostable

4. DOMESTIC HAZARDOUS WASTE (RED BAG - SEPARATE COLLECTION):
- Electronics: batteries (all types), CFL/LED bulbs, small electronic items
- Chemicals: paints, pesticides, cleaning agents, expired medicines
- Sharp objects: razor blades, needles, broken glass
- Toxic materials: nail polish, hair dye, printer cartridges, mercury thermometers
- Preprocessing: Keep in original packaging when possible, label clearly
- Collection: Special collection drives, designated drop-off points, pharmacy take-back
- Processing: Treatment at authorized hazardous waste treatment facilities
- COMPOSTING: CANNOT BE COMPOSTED - All hazardous items are never compostable

5. E-WASTE (SEPARATE COLLECTION BOX):
- Large electronics: computers, TVs, refrigerators, washing machines, air conditioners
- Small electronics: phones, tablets, chargers, cables, earphones
- Components: circuit boards, hard drives, monitors, keyboards
- Accessories: mice, speakers, printers, cameras
- Collection: Extended Producer Responsibility, authorized e-waste collectors
- Processing: Dismantling, precious metal recovery, safe disposal of toxic components
- COMPOSTING: CANNOT BE COMPOSTED - No electronic items can be composted

6. CONSTRUCTION & DEMOLITION WASTE (C&D):
- Materials: concrete, bricks, tiles, ceramics, soil, stones, sand
- Wood: construction timber, large furniture pieces
- Metals: iron rods, steel beams, aluminum sections
- Collection: Not door-to-door, requires ULB notification and special arrangement
- Processing: Crushing, recycling for new construction materials
- COMPOSTING: CANNOT BE COMPOSTED - Construction materials are never compostable

7. COMPOSITE/MULTI-LAYERED PACKAGING (BLUE BIN - LIMITED RECYCLABILITY):
- Examples: chip packets, chocolate wrappers, tetra packs, laminated pouches
- Challenge: Multiple materials bonded together (plastic + aluminum + paper)
- Current classification: Goes to dry waste bin but has limited recyclability
- Extended Producer Responsibility (EPR) programs increasingly handling these
- Processing: Advanced recycling, energy recovery, or secure landfilling
- COMPOSTING: CANNOT BE COMPOSTED - Multi-layer packaging is never compostable

CRITICAL COMPOSTING RULES:
- ONLY organic materials from GREEN BIN category can be composted
- ALL metals (including cans, foil, containers) CANNOT be composted under any circumstances
- ALL plastics CANNOT be composted under any circumstances
- ALL glass CANNOT be composted under any circumstances
- Paper and cardboard CANNOT be composted as per SWM 2016 (classified as dry waste)
- When in doubt: If it's not fresh organic matter, it CANNOT be composted

BIN COLOR CODING SYSTEM:
- GREEN: Biodegradable/Wet waste (ONLY these items can be composted)
- BLUE: Dry/Recyclable waste (NEVER compostable)
- WHITE/YELLOW: Sanitary and reject waste (NEVER compostable)
- RED: Hazardous waste (NEVER compostable)

COLLECTION FREQUENCY:
- Green waste: Daily collection
- Blue waste: Alternate days
- White/Yellow waste: On-demand/weekly
- Red waste: Special collection events/drives

HOME COMPOSTING GUIDELINES (ONLY FOR GREEN BIN ITEMS):
- Suitable materials: Vegetable peels, fruit waste, tea/coffee grounds, eggshells, flowers, leaves
- NOT suitable: Any metal, plastic, glass, paper, synthetic materials
- Methods: Pit composting, vermicomposting, aerobic composters, bokashi composting
- Maintenance: Turn regularly, maintain proper moisture (50-60%), ensure air circulation
- C:N Ratio: Mix green materials (nitrogen) with brown materials (carbon) in 1:3 ratio
- Timeline: 3-6 months for finished compost
- Troubleshooting: Add dry leaves if too wet, add water if too dry, turn more frequently if smelly

CITIZEN RESPONSIBILITIES:
- Segregate waste at source (mandatory under SWM Rules 2016)
- Clean and dry recyclables before disposal
- Wrap sanitary waste appropriately
- Attempt home composting ONLY of wet waste from GREEN BIN if space and conditions permit
- Use authorized collectors for hazardous/e-waste
- Pay prescribed user charges to municipality

PENALTIES: Non-compliance with segregation can result in fines as per municipal bylaws 
"""

# Enhanced Prompt Template with EXPLICIT composting instructions
combined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
prompt_template = f"""You are an expert waste management consultant specializing in India's Solid Waste Management Rules 2016. Analyze the image carefully and provide accurate, practical answers.

KNOWLEDGE BASE:
{swm_knowledge_base}

CRITICAL COMPOSTING RULES - READ CAREFULLY:
- ONLY items that go in GREEN BIN can be composted (food scraps, garden waste, natural organic materials)
- ALL METALS (cans, containers, foil) CANNOT be composted - they go in BLUE BIN
- ALL PLASTICS (bottles, containers, bags) CANNOT be composted - they go in BLUE BIN  
- ALL GLASS (bottles, jars) CANNOT be composted - they go in BLUE BIN
- Paper and cardboard CANNOT be composted as per SWM 2016 - they go in BLUE BIN
- If you see metal, plastic, glass, or synthetic materials - the answer to composting is always "NO, cannot be composted"

INSTRUCTIONS:
- Examine the image carefully to identify materials, colors, shapes, and any text/labels
- Apply SWM Rules 2016 classification strictly and accurately
- For composting questions: If the item is metal, plastic, glass, or synthetic material - answer "No, cannot be composted"
- Remember: ALL paper and cardboard are NON-BIODEGRADABLE and go in BLUE BIN as per SWM 2016
- Provide specific, actionable answers based on visual evidence
- Each answer should be 30-80 words, factual and clear
- Use only standard alphanumeric characters and basic punctuation
- If uncertain about specific details, base answers on most likely classification from visible features
- Avoid vague responses like "depends" or "varies" - provide the most applicable standard answer

QUESTIONS TO ANSWER:
{combined_questions}

Format your response as numbered answers (1. answer, 2. answer, etc.) with NO additional text or explanations.
"""

# Enhanced image encoding with validation
def image_to_base64(path):
    try:
        with Image.open(path) as img:
            # Validate image
            if img.size[0] < 32 or img.size[1] < 32:
                logger.warning(f"Image {path} too small: {img.size}")
                return None

            # Convert and resize if too large
            img = img.convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            buf = BytesIO()
            img.save(buf, format='JPEG', quality=85)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"Failed to encode image {path}: {e}")
        return None

# Enhanced answer validation and cleaning with composting validation
def clean_and_validate_answer(ans, question_text=""):
    if not ans or ans.strip() == "":
        return None

    # Clean answer
    cleaned = re.sub(r"[^\w\s\.-]", "", ans).strip()

    # Special validation for composting questions
    if "compost" in question_text.lower():
        # Check for contradictory composting answers
        metal_keywords = ["can", "metal", "aluminum", "steel", "tin", "bottle cap", "foil"]
        plastic_keywords = ["plastic", "bottle", "container", "bag", "synthetic"]
        glass_keywords = ["glass", "jar", "bottle"]

        item_is_non_compostable = any(keyword in cleaned.lower() for keyword in metal_keywords + plastic_keywords + glass_keywords)
        answer_says_compostable = any(phrase in cleaned.lower() for phrase in ["yes", "can be composted", "compostable", "suitable for composting"])

        if item_is_non_compostable and answer_says_compostable:
            logger.warning(f"Detected contradictory composting answer: {cleaned}")
            return "No, cannot be composted. This item goes in blue bin for recycling."

    # Check for invalid responses
    invalid_patterns = [
        r"^(n/?a|not available|unknown|unclear|cannot determine)$",
        r"^(depends|varies|not sure|maybe|possibly)$",
        r"^\s*$",
        r"^[\d\.\s]*$"  # Only numbers and dots
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, cleaned.lower()):
            return None

    # Minimum length check
    if len(cleaned.split()) < 2:
        return None

    return cleaned

# Query model with retry logic
def query_model_with_retry(img_b64, max_retries=5):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")

            response = ollama.chat(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": prompt_template,
                    "images": [img_b64]
                }],
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )

            raw_answer = response["message"]["content"].strip()
            if not raw_answer:
                raise ValueError("Empty response from model")

            return raw_answer

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise

# Parse and validate answers with enhanced composting validation
def parse_answers(raw_answer, questions):
    answer_lines = raw_answer.split("\n")
    parsed_answers = []

    for i, question in enumerate(questions):
        answer = None

        # Try to find answer by number
        for line in answer_lines:
            if line.strip().startswith(f"{i+1}."):
                answer = line.split(".", 1)[-1].strip()
                break

        # Fallback: use line by index
        if not answer and i < len(answer_lines):
            line = answer_lines[i].strip()
            answer = line.split(".", 1)[-1].strip() if '.' in line else line

        # Validate and clean with question context
        validated_answer = clean_and_validate_answer(answer, question)
        parsed_answers.append(validated_answer if validated_answer else "Unable to determine from image")

    return parsed_answers

# Create CSV with headers
def initialize_csv():
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or os.stat(output_csv).st_size == 0:
            writer.writerow(['image_path', 'category', 'question', 'answer'])

# Main processing function
def process_dataset():
    initialize_csv()

    total_images = 0
    processed_images = 0
    failed_images = 0

    # Count total images
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_images += len(images)

    logger.info(f"Found {total_images} images to process")

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for category in sorted(os.listdir(dataset_root)):
            category_path = os.path.join(dataset_root, category)
            if not os.path.isdir(category_path):
                continue

            logger.info(f"Processing category: {category}")
            images = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_filename in sorted(images):
                img_path = os.path.join(category_path, img_filename)
                logger.info(f"Processing: {img_path} ({processed_images + 1}/{total_images})")

                # Encode image
                img_b64 = image_to_base64(img_path)
                if not img_b64:
                    logger.error(f"Failed to encode {img_path}")
                    failed_images += 1
                    continue

                try:
                    # Query model with retries
                    raw_answer = query_model_with_retry(img_b64, max_retries)

                    # Parse and validate answers
                    answers = parse_answers(raw_answer, questions)

                    # Check if we got valid answers
                    valid_answers = sum(1 for ans in answers if ans != "Unable to determine from image")
                    if valid_answers < len(questions) * 0.7:  # At least 70% valid answers
                        logger.warning(f"Too many invalid answers for {img_path}, retrying...")
                        raw_answer = query_model_with_retry(img_b64, max_retries)
                        answers = parse_answers(raw_answer, questions)

                    # Write to CSV
                    for question, answer in zip(questions, answers):
                        writer.writerow([img_path, category, question, answer])

                    processed_images += 1
                    logger.info(f"Successfully processed {img_path}")

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {e}")
                    failed_images += 1

                    # Write error entries
                    for question in questions:
                        writer.writerow([img_path, category, question, f"Processing error: {str(e)[:50]}"])

    logger.info(f"Processing complete! Processed: {processed_images}, Failed: {failed_images}, Total: {total_images}")

# Run the script
if __name__ == "__main__":
    try:
        process_dataset()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
