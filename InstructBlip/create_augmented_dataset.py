from dataset import VQADatasetCreator, create_dataloaders
from augment_dataset_utils import augment_dataset, split_augmented_data
import os

# === Root dataset directory (one level up from this script) ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 0: Load and expand
creator = VQADatasetCreator(json_path="vqa_final.json")
creator.load_json_data()
expanded = creator.expand_data()

# Step 1: Augment and triple the dataset (original + 3 augmentations)
augmented_data = augment_dataset(
    expanded,
    output_dir=os.path.join(ROOT_DIR, "augmented_images"),
    augment_times=3,
)

# Step 2: Split into train/val/test
train, val, test = split_augmented_data(augmented_data)

# Step 3: Save splits (optional)
creator.save_splits(train, val, test, output_dir=os.path.join(ROOT_DIR, "split_jsons"))
