import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import uuid
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_augmentation_pipeline(image_height, image_width):
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.4),
        A.ColorJitter(p=0.3),
        A.Blur(p=0.2),
        A.Resize(224,224)
    ]

        

    return A.Compose(transforms)


def augment_dataset(data, output_dir="augmented_images", augment_times=2, seed=42):
    """
    Augments VQA dataset and preserves original + N augmentations per image.

    Args:
        data (list of dict): Each dict must contain keys:
            'image_path', 'question', 'answer', 'label', 'question_id'
        output_dir (str): Directory to store augmented images
        augment_times (int): Number of augmentations per image

    Returns:
        List[dict]: Combined list of original + augmented QA entries
    """
    # Seed numpy so albumentations probabilistic transforms are reproducible
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    final_data = []
    skipped_images = []

    for item in tqdm(data, desc="Augmenting dataset"):
        orig_img_path = item['image_path']
        orig_img = cv2.imread(orig_img_path)

        if orig_img is None:
            print(f"Failed to load: {orig_img_path}")
            skipped_images.append(orig_img_path)
            continue

        h, w = orig_img.shape[:2]
        aug_pipeline = get_augmentation_pipeline(h, w)

        # === Include original entry ===
        final_data.append(item)

        # === Generate augmentations ===
        for _ in range(augment_times):
            try:
                aug_img = aug_pipeline(image=orig_img)['image']
            except Exception as e:
                print(f"Augmentation failed for {orig_img_path}: {e}")
                continue

            # Save augmented image
            aug_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(orig_img_path)}"
            aug_path = os.path.join(output_dir, aug_filename)
            cv2.imwrite(aug_path, aug_img)

            # Create new QA record
            final_data.append({
                'image_path': aug_path,
                'question': item['question'],
                'answer': item['answer'].lower(),
                'label': item['label'],
                'question_id': item.get('question_id', 'original')
            })

    print(f"Augmentation complete. Total QA pairs: {len(final_data)}")
    if skipped_images:
        print(f"Skipped {len(skipped_images)} broken or unreadable images.")
        with open("skipped_images.txt", "w") as f:
            f.write("\n".join(skipped_images))

    return final_data





def split_augmented_data(augmented_data, test_size=0.3, val_size=0.5, seed=42):
    # Get all unique image paths
    unique_images = list(set([item['image_path'] for item in augmented_data]))

    train_imgs, temp_imgs = train_test_split(unique_images, test_size=test_size, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1 - val_size, random_state=seed)

    train_set = [item for item in augmented_data if item['image_path'] in train_imgs]
    val_set = [item for item in augmented_data if item['image_path'] in val_imgs]
    test_set = [item for item in augmented_data if item['image_path'] in test_imgs]

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    return train_set, val_set, test_set #list of dict
