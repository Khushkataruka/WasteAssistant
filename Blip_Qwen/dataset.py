import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Dict, Tuple


class VQADatasetCreator:
    """
    Creates VQA dataset from JSON with questions and answers
    """
    def __init__(self, json_path: str):
        """
        Initialize with path to JSON file
        
        Args:
            json_path: Path to JSON file containing questions and data
        """
        self.json_path = json_path
        self.questions = None
        self.data = None
        self.expanded_data = None
        
    def load_json_data(self):
        """Load data from JSON file"""
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)
        
        self.questions = json_data['questions']
        self.data = json_data['data']
        
        print(f"Loaded {len(self.questions)} questions")
        print(f"Loaded {len(self.data)} images")
        print(f"Total QA pairs: {len(self.questions) * len(self.data)}")
        
    def expand_data(self) -> List[Dict]:
        """
        Expand data to create individual question-answer pairs
        
        Returns:
            List of dictionaries with image_path, question, answer, label
        """
        expanded_data = []
        for item in self.data:
            image_path = item['image_path']
            label = item['label']
            answers = item['answers']
         
            # Ensure we have answers for all questions
            if len(answers) != len(self.questions):
                print(f"Warning: Mismatch in answers count for {image_path}")
                continue
            # Create QA pairs
            for i, (question, answer) in enumerate(zip(self.questions, answers)):
                if answer!='NaN':
                    expanded_data.append({
                        'image_path': image_path,
                        'question': question,
                        'answer': answer.lower(),
                        'label': label,
                        'question_id': i
                    })
                else:
                    print("SKIP\n")
        self.expanded_data = expanded_data
        return expanded_data
    
    def create_splits(self, 
                     test_size: float = 0.3,
                     val_size: float = 0.5,  # This will be 0.5 of the remaining 30%, so 15% of total
                     random_seed: int = 42) -> Tuple[List, List, List]:
        """
        Create train/validation/test splits using sklearn's train_test_split
        
        Args:
            test_size: Proportion for test + validation (default 0.3 = 30%)
            val_size: Proportion of test_size for validation (default 0.5, so 15% of total)
            split_by: 'image' or 'qa_pair' - how to split the data
            stratify_by: 'label' to maintain label distribution, None for no stratification
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        
            # Split by images to avoid data leakage
        unique_images = list(set([item['image_path'] for item in self.data]))
            
            
            # First split: train vs (val + test)
        train_images, temp_images = train_test_split(
                unique_images,
                test_size=test_size,
                random_state=random_seed
            )
            
            
        val_images, test_images = train_test_split(
                temp_images,
                test_size=1-val_size,  # If val_size=0.5, test_size=0.5 of temp
                random_state=random_seed
            )
            
            # Convert to sets for faster lookup
        train_images_set = set(train_images)
        val_images_set = set(val_images)
        test_images_set = set(test_images)
            
            # Filter expanded data based on image splits
        train_data = [item for item in self.expanded_data if item['image_path'] in train_images_set]
        val_data = [item for item in self.expanded_data if item['image_path'] in val_images_set]
        test_data = [item for item in self.expanded_data if item['image_path'] in test_images_set]
            
        
        print(f"Train: {len(train_data)} samples ({len(train_data)/len(self.expanded_data)*100:.1f}%)")
        print(f"Validation: {len(val_data)} samples ({len(val_data)/len(self.expanded_data)*100:.1f}%)")
        print(f"Test: {len(test_data)} samples ({len(test_data)/len(self.expanded_data)*100:.1f}%)")
        print(f"Split by images: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data: List, val_data: List, test_data: List, output_dir: str):
        """Save the splits to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'train.json'), 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(os.path.join(output_dir, 'val.json'), 'w') as f:
            json.dump(val_data, f, indent=2)
            
        with open(os.path.join(output_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f, indent=2)
            
        print(f"Splits saved to {output_dir}")
    
    def analyze_dataset(self):
        """Analyze the dataset for insights"""
        if not self.expanded_data:
            print("No expanded data available. Run expand_data() first.")
            return
        
        # Question distribution
        question_counts = {}
        for item in self.expanded_data:
            q_id = item['question_id']
            question_counts[q_id] = question_counts.get(q_id, 0) + 1
        
        # Answer length distribution
        answer_lengths = [len(item['answer'].split()) for item in self.expanded_data]
        
        # Label distribution
        label_counts = {}
        for item in self.expanded_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\n=== Dataset Analysis ===")
        print(f"Total QA pairs: {len(self.expanded_data)}")
        print(f"Unique images: {len(set([item['image_path'] for item in self.expanded_data]))}")
        print(f"Questions per image: {len(self.questions)}")
        
        print(f"\nAnswer length stats:")
        print(f"  Min: {min(answer_lengths)} words")
        print(f"  Max: {max(answer_lengths)} words") 
        print(f"  Avg: {sum(answer_lengths)/len(answer_lengths):.2f} words")
        
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} samples")


class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA training
    """
    def __init__(self, data: List[Dict], processor, max_length: int = 77):
        """
        Initialize VQA Dataset
        
        Args:
            data: List of dictionaries with image_path, question, answer
            processor: BLIP2 processor for tokenization and image processing
            max_length: Maximum sequence length
        """
        self.data = data
        self.processor = processor
        self.max_length = max_length
        
        # Filter out items with missing images
        self.valid_data = []
        for item in data:
            if os.path.exists(item['image_path']):
                self.valid_data.append(item)
            else:
                print(f"Warning: Image not found: {item['image_path']}")
        
        print(f"Dataset initialized with {len(self.valid_data)}/{len(data)} valid samples")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        try:
            # Load image
            image = Image.open(item['image_path']).convert('RGB')
            
            # Process inputs
            inputs = self.processor(
                image, 
                item['question'], 
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Process target answer
            targets = self.processor.tokenizer(
                item['answer'],
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=30  # Shorter for answers
            )
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze(),
                'image_path': item['image_path'],
                'question': item['question'],
                'answer': item['answer']
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a dummy item in case of error
            return self.__getitem__(0)


def create_dataloaders(train_dataset,test_dataset,val_dataset,batch_size=32,num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets
    """

    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader