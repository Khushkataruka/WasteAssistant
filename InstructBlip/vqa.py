import os
import json
import random

# === Root dataset directory (one level up from this script) ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# === Load Model and Processor ===
model_name = "Salesforce/instructblip-flan-t5-xl"

# Load model with bfloat16 for stability
model = InstructBlipForConditionalGeneration.from_pretrained(
model_name,
torch_dtype=torch.bfloat16, # Use bfloat16 for better stability
device_map="auto" # Automatically distribute across available GPUs
)

# Load processor without torch_dtype (not needed)
processor = InstructBlipProcessor.from_pretrained(model_name)

# Optional: Sync processor config to model
processor.num_query_tokens = model.config.num_query_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model is already on device due to device_map="auto"

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

# === Freeze Vision Encoder Initially ===
for name, param in model.vision_model.named_parameters():
    param.requires_grad = False

# === Custom Dataset for VQA ===
class VQADataset(Dataset):
    def __init__(self, json_data):
        self.data = json_data

        def __len__(self):
            return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                image_path = item["image_path"]
                question = item["question"]
                answer = item["answer"]

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f" Error loading image {image_path}: {e}")
                    image = Image.new("RGB", (224, 224))

                    prompt = f"Question: {question} Answer:"
                    return {"image": image, "text": prompt, "answer": answer}

# === Custom Collate Function ===
def vqa_collate_fn(batch):
    images = [item["image"] for item in batch]
    prompts = [item["text"] for item in batch]
    answers = [item["answer"] for item in batch]

    try:
        # Process inputs (images + text prompts)
        processor_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
        )

        # Process target labels (answers)
        labels = processor.tokenizer(
        answers, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=getattr(processor.tokenizer, 'model_max_length', 512)
        ).input_ids

        # Replace padding tokens with -100 (ignored in loss calculation)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        processor_inputs["labels"] = labels
        return processor_inputs

    except Exception as e:
        print(f"Error in collate function: {e}")
        # Return a minimal batch if there's an error
        return {
        "input_ids": torch.tensor([[0]]),
        "attention_mask": torch.tensor([[1]]),
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "labels": torch.tensor([[-100]])
        }

# === Load JSON datasets ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

train_json = load_json(os.path.join(ROOT_DIR, "split_jsons", "train.json"))
val_json = load_json(os.path.join(ROOT_DIR, "split_jsons", "val.json"))

train_dataset = VQADataset(train_json)
val_dataset = VQADataset(val_json)

# === Seeded DataLoader for reproducible batch ordering ===
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=vqa_collate_fn, worker_init_fn=seed_worker, generator=g)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=vqa_collate_fn)

# === Optimizer and Scheduler ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=7e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# === Training Configuration ===
num_epochs = 50
patience = 10
best_loss = float("inf")
early_stopping_counter = 0
tracking_info = []
lr_history = []

save_dir = "instructor_blip_freeze_unfreeze"
os.makedirs(save_dir, exist_ok=True)

# === Gradual Unfreeze Strategy ===
unfreeze_epoch = 10

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    if epoch == unfreeze_epoch:
        print(" Unfreezing vision encoder...")
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = True
            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=7e-5)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
                # Move batch to device
                for k in batch:
                    batch[k] = batch[k].to(device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Use bfloat16 mixed precision (no GradScaler needed)
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**batch)
                        loss = outputs.loss

                        epoch_loss += loss.item()

                        # Backward pass (bfloat16 doesn't need scaler)
                        loss.backward()

                        # Optional: Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()

                        scheduler.step()
                        current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr < 1e-6:
                            optimizer.param_groups[0]["lr"] = 1e-6
                            lr_history.append(current_lr)

                            # === Validation Loop ===
                            model.eval()
                            val_loss = 0
                            with torch.no_grad():
                                for batch in tqdm(val_dataloader, desc="Validating"):
                                    for k in batch:
                                        batch[k] = batch[k].to(device)

                                        # Use same mixed precision as training
                                        with autocast(device_type="cuda", dtype=torch.bfloat16):
                                            outputs = model(**batch)
                                            loss = outputs.loss
                                            val_loss += loss.item()

                                            avg_train_loss = epoch_loss / len(train_dataloader)
                                            avg_val_loss = val_loss / len(val_dataloader)
                                            tracking_info.append((epoch + 1, avg_train_loss, avg_val_loss))

                                            print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

                                            if avg_val_loss < best_loss:
                                                best_loss = avg_val_loss
                                                model.save_pretrained(save_dir)
                                                processor.save_pretrained(save_dir)
                                                print(f" Model saved at epoch {epoch+1} with val loss {best_loss:.4f}")
                                                early_stopping_counter = 0
                                            else:
                                                early_stopping_counter += 1
                                                if early_stopping_counter >= patience:
                                                    print(f" Early stopping triggered after {epoch+1} epochs.")
                                                    break

# === Plot Loss Curves ===
epochs = [e for e, _, _ in tracking_info]
train_losses = [t for _, t, _ in tracking_info]
val_losses = [v for _, _, v in tracking_info]

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Val Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Instructor-BLIP VQA Training Loss Curve")
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(loss_plot_path)
plt.show()

# === Plot Learning Rate Curve ===
plt.figure()
plt.plot(epochs, lr_history, label="Learning Rate", color='orange', marker='^')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
lr_plot_path = os.path.join(save_dir, "lr_schedule.png")
plt.savefig(lr_plot_path)
plt.show()

# === Save Loss + LR to CSV ===
csv_path = os.path.join(save_dir, "loss_tracking.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Learning Rate"])
    for i in range(len(tracking_info)):
        writer.writerow([tracking_info[i][0], tracking_info[i][1], tracking_info[i][2], lr_history[i]])

print(f" Final model and plots saved in: {save_dir}")
print(f" Loss and LR values saved to: {csv_path}")
