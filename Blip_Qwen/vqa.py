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
from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(os.path.join(ROOT_DIR, ".env"))
HF_TOKEN = os.environ.get("HF_TOKEN")

# === Load Model and Processor ===
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", token=HF_TOKEN)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True, token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

# === Custom Dataset for VQA ===
class VQADataset(Dataset):
    def __init__(self, json_data, processor):
        self.data = json_data
        self.processor = processor

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
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (224, 224))  # fallback dummy image

        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=100, padding="max_length", truncation=True, return_tensors="pt"
        )
        encoding["labels"] = labels.squeeze()

        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding

# === Load JSON datasets ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

train_json = load_json(os.path.join(ROOT_DIR, "split_jsons", "train.json"))
val_json = load_json(os.path.join(ROOT_DIR, "split_jsons", "val.json"))

train_dataset = VQADataset(train_json, processor)
val_dataset = VQADataset(val_json, processor)

# === Seeded DataLoader for reproducible batch ordering ===
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# === Optimizer and Scheduler ===
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scaler = torch.cuda.amp.GradScaler()

# === Training Configuration ===
num_epochs = 100
patience = 10
best_loss = float("inf")
early_stopping_counter = 0
tracking_info = []
lr_history = []

save_dir = "blip_model_all_layers"
os.makedirs(save_dir, exist_ok=True)
for param in model.vision_model.parameters():
    param.requires_grad = False
print("Vision encoder frozen for first 10 epochs.")

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    if epoch == 10:
        for param in model.vision_model.parameters():
            param.requires_grad = True
        print("Vision encoder unfrozen.")
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    if optimizer.param_groups[0]["lr"] < 1e-6:
        optimizer.param_groups[0]["lr"] = 1e-6
    current_lr = optimizer.param_groups[0]["lr"]
    lr_history.append(current_lr)

    # === Validation Loop ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels
                )
            loss = outputs.loss
            val_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    tracking_info.append((epoch + 1, avg_train_loss, avg_val_loss))
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

    # === Save best model ===
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"Model saved at epoch {epoch+1} with val loss {best_loss:.4f}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
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
plt.title("BLIP VQA Training Loss Curve")
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

print(f"Final model and plots saved in: {save_dir}")
print(f"Loss and LR values saved to: {csv_path}")
