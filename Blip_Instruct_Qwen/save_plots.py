import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
csv_path = "instructor_blip_freeze_unfreeze/loss_tracking.csv"
loss_curve_path = "instructor_blip_freeze_unfreeze/loss_curve.png" # linear loss
loss_curve_zoom_path = "instructor_blip_freeze_unfreeze/loss_curve_epoch.png" # will now be log-scale loss
lr_schedule_path = "instructor_blip_freeze_unfreeze/lr_schedule.png" # LR linear
convergence_plot_path = "instructor_blip_freeze_unfreeze/convergence_plot.png"

# Load CSV
df = pd.read_csv(csv_path)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create combined metrics plot (2x2 grid)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LLAVA Training Metrics Analysis', fontsize=16, weight='bold')

# === 1. Training and Validation Loss (Linear Scale) ===
ax1.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color="#1f77b4", linewidth=2)
ax1.plot(df["Epoch"], df["Val Loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training and Validation Loss (Linear Scale)", fontsize=14, weight='bold')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)

# === 2. Training and Validation Loss (Logarithmic Scale) ===
ax2.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color="#1f77b4", linewidth=2)
ax2.plot(df["Epoch"], df["Val Loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
ax2.set_yscale("log")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss (Log Scale)", fontsize=12)
ax2.set_title("Training and Validation Loss (Logarithmic Scale)", fontsize=14, weight='bold')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)

# === 3. Learning Rate Schedule ===
ax3.plot(df["Epoch"], df["Learning Rate"], label="Learning Rate", color="#2ca02c", linewidth=2)
ax3.set_xlabel("Epoch", fontsize=12)
ax3.set_ylabel("Learning Rate (Log Scale)", fontsize=12)
ax3.set_title("Learning Rate Schedule", fontsize=14, weight='bold')
ax3.set_yscale("log")
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.4)

# === 4. Loss Change Rate (Convergence Analysis) ===
# Calculate the absolute change rate in loss between consecutive epochs
train_loss_change = np.abs(np.diff(df["Train Loss"]))
val_loss_change = np.abs(np.diff(df["Val Loss"]))

# For plotting, we'll use epochs 1 to n-1 since diff reduces length by 1
epochs_diff = df["Epoch"][1:].values

ax4.plot(epochs_diff, train_loss_change, label="Train Loss Change Rate", color="#1f77b4", linewidth=2)
ax4.plot(epochs_diff, val_loss_change, label="Val Loss Change Rate", color="#ff7f0e", linewidth=2)
ax4.set_yscale("log")
ax4.set_xlabel("Epoch", fontsize=12)
ax4.set_ylabel("Absolute Loss Change Rate (Log Scale)", fontsize=12)
ax4.set_title("Loss Change Rate (Convergence Analysis)", fontsize=14, weight='bold')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# === Individual plots for backward compatibility ===

# 1. Loss vs Epoch (Linear Scale)
plt.figure(figsize=(12, 6))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color="#1f77b4", linewidth=2)
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss vs Epoch (Linear Scale)", fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("instructor_blip_freeze_unfreeze/loss_curve_linear.png", dpi=300)
plt.close()

# 2. Loss vs Epoch (Log Scale)
plt.figure(figsize=(12, 6))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color="#1f77b4", linewidth=2)
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (Log Scale)", fontsize=12)
plt.title("Loss vs Epoch (Log Scale)", fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(loss_curve_zoom_path, dpi=300)
plt.close()

# 3. Learning Rate vs Epoch
plt.figure(figsize=(12, 6))
plt.plot(df["Epoch"], df["Learning Rate"], label="Learning Rate", color="#2ca02c", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Learning Rate (Log Scale)", fontsize=12)
plt.title("Learning Rate vs Epoch", fontsize=14, weight='bold')
plt.yscale("log")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(lr_schedule_path, dpi=300)
plt.close()

# 4. Convergence Plot
plt.figure(figsize=(12, 6))
plt.plot(epochs_diff, train_loss_change, label="Train Loss Change Rate", color="#1f77b4", linewidth=2)
plt.plot(epochs_diff, val_loss_change, label="Val Loss Change Rate", color="#ff7f0e", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Absolute Loss Change Rate (Log Scale)", fontsize=12)
plt.title("Loss Change Rate (Convergence Analysis)", fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(convergence_plot_path, dpi=300)
plt.close()

print(" Combined plots saved successfully!")
print(f"- {loss_curve_path} (Combined 2x2 Grid)")
print(f"- {loss_curve_zoom_path} (Log Scale Loss)")
print(f"- {lr_schedule_path} (Learning Rate)")
print(f"- {convergence_plot_path} (Convergence Analysis)")
print("- instructor_blip_freeze_unfreeze/loss_curve_linear.png (Linear Scale Loss)")