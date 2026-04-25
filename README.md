# ♻️ WasteAssistant: Language-Guided Vision-AI System for Intelligent Waste Segregation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **WasteAssistant** is a language-guided Vision–AI framework that integrates Vision–Language Models (VLMs) with Multimodal Large Language Models (MLLMs) for joint visual–linguistic reasoning, aligned with India's **Solid Waste Management (SWM) Rules, 2016**.

---

## 📖 Overview

Efficient waste segregation is critical for sustainable urban management. Existing automated systems are limited by single-modality visual processing and lack of contextual understanding. **WasteAssistant** addresses these challenges by:

- Introducing a **Visual Question Answering (VQA)** paradigm explicitly aligned with India's SWM Rules (2016)
- Dynamically interpreting visual content and answering policy-driven queries such as:
  - *"Is this item recyclable under SWM 2016?"*
  - *"Which colored bin should this be disposed of in?"*
- Fine-tuning state-of-the-art VLMs (**BLIP** and **InstructBLIP**) on a custom regulation-aware dataset

---

## 🏗️ Architecture

The model architecture consists of three main components:

1. **Image Encoder** — Extracts visual features using Vision Transformers (ViT-B/16 for BLIP, ViT-L/14 for InstructBLIP)
2. **Image-Grounded Question Encoder** — Aligns question embeddings with image features via self-attention and cross-attention
3. **Answer Decoder** — Autoregressively generates policy-compliant answers using causal decoding

```
Image (I) ──► Image Encoder ──► Visual Features (V)
                                       │
Question (Q) ──► Question Encoder ─────┤──► Cross-Attention ──► H_Q
                                       │
                                       ▼
                              Answer Decoder ──► Answer (A)
```

### Models Used

| Model | Params | Base Model | Vision Encoder | GT Generator |
|-------|--------|------------|----------------|--------------|
| BLIP | 0.25B | vqa-base | ViT-B/16 | LLAVA (7B) |
| BLIP | 0.25B | vqa-base | ViT-B/16 | Qwen2-VL (7B) |
| InstructBLIP | 3B | Flan-T5-XL | ViT-L/14 | LLAVA (7B) |
| InstructBLIP | 3B | Flan-T5-XL | ViT-L/14 | Qwen2-VL (7B) |

---

## 📊 WasteVQA Dataset

**WasteVQA** is the first regulation-aware VQA dataset for solid waste management.

| Statistic | Value |
|-----------|-------|
| Total Images (incl. augmentations) | 6,000+ |
| Total Q&A Pairs | 13,500 |
| Waste Categories | 21 |
| Avg Q&A Pairs per Image | ~2.3 |
| Annotation Format | JSON + CSV |
| Source Datasets | TrashNet, TrashBox, BDwaste |

**Waste categories** include: Biodegradable, Recyclable (Plastic, Glass, Metal, Paper), Hazardous, E-waste, Sanitary, Biomedical, Domestic Hazardous, Composite, Inert, and more.

---

## 📈 Results

### Main Results: WasteVQA Test Set

| Model | LLAVA BLEU | LLAVA BERT | Qwen2-VL BLEU | Qwen2-VL BERT |
|-------|-----------|-----------|--------------|--------------|
| **BLIP** | **0.8291** | **0.9273** | **0.8785** | **0.9517** |
| InstructBLIP | 0.6345 | 0.7612 | 0.6798 | 0.7935 |

> BLIP + Qwen2-VL-7B achieves the best overall performance with BLEU = 0.8785 and BERTScore = 0.9517.

### Case Study: Real-World Waste Video Analysis

| Metric | BLIP (LLAVA) | InstructBLIP (LLAVA) | BLIP (Qwen) | InstructBLIP (Qwen) |
|--------|-------------|---------------------|------------|---------------------|
| Precision | 59.05 | 63.86 | 57.20 | 60.92 |
| F1-Score | 54.62 | 59.28 | 53.65 | 52.25 |
| BERTScore | 50.23 | 54.56 | 54.84 | 48.96 |

### Training Curves

<p align="center">
  <img src="Blip/blip_model_all_layers/loss_curve.png" width="400" alt="BLIP Loss Curve"/>
  <img src="InstructBlip/instructor_blip_freeze_unfreeze/loss_curve.png" width="400" alt="InstructBLIP Loss Curve"/>
</p>
<p align="center">
  <em>Left: BLIP training/validation loss. Right: InstructBLIP training/validation loss.</em>
</p>

<p align="center">
  <img src="Blip/blip_model_all_layers/lr_schedule.png" width="400" alt="BLIP LR Schedule"/>
  <img src="InstructBlip/instructor_blip_freeze_unfreeze/lr_schedule.png" width="400" alt="InstructBLIP LR Schedule"/>
</p>
<p align="center">
  <em>Left: BLIP learning rate schedule. Right: InstructBLIP learning rate schedule.</em>
</p>

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision transformers pillow tqdm nltk bert-score albumentations scikit-learn pandas matplotlib opencv-python python-dotenv
```

### Environment Setup

1. Copy the example environment file and add your HuggingFace token:
```bash
cp .env.example .env
# Edit .env and replace 'your_huggingface_token_here' with your actual token
```

2. Get a HuggingFace token at: https://huggingface.co/settings/tokens

---

### Full Pipeline (from scratch)

The dataset creation and training pipeline follows this order:

```
Step 1          Step 2             Step 3                Step 4         Step 5        Step 6
create_data.py → data_filtering.py → create_augmented_    → vqa.py      → testing.py  → save_plots.py
(GT generation)  (CSV → JSON)        dataset.py           (Training)    (Evaluation)   (Visualization)
                                     (Augment + Split)
```

#### Step 1: Generate Ground-Truth Answers

Uses a Multimodal LLM (LLAVA via Ollama or Qwen2-VL-7B via HuggingFace) to generate Q&A pairs for each image in the `dataset/` folder.

```bash
cd Blip/           # Uses LLAVA (via Ollama, must be running locally)
python create_data.py

cd Blip_Qwen/      # Uses Qwen2-VL-7B (via HuggingFace, requires GPU + HF_TOKEN)
python create_data.py
```

**Output:** `vqa_generated.csv` — raw Q&A pairs for all images

> **Note:** `Blip/` and `InstructBlip/` use **LLAVA** (requires [Ollama](https://ollama.com/) running locally).  
> `Blip_Qwen/` and `Blip_Instruct_Qwen/` use **Qwen2-VL-7B** (requires GPU and `HF_TOKEN` in `.env`).

#### Step 2: Filter & Structure the Dataset

Converts the raw CSV into a structured JSON format suitable for training.

```bash
python data_filtering.py
```

**Input:** `vqa_generated.csv` → **Output:** `vqa_final.json`

#### Step 3: Augment & Split

Applies image augmentations (rotation, flip, brightness, etc.) and creates train/val/test splits.

```bash
python create_augmented_dataset.py
```

**Output (at root level):**
- `augmented_images/` — augmented image files
- `split_jsons/train.json`, `val.json`, `test.json`

#### Step 4: Train the Model

```bash
# BLIP (LLAVA ground truth)
cd Blip/ && python vqa.py

# BLIP (Qwen ground truth)
cd Blip_Qwen/ && python vqa.py

# InstructBLIP (LLAVA ground truth)
cd InstructBlip/ && python vqa.py

# InstructBLIP (Qwen ground truth)
cd Blip_Instruct_Qwen/ && python vqa.py
```

**Output:** Model checkpoint saved to `blip_model_all_layers/` or `instructor_blip_freeze_unfreeze/`

#### Step 5: Evaluate on Test Set

```bash
python testing.py
```

**Output:**
- `vqa_eval_results.csv` (BLIP) or `instructblip_vqa_eval_results.csv` (InstructBLIP) — per-sample BLEU and BERTScore
- `instructblip_eval_summary.txt` — aggregate metrics (InstructBLIP only)

#### Step 6: Generate Plots (Optional)

```bash
python save_plots.py
```

---

### Case Study: Real-World Video Evaluation

This evaluates the trained model on frames extracted from a real-world waste disposal video.

#### Step 1: Generate Video Ground Truth

Uses Qwen2-VL-7B to generate ground-truth Q&A for each extracted video frame.

```bash
cd Blip_Qwen/  # or Blip_Instruct_Qwen/
# Ensure video frames are in output_frames/Frames/
# Then generate ground truth annotations:
python create_data.py  # Note: temporarily change Config.DATASET_ROOT to "output_frames" and Config.OUTPUT_CSV to "video_ground.csv"
```

> **Important:** For the case study, the `create_data.py` config needs to point to `output_frames/` instead of `dataset/`. The default config is set for dataset generation. Modify `Config.DATASET_ROOT` and `Config.OUTPUT_CSV` before running.

#### Step 2: Run VQA on Video Frames

```bash
python video_evaluation.py
```

**Output:** `frame_vqa_results.csv` — model predictions for each frame × question

#### Step 3: Compute Evaluation Metrics

```bash
python evaluate_res.py
```

**Input:** `frame_vqa_results.csv` + `video_ground.csv`  
**Output:** `full_evaluation_results.csv` — Precision, F1, BLEU, BERTScore per sample

---

## 🔧 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Total Epochs | 50 (with early stopping, patience=10) |
| Learning Rate | 4×10⁻⁵ (BLIP) / 5×10⁻⁵ (InstructBLIP) |
| LR Scheduler | ExponentialLR (γ=0.9) |
| Weight Decay | 5×10⁻⁶ (BLIP) / 7×10⁻⁵ (InstructBLIP) |
| Batch Size | 32 |
| Vision Encoder Frozen | First 10 epochs |
| Mixed Precision | FP16 (BLIP) / BF16 (InstructBLIP) |
| Random Seed | 42 (all libraries) |

---

## 📂 Directory Structure

```
FoundationModel/
├── README.md
├── .env.example                    # ← Template for HuggingFace token
├── .gitignore
├── split_jsons/                    # ← Shared dataset splits (generated)
│   ├── train.json
│   ├── val.json
│   └── test.json
├── augmented_images/               # ← Shared augmented images (generated)
│
├── Blip/                           # BLIP + LLAVA ground truth
│   ├── create_data.py              #   Step 1: GT generation (LLAVA via Ollama)
│   ├── data_filtering.py           #   Step 2: CSV → structured JSON
│   ├── create_augmented_dataset.py #   Step 3: Augmentation + split
│   ├── vqa.py                      #   Step 4: Training script
│   ├── testing.py                  #   Step 5: Evaluation script
│   ├── save_plots.py               #   Step 6: Plot generation
│   ├── dataset.py                  #   Dataset class & utilities
│   ├── augment_dataset_utils.py    #   Augmentation pipeline
│   ├── blip_model_all_layers/      #   Saved model checkpoint
│   │   ├── loss_curve.png
│   │   ├── lr_schedule.png
│   │   └── loss_tracking.csv
│   ├── vqa_eval_results.csv        #   ✅ BLEU=0.8291, BERT=0.9273
│   └── vqa_final.json              #   Source Q&A data
│
├── Blip_Qwen/                      # BLIP + Qwen2-VL-7B ground truth
│   ├── create_data.py              #   Step 1: GT generation (Qwen2-VL via HF)
│   ├── data_filtering.py           #   Step 2: CSV → structured JSON
│   ├── create_augmented_dataset.py #   Step 3: Augmentation + split
│   ├── vqa.py                      #   Step 4: Training
│   ├── testing.py                  #   Step 5: Evaluation
│   ├── save_plots.py               #   Step 6: Plot generation
│   ├── video_evaluation.py         #   Case study: VQA on video frames
│   ├── evaluate_res.py             #   Case study: Metric computation
│   ├── blip_model_all_layers/
│   ├── vqa_eval_results.csv        #   ✅ BLEU=0.8785, BERT=0.9517
│   ├── frame_vqa_results.csv       #   Video case study predictions
│   └── video_ground.csv            #   Video ground truth
│
├── InstructBlip/                   # InstructBLIP + LLAVA ground truth
│   ├── create_data.py              #   Step 1: GT generation (LLAVA via Ollama)
│   ├── data_filtering.py           #   Step 2: CSV → structured JSON
│   ├── create_augmented_dataset.py #   Step 3: Augmentation + split
│   ├── vqa.py                      #   Step 4: Training
│   ├── testing.py                  #   Step 5: Evaluation
│   ├── save_plots.py               #   Step 6: Plot generation
│   ├── instructor_blip_freeze_unfreeze/  # Saved model checkpoint
│   ├── instructblip_vqa_eval_results.csv # ✅ BLEU=0.6345, BERT=0.7612
│   └── instructblip_eval_summary.txt
│
└── Blip_Instruct_Qwen/            # InstructBLIP + Qwen2-VL-7B ground truth
    ├── create_data.py              #   Step 1: GT generation (Qwen2-VL via HF)
    ├── data_filtering.py           #   Step 2: CSV → structured JSON
    ├── create_augmented_dataset.py #   Step 3: Augmentation + split
    ├── vqa.py                      #   Step 4: Training
    ├── testing.py                  #   Step 5: Evaluation
    ├── save_plots.py               #   Step 6: Plot generation
    ├── video_evaluation.py         #   Case study: VQA on video frames
    ├── evaluate_res.py             #   Case study: Metric computation
    ├── instructor_blip_freeze_unfreeze/
    ├── instructblip_vqa_eval_results.csv # ✅ BLEU=0.6798, BERT=0.7935
    ├── instructblip_eval_summary.txt
    ├── frame_vqa_results.csv       #   Video case study predictions
    └── video_ground.csv            #   Video ground truth
```

### Results Location Quick Reference

| Result | File Location |
|--------|--------------|
| BLIP + LLAVA scores | `Blip/vqa_eval_results.csv` |
| BLIP + Qwen scores | `Blip_Qwen/vqa_eval_results.csv` |
| InstructBLIP + LLAVA scores | `InstructBlip/instructblip_vqa_eval_results.csv` |
| InstructBLIP + Qwen scores | `Blip_Instruct_Qwen/instructblip_vqa_eval_results.csv` |
| BLIP loss curve | `Blip/blip_model_all_layers/loss_curve.png` |
| InstructBLIP loss curve | `InstructBlip/instructor_blip_freeze_unfreeze/loss_curve.png` |
| BLIP LR schedule | `Blip/blip_model_all_layers/lr_schedule.png` |
| InstructBLIP LR schedule | `InstructBlip/instructor_blip_freeze_unfreeze/lr_schedule.png` |
| Video case study (BLIP+Qwen) | `Blip_Qwen/frame_vqa_results.csv` |
| Video case study (InstructBLIP+Qwen) | `Blip_Instruct_Qwen/frame_vqa_results.csv` |
| Video evaluation metrics | `Blip_Qwen/full_evaluation_results.csv` |

---

## 🔬 Evaluation Metrics

### BLEU-4
Sentence-level BLEU-4 with Chen & Cherry's **Smoothing Method 4** for numerical stability across variable-length answers.

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method4
bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
```

### BERTScore
Semantic similarity via BERTScore F1 with **baseline rescaling** for human-interpretable scores.

```python
from bert_score import score as bert_score
P, R, F1 = bert_score([prediction], [answer], lang="en", rescale_with_baseline=True)
```

### Case Study Metrics
- **Token Precision** — proportion of predicted tokens found in ground truth
- **Token F1-Score** — harmonic mean of token precision and recall
- **BERTScore** — contextual semantic similarity

---

## 🔄 Reproducibility

All scripts are fully seeded (`SEED=42`) across:

- `random.seed(42)` — Python stdlib
- `np.random.seed(42)` — NumPy / albumentations augmentations
- `torch.manual_seed(42)` — PyTorch CPU
- `torch.cuda.manual_seed_all(42)` — PyTorch CUDA
- `torch.backends.cudnn.deterministic = True` — CuDNN determinism
- `DataLoader` with `worker_init_fn` and seeded `Generator`
- `train_test_split(random_state=42)` — Data splitting

The saved model checkpoints and `split_jsons/` are provided, so reported metrics can be exactly reproduced by running `testing.py` against the pre-trained checkpoints.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{wasteassistant2025,
  title={WasteAssistant: Language-Guided Vision-AI System for Intelligent Waste Segregation},
  author={Khush and Harshit and Anuja and Murari Mandal and Praveen},
  year={2025}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Salesforce** for the BLIP and InstructBLIP model families
- **TrashNet**, **TrashBox**, and **BDwaste** dataset creators
- India's Ministry of Environment, Forest and Climate Change for the SWM Rules, 2016
