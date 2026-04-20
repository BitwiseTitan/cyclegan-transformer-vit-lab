# Generative AI — Assignment 2 (Implementation)

This repository contains **three independent tasks** implemented in a single, clean, reproducible project:

- **Q1**: CycleGAN for **face ↔ sketch** translation (with a minimal Flask demo UI)
- **Q2**: **English → Urdu** machine translation using Transformers (train + evaluate + inference)
- **Q3**: **Vision Transformer vs CNN** on CIFAR-10 (custom ViT, pretrained ViT fine-tune, CNN baseline)

All project code currently lives at the **repository root** (flat `.py` files).

## Quickstart

### 1) Setup environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Download / prepare datasets (must be done first)

From the project root:

```bash
py -3 download_data.py --task all
```

This creates:

- `data/q1/` (Kaggle: Person Face Sketches)
- `data/q2/` (Kaggle: English–Urdu parallel corpus, or manual alternative)
- `data/q3/` (CIFAR-10; optionally downloads via torchvision into this folder)

If Kaggle download isn’t available, the script prints **exact manual steps** and the **target folders**.

## Project layout

```
project_root/
  README.md
  requirements.txt
  .gitignore
  download_data.py
  common_utils.py
  common_metrics.py
  common_plotting.py
  common_seed.py
  q1_config.py
  q1_dataset.py
  q1_models.py
  q1_losses.py
  q1_train.py
  q1_infer.py
  q1_app.py
  q2_config.py
  q2_dataset.py
  q2_tokenizer_utils.py
  q2_train.py
  q2_evaluate.py
  q2_infer.py
  q3_config.py
  q3_cnn.py
  q3_vit_custom.py
  q3_vit_pretrained.py
  q3_train_cnn.py
  q3_train_vit.py
  q3_train_vit_pretrained.py
  q3_evaluate.py
  data/
    q1/
    q2/
    q3/
  artifacts/
    q1/
    q2/
    q3/
```

## Running (high level)

### Q1 (CycleGAN)

- Train: `py -3 q1_train.py`
- Inference: `py -3 q1_infer.py --input <path> --direction auto`
- Demo UI: `py -3 q1_app.py` then open the shown local URL.

Outputs (checkpoints, sample grids, logs) go under `artifacts/q1/`.

### Q2 (English → Urdu Translation)

- Train: `py -3 q2_train.py`
- Evaluate: `py -3 q2_evaluate.py`
- Inference: `py -3 q2_infer.py --text "Hello world"`

Outputs go under `artifacts/q2/`.

### Q3 (CIFAR-10: CNN vs ViT)

- CNN train: `py -3 q3_train_cnn.py`
- Custom ViT train: `py -3 q3_train_vit.py`
- Pretrained ViT fine-tune: `py -3 q3_train_vit_pretrained.py`
- Evaluate: `py -3 q3_evaluate.py`

Outputs go under `artifacts/q3/`.

## Notes

- This repo is designed to be **Colab-friendly** (run from project root).
- The download script is **idempotent** and will skip work if outputs already exist.
- For Q1/Q2 Kaggle downloads, install and configure Kaggle API credentials:
  - Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

## (Removed)
The remainder of this README previously described an older layout; it has been removed to match the current flat-file structure.

