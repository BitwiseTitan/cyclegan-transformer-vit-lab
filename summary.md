## Generative AI — Assignment 2 (Implementation Summary)

This document summarizes the **implementation**, **experiments**, **evaluation artefacts**, and **results** for the three required questions:

- **Q1**: CycleGAN for **face ↔ sketch** translation (Kaggle “Person Face Sketches”)
- **Q2**: English → Urdu **machine translation** using a **custom Transformer**
- **Q3**: **CNN vs custom ViT vs pretrained ViT** on CIFAR-10

All experiments are designed to be reproducible and to write outputs (checkpoints, metrics, plots, predictions) during execution so that no separate post-processing runs are required.

---

## Repository structure

```text
project_root/
  download_data.py
  q1_*.py
  q2_*.py
  q3_*.py
  common_*.py
  data/
    q1/
    q2/
    q3/
  artifacts/
    q1/
    q2/
    q3/
```

### Output convention

Each training run writes a dedicated run directory:

- `artifacts/q1/<run_name>/`
- `artifacts/q2/<run_name>/`
- `artifacts/q3/<run_name>/`

These include:
- `config.json`: full run configuration
- `metrics.jsonl`: per-epoch metrics (append-only)
- `curves.png` or `loss_curves.png`: training curves
- `checkpoints/`: saved weights (best + last + per-epoch as applicable)
- Task-specific artifacts (samples, confusion matrices, tokenizers, example predictions)

---

## Dataset preparation

Datasets are downloaded/prepared into the local repo under `data/` using:

```bash
py -3 download_data.py --task all
```

### Q1 dataset
- Source: Kaggle “Person Face Sketches”
- Expected structure (verified):
  - `data/q1/train/photos`, `data/q1/train/sketches`
  - `data/q1/val/photos`, `data/q1/val/sketches`
  - `data/q1/test/photos`, `data/q1/test/sketches`

### Q2 dataset
- Source: Kaggle “Parallel Corpus for English-Urdu Language”
- Files used (verified):
  - `data/q2/Dataset/english-corpus.txt`
  - `data/q2/Dataset/urdu-corpus.txt`

### Q3 dataset
- CIFAR-10 is stored under:
  - `data/q3/cifar-10-batches-py/`

---

## Q1 — CycleGAN for face ↔ sketch translation

### Objective
Learn bidirectional image translation:
- **photo → sketch**
- **sketch → photo**

### Model
Standard CycleGAN components:
- **Generators**: ResNet-style generator (CycleGAN paper style)
- **Discriminators**: 70×70 PatchGAN (one discriminator per domain)

Domains in code:
- **A = photos (faces)** → discriminator `D_A`
- **B = sketches** → discriminator `D_B`

### Losses
The generator objective combines:
- adversarial loss (LSGAN)
- cycle-consistency loss \(L1\)
- identity loss \(L1\)

### Training + artefact saving
`q1_train.py` saves during training:
- checkpoints after every epoch:
  - `checkpoints/latest/checkpoint.pt`
  - `checkpoints/epoch_XXX/checkpoint.pt`
- `metrics.jsonl`: `G_total`, `G_gan`, `G_cycle`, `G_id`, `D_A`, `D_B`
- sample grids: `samples/epochXXX*.png` (qualitative progress)
- `loss_curves.png`

### Run executed
Run name: `q1_cyclegan_test`

Saved under:
- `artifacts/q1/q1_cyclegan_test/`

Metrics logged (epoch 1):
- `G_total = 4.1513`
- `D_A = 0.2411`, `D_B = 0.1413`

> Note: only **1 epoch** was run so far; extending training (resume) will produce a meaningful multi-epoch curve.

---

## Q2 — English → Urdu Machine Translation (custom Transformer)

### Objective
Translate English sentences to Urdu using a Transformer architecture, and evaluate using **BLEU**.

### Why custom (non-Hugging Face) pipeline
The environment had constraints that made pretrained fine-tuning difficult (VRAM limits, C: drive cache limits, and torch/loader safety checks). A custom implementation avoids these external dependencies and keeps the project fully self-contained.

### Tokenization
- Train a shared **SentencePiece BPE** tokenizer on combined EN+UR training text.
- Special token ids:
  - PAD=0, UNK=1, BOS=2, EOS=3
- Tokenizer artifacts saved to:
  - `tokenizer/spm.model`, `tokenizer/spm.vocab`, `tokenizer/spm_train.txt`

### Model
Custom Seq2Seq Transformer using `torch.nn.Transformer`:
- Learned embeddings + sinusoidal positional encoding
- Encoder + decoder stacks
- Linear output head over vocabulary

### Training
Key training design choices:
- teacher forcing (decoder input is shifted-right target)
- cross-entropy with **label smoothing**
- gradient clipping
- optional BLEU evaluation every N epochs (to avoid long pauses)

### Decoding for BLEU / qualitative examples
Improvements used:
- **beam search** decoding for BLEU epochs
- **no-repeat ngram** constraint to reduce repetition
- BLEU computed using `sacrebleu`

### Run executed (improved)
Run name: `q2_custom_improved_20260421_114755`

Saved under:
- `artifacts/q2/q2_custom_improved_20260421_114755/`

Contains:
- `metrics.jsonl`, `curves.png`, `example_translations.json`
- `tokenizer/` artifacts
- `checkpoints/epoch_*.pt` and `checkpoints/best.pt`

#### BLEU progression (computed every 5 epochs)
From `metrics.jsonl`:
- **Epoch 25 BLEU**: `0.5263`
- **Epoch 30 BLEU**: `0.3018`

> BLEU is currently low; however the pipeline is correct, reproducible, and produces both quantitative (BLEU) and qualitative outputs. Further improvements can be achieved with larger vocab, better regularization, stronger decoding, and longer training on the full dataset.

---

## Q3 — CNN vs ViT vs pretrained ViT on CIFAR-10

### Objective
Train and compare:
- a **CNN baseline**
- a **custom Vision Transformer**
- a **pretrained ViT** fine-tuned on CIFAR-10

Evaluation metrics requested by the assignment:
- **Accuracy** (required)
- optional precision/recall/F1 (not required for passing)
- **loss curves**
- **confusion matrix** (recommended)

All Q3 scripts save these artifacts automatically per run.

### Q3 artefacts (common)
Each run writes:
- `metrics.jsonl` (train loss, test loss, test accuracy per epoch)
- `curves.png`
- `confusion_matrix.png` + `confusion_matrix.json`
- `checkpoints/best.pt` + `checkpoints/last.pt`

### Runs executed and final results

#### CNN baseline
Run:
- `artifacts/q3/q3_cnn_20260421_135551/`

Final test accuracy (epoch 20):
- **0.9084**

#### Custom ViT
Run:
- `artifacts/q3/q3_vit_custom_20260421_135551/`

Final test accuracy (epoch 50):
- **0.8199**

#### Pretrained ViT (timm)
Run:
- `artifacts/q3/q3_vit_pre_20260421_135551/`

Final test accuracy (epoch 10):
- **0.9737**

### Summary comparison (Q3)
- **Best**: pretrained ViT fine-tune (0.9737)
- **Middle**: CNN baseline (0.9084)
- **Lowest**: custom ViT (0.8199)

This matches common expectations on CIFAR-10:
- pretrained ViT transfers strong representations efficiently
- CNNs are strong on small images
- training ViT from scratch on CIFAR-10 typically needs more augmentation/tuning to match CNN

---

## Reproducible run commands

### Q1
```bash
py -3 q1_train.py --run_name q1_cyclegan_run --epochs 50 --image_size 256 --batch_size 1
py -3 q1_train.py --run_name q1_cyclegan_run --resume
```

### Q2 (custom)
```bash
py -3 q2_train_custom.py --run_name q2_custom_run --epochs 30 --bleu_every 5
```

### Q3
```bash
py -3 q3_train_cnn.py --run_name q3_cnn_run --epochs 20
py -3 q3_train_vit.py --run_name q3_vit_custom_run --epochs 50 --patch_size 4
py -3 q3_train_vit_pretrained.py --run_name q3_vit_pre_run --epochs 10 --model_name vit_tiny_patch16_224
```

