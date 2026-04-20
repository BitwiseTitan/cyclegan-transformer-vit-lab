"""
Q2 training entrypoint (Transformer MT).

Trains an English -> Urdu translation model with:
- mBART-50 fine-tuning baseline (fast + strong)
- BLEU evaluation during training
- saved artifacts (config, trainer logs, BLEU, predictions) under artifacts/q2/<run_name>/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common_seed import seed_everything
from common_utils import ensure_dir, project_root, save_json
from q2_config import Q2Config


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_parallel_texts(en_path: Path, ur_path: Path, max_lines: int | None = None) -> Tuple[List[str], List[str]]:
    en_lines = en_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ur_lines = ur_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = min(len(en_lines), len(ur_lines))
    if max_lines is not None:
        n = min(n, int(max_lines))
    en = [s.strip() for s in en_lines[:n] if s.strip()]
    ur = [s.strip() for s in ur_lines[:n] if s.strip()]
    n2 = min(len(en), len(ur))
    return en[:n2], ur[:n2]


def _parse_args() -> argparse.Namespace:
    cfg = Q2Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(cfg.data_dir))
    p.add_argument("--artifacts_dir", type=str, default=str(cfg.artifacts_dir))
    p.add_argument("--run_name", type=str, default=f"mbart50_en_ur_{_timestamp()}")
    p.add_argument("--model_name", type=str, default=cfg.model_name)

    p.add_argument("--max_source_len", type=int, default=cfg.max_source_len)
    p.add_argument("--max_target_len", type=int, default=cfg.max_target_len)
    p.add_argument("--train_batch_size", type=int, default=cfg.train_batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--num_train_epochs", type=int, default=cfg.num_train_epochs)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--max_train_lines", type=int, default=0, help="If >0, limit number of sentence pairs for quick runs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    # BLEU
    import evaluate

    data_dir = Path(args.data_dir)
    en_path = data_dir / "Dataset" / "english-corpus.txt"
    ur_path = data_dir / "Dataset" / "urdu-corpus.txt"
    if not en_path.exists() or not ur_path.exists():
        raise FileNotFoundError(f"Expected:\n  {en_path}\n  {ur_path}\nRun download_data.py for q2 first.")

    max_lines = None if int(args.max_train_lines) <= 0 else int(args.max_train_lines)
    en, ur = _read_parallel_texts(en_path, ur_path, max_lines=max_lines)
    print(f"[q2] loaded sentence pairs: {len(en)}")

    # Simple split (deterministic)
    n = len(en)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_train = int(0.9 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    ds_train = Dataset.from_dict({"en": [en[i] for i in train_idx], "ur": [ur[i] for i in train_idx]})
    ds_val = Dataset.from_dict({"en": [en[i] for i in val_idx], "ur": [ur[i] for i in val_idx]})
    print(f"[q2] train_pairs={len(ds_train)} val_pairs={len(ds_val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # mBART language codes (AutoTokenizer for mbart supports these attributes)
    src_lang = "en_XX"
    tgt_lang = "ur_PK"
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang  # type: ignore[attr-defined]
    forced_bos_token_id = None
    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        model.config.forced_bos_token_id = forced_bos_token_id
    except Exception:
        forced_bos_token_id = None

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            batch["en"],
            max_length=args.max_source_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["ur"],
                max_length=args.max_target_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_train_tok = ds_train.map(preprocess, batched=True, remove_columns=["en", "ur"])
    ds_val_tok = ds_val.map(preprocess, batched=True, remove_columns=["en", "ur"])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_out = bleu.compute(predictions=[p.strip() for p in decoded_preds], references=[[l.strip()] for l in decoded_labels])
        return {"bleu": float(bleu_out["score"])}

    out_root = Path(args.artifacts_dir)
    run_dir = ensure_dir(out_root / args.run_name)
    ensure_dir(run_dir / "checkpoints")
    save_json(run_dir / "config.json", vars(args))
    print(f"[q2] run_dir={run_dir}")
    print(f"[q2] model_name={args.model_name} src_lang=en_XX tgt_lang=ur_PK")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        num_train_epochs=float(args.num_train_epochs),
        predict_with_generate=True,
        fp16=True,
        save_total_limit=2,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("[q2] training complete; running final evaluation + saving artifacts...")

    # Final evaluation + save a few qualitative examples
    metrics = trainer.evaluate(max_length=args.max_target_len, num_beams=4)
    save_json(run_dir / "final_metrics.json", metrics)
    print(f"[q2] saved final metrics: {run_dir / 'final_metrics.json'}")

    # Save example translations (no extra script needed)
    sample_n = min(20, len(ds_val))
    sample = ds_val.select(range(sample_n))
    inputs = tokenizer(sample["en"], return_tensors="pt", padding=True, truncation=True, max_length=args.max_source_len).to(trainer.model.device)
    gen = trainer.model.generate(**inputs, max_length=args.max_target_len, num_beams=4)
    pred_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

    examples = []
    for i in range(sample_n):
        examples.append({"en": sample["en"][i], "ur_ref": sample["ur"][i], "ur_pred": pred_text[i]})
    save_json(run_dir / "example_translations.json", examples)
    print(f"[q2] saved example translations: {run_dir / 'example_translations.json'}")


if __name__ == "__main__":
    main()

