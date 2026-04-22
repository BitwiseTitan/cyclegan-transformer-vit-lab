"""
Q2 (custom) English -> Urdu translation using a small Transformer (PyTorch).

This avoids Hugging Face downloads/caches and is designed for low VRAM GPUs.

What it saves automatically under artifacts/q2/<run_name>/:
- checkpoints/epoch_*.pt and checkpoints/best.pt
- metrics.jsonl (train loss, val loss, BLEU)
- curves.png (loss + BLEU curves)
- tokenizer/ (SentencePiece model + vocab)
- example_translations.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common_plotting import save_curves
from common_seed import seed_everything
from common_utils import ensure_dir, project_root, save_json


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_parallel(en_path: Path, ur_path: Path, max_lines: int | None) -> Tuple[List[str], List[str]]:
    en_lines = en_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ur_lines = ur_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = min(len(en_lines), len(ur_lines))
    if max_lines is not None:
        n = min(n, int(max_lines))
    en = [s.strip() for s in en_lines[:n] if s.strip()]
    ur = [s.strip() for s in ur_lines[:n] if s.strip()]
    n2 = min(len(en), len(ur))
    return en[:n2], ur[:n2]


@dataclass(frozen=True)
class SpmIds:
    pad: int = 0
    unk: int = 1
    bos: int = 2
    eos: int = 3


def train_sentencepiece(
    out_dir: Path,
    en_texts: List[str],
    ur_texts: List[str],
    vocab_size: int,
) -> Path:
    """
    Train a single shared SentencePiece BPE model on combined EN+UR text.
    """
    import sentencepiece as spm  # type: ignore

    out_dir = ensure_dir(out_dir)
    train_txt = out_dir / "spm_train.txt"
    with train_txt.open("w", encoding="utf-8") as f:
        for s in en_texts:
            f.write(s.replace("\t", " ").strip() + "\n")
        for s in ur_texts:
            f.write(s.replace("\t", " ").strip() + "\n")

    model_prefix = str(out_dir / "spm")
    spm.SentencePieceTrainer.train(
        input=str(train_txt),
        model_prefix=model_prefix,
        vocab_size=int(vocab_size),
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[],
    )
    return Path(model_prefix + ".model")


class ParallelSpmDataset(Dataset):
    def __init__(
        self,
        en_texts: List[str],
        ur_texts: List[str],
        sp_model_path: Path,
        max_len: int,
    ) -> None:
        import sentencepiece as spm  # type: ignore

        self.en = en_texts
        self.ur = ur_texts
        self.max_len = int(max_len)
        self.ids = SpmIds()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))

    def _encode(self, text: str) -> List[int]:
        # reserve room for BOS/EOS
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: max(0, self.max_len - 2)]
        return [self.ids.bos] + ids + [self.ids.eos]

    def __len__(self) -> int:
        return min(len(self.en), len(self.ur))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self._encode(self.en[idx]), dtype=torch.long)
        tgt = torch.tensor(self._encode(self.ur[idx]), dtype=torch.long)
        return src, tgt


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    srcs, tgts = zip(*batch)
    src = pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)

        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=512)
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src.eq(self.pad_id)

    def _make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        return tgt.eq(self.pad_id)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        # tgt_inp is shifted right (B, T)
        src_key_padding_mask = self._make_src_key_padding_mask(src)
        tgt_key_padding_mask = self._make_tgt_key_padding_mask(tgt_inp)
        T = tgt_inp.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_inp.device)

        src_e = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_e = self.pos(self.tgt_emb(tgt_inp) * math.sqrt(self.d_model))
        out = self.tf(
            src_e,
            tgt_e,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.lm_head(out)  # (B, T, V)


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> torch.Tensor:
    model.eval()
    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        logits = model(src, ys)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        if (next_id.squeeze(1) == eos_id).all():
            break
    return ys


@torch.no_grad()
def beam_search_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """
    Simple beam search decoding (batch size supported; independent beams per sample).
    Returns token ids (B, <=max_len).
    """
    model.eval()
    device = src.device
    B = src.size(0)
    num_beams = int(max(1, num_beams))

    # For simplicity and robustness, do per-sample beam search.
    outs: List[torch.Tensor] = []
    for b in range(B):
        src_b = src[b : b + 1]
        beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0, False)]  # (seq, score, done)

        def violates_no_repeat(seq: torch.Tensor, next_id: int) -> bool:
            n = int(no_repeat_ngram_size)
            if n <= 0:
                return False
            ids = seq.squeeze(0).tolist()
            if len(ids) + 1 < n:
                return False
            # build previous ngrams
            prev = set()
            for i in range(len(ids) - n + 1):
                prev.add(tuple(ids[i : i + n]))
            cand = ids + [int(next_id)]
            last = tuple(cand[-n:])
            return last in prev

        for _ in range(max_len - 1):
            new_beams = []
            for seq, score, done in beams:
                if done:
                    new_beams.append((seq, score, done))
                    continue
                logits = model(src_b, seq)[:, -1, :]  # (1, V)
                logp = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
                topk = torch.topk(logp, k=min(num_beams, logp.numel()), dim=-1)
                for next_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                    if violates_no_repeat(seq, next_id):
                        continue
                    seq2 = torch.cat([seq, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
                    done2 = bool(next_id == eos_id)
                    new_beams.append((seq2, float(score + lp), done2))

            # prune
            # length penalty: divide score by len^alpha (alpha = length_penalty)
            def normed(item: Tuple[torch.Tensor, float, bool]) -> float:
                seq, score, _ = item
                L = seq.size(1)
                if length_penalty == 1.0:
                    return score
                return score / (float(L) ** float(length_penalty))

            new_beams.sort(key=normed, reverse=True)
            beams = new_beams[:num_beams]
            if all(done for _, _, done in beams):
                break

        best = max(beams, key=lambda x: x[1])[0]
        outs.append(best.squeeze(0).detach().cpu())

    # pad to same length
    maxL = max(int(t.size(0)) for t in outs)
    out = torch.full((B, maxL), eos_id, dtype=torch.long)
    for i, t in enumerate(outs):
        out[i, : t.size(0)] = t
    return out.to(device)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(project_root() / "data" / "q2"))
    p.add_argument("--artifacts_dir", type=str, default=str(project_root() / "artifacts" / "q2"))
    p.add_argument("--run_name", type=str, default=f"q2_custom_{_timestamp()}")

    p.add_argument("--max_lines", type=int, default=0, help="If >0, limit sentence pairs.")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--max_len", type=int, default=96)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=3)
    p.add_argument("--dec_layers", type=int, default=3)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--bleu_num_beams", type=int, default=4)
    p.add_argument("--bleu_length_penalty", type=float, default=1.0)
    p.add_argument("--bleu_no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--bleu_every", type=int, default=5, help="Compute BLEU every N epochs (0 disables BLEU).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    en_path = data_dir / "Dataset" / "english-corpus.txt"
    ur_path = data_dir / "Dataset" / "urdu-corpus.txt"
    if not en_path.exists() or not ur_path.exists():
        raise FileNotFoundError(f"Expected:\n  {en_path}\n  {ur_path}\nRun download_data.py --task q2 first.")

    max_lines = None if int(args.max_lines) <= 0 else int(args.max_lines)
    en, ur = _read_parallel(en_path, ur_path, max_lines=max_lines)
    n = len(en)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_train = int(0.9 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    en_train = [en[i] for i in train_idx]
    ur_train = [ur[i] for i in train_idx]
    en_val = [en[i] for i in val_idx]
    ur_val = [ur[i] for i in val_idx]

    run_dir = ensure_dir(Path(args.artifacts_dir) / args.run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    tok_dir = ensure_dir(run_dir / "tokenizer")
    save_json(run_dir / "config.json", vars(args))

    print(f"[q2-custom] run_dir={run_dir}")
    print(f"[q2-custom] pairs: total={n} train={len(en_train)} val={len(en_val)}")

    sp_model = train_sentencepiece(tok_dir, en_train, ur_train, vocab_size=args.vocab_size)
    print(f"[q2-custom] trained SentencePiece: {sp_model}")

    ds_train = ParallelSpmDataset(en_train, ur_train, sp_model_path=sp_model, max_len=args.max_len)
    ds_val = ParallelSpmDataset(en_val, ur_val, sp_model_path=sp_model, max_len=args.max_len)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    vocab_size = int(Path(str(sp_model).replace(".model", ".vocab")).read_text(encoding="utf-8", errors="ignore").count("\n") + 1)
    ids = SpmIds()

    device = torch.device(args.device)
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        pad_id=ids.pad,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Label smoothing helps avoid degenerate/repetitive outputs early on.
    try:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ids.pad, label_smoothing=float(args.label_smoothing))
    except TypeError:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ids.pad)

    import sacrebleu
    import sentencepiece as spm  # type: ignore

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model))

    def decode(ids_list: List[int]) -> str:
        # strip special tokens
        cleaned = [i for i in ids_list if i not in {ids.pad, ids.bos, ids.eos}]
        return sp.decode(cleaned).strip()

    curves: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "bleu": []}
    metrics_path = run_dir / "metrics.jsonl"
    best_bleu = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_steps = 0
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"q2-custom epoch {epoch}", leave=True)
        for step, (src, tgt) in enumerate(pbar, start=1):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss = loss / max(1, int(args.grad_accum_steps))
            loss.backward()

            if step % int(args.grad_accum_steps) == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                opt.step()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * max(1, int(args.grad_accum_steps))
            n_steps += 1
            pbar.set_postfix(loss=float(loss.item()) * max(1, int(args.grad_accum_steps)))

        train_loss = total_loss / max(1, n_steps)

        # Validation loss (+ optional BLEU)
        model.eval()
        val_losses = []
        preds_text: List[str] = []
        refs_text: List[str] = []
        do_bleu = int(args.bleu_every) > 0 and (epoch % int(args.bleu_every) == 0)

        # Bar 1: validation loss (fast)
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"q2 val epoch {epoch}", leave=False)
            for src, tgt in val_pbar:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_inp = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_inp)
                vloss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                val_losses.append(float(vloss.item()))
                if len(val_losses) % 10 == 0:
                    val_pbar.set_postfix(vloss=float(np.mean(val_losses[-10:])))

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        # Bar 2: BLEU decoding (slow, only every N epochs)
        if do_bleu:
            bleu_pbar = tqdm(val_loader, desc=f"q2 bleu epoch {epoch}", leave=False)
            with torch.no_grad():
                for src, tgt in bleu_pbar:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    gen = beam_search_decode(
                        model,
                        src,
                        bos_id=ids.bos,
                        eos_id=ids.eos,
                        max_len=args.max_len,
                        num_beams=int(args.bleu_num_beams),
                        length_penalty=float(args.bleu_length_penalty),
                        no_repeat_ngram_size=int(args.bleu_no_repeat_ngram_size),
                    )
                    for i in range(gen.size(0)):
                        preds_text.append(decode(gen[i].tolist()))
                        refs_text.append(decode(tgt[i].tolist()))
                    bleu_pbar.set_postfix(done=len(preds_text))

            bleu = float(sacrebleu.corpus_bleu(preds_text, [refs_text]).score) if preds_text else float("nan")
            tqdm.write(f"[q2-custom] epoch {epoch}: BLEU={0.0 if math.isnan(bleu) else bleu:.2f}")
        else:
            bleu = float("nan")
            tqdm.write(f"[q2-custom] epoch {epoch}: BLEU skipped (bleu_every={int(args.bleu_every)})")

        curves["train_loss"].append(float(train_loss))
        curves["val_loss"].append(float(val_loss))
        curves["bleu"].append(float(bleu))

        rec = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 3),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "bleu": None if math.isnan(bleu) else float(bleu),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # checkpoints each epoch + best BLEU
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "bleu": bleu}, ckpt_dir / f"epoch_{epoch:03d}.pt")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "bleu": bleu}, ckpt_dir / "best.pt")

        print(f"[q2-custom][{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} bleu={bleu:.2f}")

        # Save a few examples every epoch.
        # If BLEU was skipped, do a quick greedy decode on a small batch for qualitative monitoring.
        if not preds_text:
            try:
                src0, tgt0 = next(iter(val_loader))
                src0 = src0.to(device)
                gen0 = greedy_decode(model, src0, bos_id=ids.bos, eos_id=ids.eos, max_len=args.max_len)
                preds_text = [decode(gen0[i].tolist()) for i in range(gen0.size(0))]
                refs_text = [decode(tgt0[i].tolist()) for i in range(tgt0.size(0))]
            except Exception:
                preds_text = []
                refs_text = []
        example_n = min(20, len(preds_text), len(en_val))
        examples = [{"en": en_val[i], "ur_ref": ur_val[i], "ur_pred": preds_text[i]} for i in range(example_n)]
        save_json(run_dir / "example_translations.json", examples)

        save_curves(
            run_dir / "curves.png",
            {
                "train_loss": curves["train_loss"],
                "val_loss": curves["val_loss"],
                "bleu": curves["bleu"],
            },
            title="Q2 Custom Transformer: Loss & BLEU",
        )

    print(f"[q2-custom] done. best_bleu={best_bleu:.2f}")
    print(f"[q2-custom] artifacts at: {run_dir}")


if __name__ == "__main__":
    main()

