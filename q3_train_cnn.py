"""
Q3 training entrypoint for CNN baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from common_plotting import save_curves
from common_seed import seed_everything
from common_utils import ensure_dir, project_root, save_json, select_torch_device
from q3_cnn import Q3CnnBaseline


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    acc = float((logits_np.argmax(axis=1) == labels_np).mean())
    return float(np.mean(losses)), acc, logits_np, labels_np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(project_root() / "data" / "q3"))
    p.add_argument("--artifacts_dir", type=str, default=str(project_root() / "artifacts" / "q3"))
    p.add_argument("--run_name", type=str, default=f"cnn_{_timestamp()}")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=select_torch_device())
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    run_dir = ensure_dir(Path(args.artifacts_dir) / args.run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    save_json(run_dir / "config.json", vars(args))
    print(f"[q3][cnn] device={device}")
    print(f"[q3][cnn] run_dir={run_dir}")

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    train_ds = CIFAR10(root=args.data_dir, train=True, download=False, transform=tf_train)
    test_ds = CIFAR10(root=args.data_dir, train=False, download=False, transform=tf_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = Q3CnnBaseline(num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    curves: Dict[str, list[float]] = {"train_loss": [], "test_loss": [], "test_acc": []}
    metrics_path = run_dir / "metrics.jsonl"

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"q3 cnn epoch {epoch}", leave=True)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

        sched.step()
        test_loss, test_acc, logits_np, labels_np = _eval(model, test_loader, device=device)
        train_loss = float(np.mean(train_losses))

        curves["train_loss"].append(train_loss)
        curves["test_loss"].append(test_loss)
        curves["test_acc"].append(test_acc)

        rec = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 3),
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "test_acc": test_acc}, ckpt_dir / "best.pt")
            print(f"[q3][cnn] new best checkpoint: {ckpt_dir / 'best.pt'} (acc={best_acc:.4f})")

        print(f"[q3][cnn][{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    torch.save({"epoch": args.epochs, "state_dict": model.state_dict(), "test_acc": curves["test_acc"][-1]}, ckpt_dir / "last.pt")
    save_curves(run_dir / "curves.png", {"train_loss": curves["train_loss"], "test_loss": curves["test_loss"], "test_acc": curves["test_acc"]}, title="Q3 CNN Curves")

    # Confusion matrix (saved automatically)
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        preds = logits_np.argmax(axis=1)
        cm = confusion_matrix(labels_np, preds, labels=list(range(10)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
        plt.title("CNN Confusion Matrix (CIFAR-10)")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        save_json(run_dir / "confusion_matrix.json", {"cm": cm.tolist()})
    except Exception as e:
        save_json(run_dir / "confusion_matrix_error.json", {"error": str(e)})


if __name__ == "__main__":
    main()

