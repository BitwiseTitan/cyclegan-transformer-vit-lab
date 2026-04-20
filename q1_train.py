"""
Q1 training entrypoint (CycleGAN).

Implements full training loop with:
- checkpoint saving after every epoch (required by assignment)
- resume-from-last support
- metrics logging (JSONL) + loss curve plot
- sample translations grid saved during training
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from common_plotting import save_curves
from common_seed import seed_everything
from common_utils import ensure_dir, project_root, save_json, select_torch_device
from q1_config import Q1Config
from q1_dataset import Q1FaceSketchDataset
from q1_losses import GANLoss
from q1_models import (
    PatchDiscriminator,
    PatchDiscriminatorConfig,
    ResnetGenerator,
    ResnetGeneratorConfig,
    init_weights,
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _latest_checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "latest"


def _save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    nets: Dict[str, nn.Module],
    opts: Dict[str, torch.optim.Optimizer],
) -> None:
    ensure_dir(ckpt_dir)
    payload = {
        "epoch": epoch,
        "nets": {k: v.state_dict() for k, v in nets.items()},
        "opts": {k: v.state_dict() for k, v in opts.items()},
    }
    torch.save(payload, ckpt_dir / "checkpoint.pt")


def _load_checkpoint(
    ckpt_path: Path,
    nets: Dict[str, nn.Module],
    opts: Dict[str, torch.optim.Optimizer],
    device: torch.device,
) -> int:
    payload = torch.load(ckpt_path, map_location=device)
    for k, sd in payload.get("nets", {}).items():
        if k in nets:
            nets[k].load_state_dict(sd)
    for k, sd in payload.get("opts", {}).items():
        if k in opts:
            opts[k].load_state_dict(sd)
    return int(payload.get("epoch", 0))


def _set_requires_grad(net: nn.Module, flag: bool) -> None:
    for p in net.parameters():
        p.requires_grad = flag


def _sample_and_save(
    out_path: Path,
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    G_A2B: nn.Module,
    G_B2A: nn.Module,
) -> None:
    """
    Save a grid: [A, G(A), cycle(A), B, G(B), cycle(B)]
    """
    G_A2B.eval()
    G_B2A.eval()
    with torch.no_grad():
        fake_B = G_A2B(real_A)
        cycled_A = G_B2A(fake_B)
        fake_A = G_B2A(real_B)
        cycled_B = G_A2B(fake_A)

        grid = make_grid(
            torch.cat([real_A, fake_B, cycled_A, real_B, fake_A, cycled_B], dim=0),
            nrow=real_A.shape[0],
            normalize=True,
            value_range=(-1, 1),
        )
        ensure_dir(out_path.parent)
        save_image(grid, out_path)

    G_A2B.train()
    G_B2A.train()


def _parse_args() -> argparse.Namespace:
    cfg = Q1Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(project_root() / "data" / "q1"))
    p.add_argument("--artifacts_dir", type=str, default=str(project_root() / "artifacts" / "q1"))
    p.add_argument("--run_name", type=str, default=cfg.run_name)

    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--lambda_cycle", type=float, default=cfg.lambda_cycle)
    p.add_argument("--lambda_id", type=float, default=cfg.lambda_id)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=select_torch_device())
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every_n_steps", type=int, default=500)
    p.add_argument("--sample_batch_size", type=int, default=4)
    p.add_argument("--unaligned", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    run_dir = ensure_dir(artifacts_dir / args.run_name)
    ckpt_dir = _latest_checkpoint_dir(run_dir)
    samples_dir = ensure_dir(run_dir / "samples")

    save_json(run_dir / "config.json", vars(args))

    train_ds = Q1FaceSketchDataset(data_dir, split="train", image_size=args.image_size, unaligned=args.unaligned)
    val_ds = Q1FaceSketchDataset(data_dir, split="val", image_size=args.image_size, unaligned=args.unaligned)
    print(f"[q1] device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"[q1] run_dir={run_dir}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.sample_batch_size, args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_iter = iter(val_loader)

    n_blocks = 9 if args.image_size >= 256 else 6
    G_A2B = ResnetGenerator(ResnetGeneratorConfig(n_blocks=n_blocks)).to(device)
    G_B2A = ResnetGenerator(ResnetGeneratorConfig(n_blocks=n_blocks)).to(device)
    D_A = PatchDiscriminator(PatchDiscriminatorConfig()).to(device)
    D_B = PatchDiscriminator(PatchDiscriminatorConfig()).to(device)
    init_weights(G_A2B)
    init_weights(G_B2A)
    init_weights(D_A)
    init_weights(D_B)

    nets: Dict[str, nn.Module] = {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}

    gan_loss = GANLoss(mode="lsgan").to(device)
    l1 = nn.L1Loss()

    opt_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opts = {"opt_G": opt_G, "opt_D_A": opt_D_A, "opt_D_B": opt_D_B}

    start_epoch = 1
    if args.resume and (ckpt_dir / "checkpoint.pt").exists():
        last_epoch = _load_checkpoint(ckpt_dir / "checkpoint.pt", nets=nets, opts=opts, device=device)
        start_epoch = last_epoch + 1
        print(f"[q1] Resumed from epoch {last_epoch}. Next epoch: {start_epoch}")

    metrics_path = run_dir / "metrics.jsonl"
    loss_curves: Dict[str, list[float]] = {k: [] for k in ["G_total", "G_gan", "G_cycle", "G_id", "D_A", "D_B"]}

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n[q1] ===== epoch {epoch}/{args.epochs} =====")
        running = {k: 0.0 for k in loss_curves.keys()}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"q1 epoch {epoch}", leave=True)
        for real_A, real_B in pbar:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Generators
            _set_requires_grad(D_A, False)
            _set_requires_grad(D_B, False)
            opt_G.zero_grad(set_to_none=True)

            fake_B = G_A2B(real_A)
            loss_gan_A2B = gan_loss(D_B(fake_B), True)
            fake_A = G_B2A(real_B)
            loss_gan_B2A = gan_loss(D_A(fake_A), True)

            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            loss_cycle = l1(rec_A, real_A) + l1(rec_B, real_B)

            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_id = l1(idt_A, real_A) + l1(idt_B, real_B)

            loss_gan = loss_gan_A2B + loss_gan_B2A
            loss_G = loss_gan + args.lambda_cycle * loss_cycle + args.lambda_id * loss_id
            loss_G.backward()
            opt_G.step()

            # Discriminators
            _set_requires_grad(D_A, True)
            _set_requires_grad(D_B, True)

            opt_D_A.zero_grad(set_to_none=True)
            loss_D_A = 0.5 * (gan_loss(D_A(real_A), True) + gan_loss(D_A(fake_A.detach()), False))
            loss_D_A.backward()
            opt_D_A.step()

            opt_D_B.zero_grad(set_to_none=True)
            loss_D_B = 0.5 * (gan_loss(D_B(real_B), True) + gan_loss(D_B(fake_B.detach()), False))
            loss_D_B.backward()
            opt_D_B.step()

            running["G_total"] += float(loss_G.item())
            running["G_gan"] += float(loss_gan.item())
            running["G_cycle"] += float(loss_cycle.item())
            running["G_id"] += float(loss_id.item())
            running["D_A"] += float(loss_D_A.item())
            running["D_B"] += float(loss_D_B.item())
            n_batches += 1
            global_step += 1
            pbar.set_postfix(
                G=float(loss_G.item()),
                D_A=float(loss_D_A.item()),
                D_B=float(loss_D_B.item()),
                step=global_step,
            )

            if args.save_every_n_steps > 0 and (global_step % args.save_every_n_steps == 0):
                try:
                    real_A_s, real_B_s = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    real_A_s, real_B_s = next(val_iter)
                _sample_and_save(
                    samples_dir / f"epoch{epoch:03d}_step{global_step:07d}.png",
                    real_A_s.to(device),
                    real_B_s.to(device),
                    G_A2B,
                    G_B2A,
                )

        epoch_metrics = {k: (running[k] / max(1, n_batches)) for k in running.keys()}
        for k, v in epoch_metrics.items():
            loss_curves[k].append(float(v))

        _save_checkpoint(ckpt_dir, epoch=epoch, nets=nets, opts=opts)
        _save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}", epoch=epoch, nets=nets, opts=opts)
        print(f"[q1] saved checkpoints: {ckpt_dir / 'checkpoint.pt'}")

        try:
            real_A_s, real_B_s = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            real_A_s, real_B_s = next(val_iter)
        _sample_and_save(samples_dir / f"epoch{epoch:03d}.png", real_A_s.to(device), real_B_s.to(device), G_A2B, G_B2A)
        print(f"[q1] saved sample grid: {samples_dir / f'epoch{epoch:03d}.png'}")

        record = {"epoch": epoch, "global_step": global_step, "time_sec": round(time.time() - t0, 3), **epoch_metrics}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[q1][epoch {epoch:03d}/{args.epochs}] " + " ".join([f"{k}={record[k]:.4f}" for k in epoch_metrics.keys()]))

    save_curves(run_dir / "loss_curves.png", loss_curves, title="Q1 CycleGAN Loss Curves", ylabel="loss")
    print(f"[q1] saved loss curves: {run_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()

