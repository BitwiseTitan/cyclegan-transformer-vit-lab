"""
Dataset preparation script (MUST be run before training).

Design goals (per assignment + repo rules):
- Run from project root (idempotent, clear logs, self-contained data folder)
- Prepare datasets under:
    data/q1/  (Person Face Sketches, Kaggle)
    data/q2/  (English–Urdu parallel corpus, Kaggle or manual)
    data/q3/  (CIFAR-10, torchvision download into this directory)
- If Kaggle auto-download isn't possible, print manual steps + exact target folders.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class KaggleDatasetSpec:
    slug: str
    url: str
    task_name: str
    target_dirname: str


Q1_SPEC = KaggleDatasetSpec(
    slug="almightyj/person-face-sketches",
    url="https://www.kaggle.com/datasets/almightyj/person-face-sketches",
    task_name="q1",
    target_dirname="q1",
)

Q2_SPEC = KaggleDatasetSpec(
    slug="zainuddin123/parallel-corpus-for-english-urdu-language",
    url="https://www.kaggle.com/datasets/zainuddin123/parallel-corpus-for-english-urdu-language",
    task_name="q2",
    target_dirname="q2",
)


def _project_root() -> Path:
    # This script lives in the project root.
    return Path(__file__).resolve().parent


def _data_root() -> Path:
    return _project_root() / "data"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _non_trivial_files_exist(p: Path) -> bool:
    """
    Return True if there are any files besides placeholders (like .gitkeep).
    """
    if not p.exists():
        return False
    for child in p.rglob("*"):
        if child.is_file() and child.name not in {".gitkeep", ".DS_Store", "Thumbs.db"}:
            return True
    return False


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _try_get_kaggle_api() -> Optional[object]:
    """
    Return an authenticated KaggleApi instance, or None if not available.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception:
        return None

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:
        return None
    return api


def _kaggle_download_and_unzip(api: object, spec: KaggleDatasetSpec, out_dir: Path) -> None:
    """
    Download and unzip a Kaggle dataset into out_dir.
    """
    _ensure_dir(out_dir)
    # KaggleApi has dataset_download_files(dataset, path=..., unzip=...)
    api.dataset_download_files(spec.slug, path=str(out_dir), unzip=True, quiet=False)  # type: ignore[attr-defined]


def _print_kaggle_manual_instructions(spec: KaggleDatasetSpec, out_dir: Path) -> None:
    print(f"[{spec.task_name}] Kaggle auto-download is NOT available.")
    print("Do this manually:")
    print(f"1) Open: {spec.url}")
    print("2) Click 'Download' to get a .zip file.")
    print(f"3) Extract the zip so the dataset files end up under:\n   {out_dir}")
    print("4) Re-run this script; it will detect the files and proceed.")
    print("\nExpected placement (flexible):")
    print(f"- It's OK if Kaggle extracts into a nested folder inside `{out_dir}`.")
    print("- Just ensure the images/text files are somewhere under that directory.")


def prepare_q1(data_root: Path, force: bool = False) -> Path:
    _print_header("Q1: Person Face Sketches (Kaggle) -> data/q1/")
    out_dir = _ensure_dir(data_root / Q1_SPEC.target_dirname)

    if _non_trivial_files_exist(out_dir) and not force:
        print(f"[q1] Found existing files under: {out_dir}")
        print("[q1] Skipping download (use --force to re-download).")
        return out_dir

    api = _try_get_kaggle_api()
    if api is None:
        _print_kaggle_manual_instructions(Q1_SPEC, out_dir)
        return out_dir

    print(f"[q1] Downloading from Kaggle dataset: {Q1_SPEC.slug}")
    _kaggle_download_and_unzip(api, Q1_SPEC, out_dir)
    print(f"[q1] Done. Files are under: {out_dir}")
    return out_dir


def prepare_q2(data_root: Path, force: bool = False) -> Path:
    _print_header("Q2: English-Urdu Parallel Corpus (Kaggle) -> data/q2/")
    out_dir = _ensure_dir(data_root / Q2_SPEC.target_dirname)

    if _non_trivial_files_exist(out_dir) and not force:
        print(f"[q2] Found existing files under: {out_dir}")
        print("[q2] Skipping download (use --force to re-download).")
        return out_dir

    api = _try_get_kaggle_api()
    if api is None:
        _print_kaggle_manual_instructions(Q2_SPEC, out_dir)
        print("\nAlternative (non-Kaggle) option from the assignment:")
        print("- UMC005 English-Urdu corpus: https://ufal.mff.cuni.cz/umc/005-en-ur/")
        print(f"- If you use UMC005, place the prepared parallel file(s) under:\n  {out_dir}")
        return out_dir

    print(f"[q2] Downloading from Kaggle dataset: {Q2_SPEC.slug}")
    _kaggle_download_and_unzip(api, Q2_SPEC, out_dir)
    print(f"[q2] Done. Files are under: {out_dir}")
    return out_dir


def prepare_q3(data_root: Path, force: bool = False, download: bool = True) -> Path:
    _print_header("Q3: CIFAR-10 (torchvision) -> data/q3/")
    out_dir = _ensure_dir(data_root / "q3")

    # torchvision creates CIFAR folder structure; we consider it present if any files exist.
    if _non_trivial_files_exist(out_dir) and not force:
        print(f"[q3] Found existing files under: {out_dir}")
        if not download:
            print("[q3] Download disabled; leaving as-is.")
        else:
            print("[q3] Skipping download (use --force to re-download).")
        return out_dir

    if not download:
        print(f"[q3] Initialized folder: {out_dir}")
        print("[q3] Download disabled by flag.")
        return out_dir

    try:
        from torchvision.datasets import CIFAR10
    except Exception as e:
        print("[q3] torchvision is required to download CIFAR-10.")
        print(f"[q3] Import error: {e}")
        print(f"[q3] You can still create the folder and manually place CIFAR-10 under:\n  {out_dir}")
        return out_dir

    print(f"[q3] Downloading CIFAR-10 into: {out_dir}")
    _ensure_dir(out_dir)
    # Download both train and test splits so the folder is fully initialized.
    CIFAR10(root=str(out_dir), train=True, download=True)
    CIFAR10(root=str(out_dir), train=False, download=True)
    print(f"[q3] Done. CIFAR-10 cached under: {out_dir}")
    return out_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/prepare datasets into ./data/<task>/")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["q1", "q2", "q3", "all"],
        help="Which task dataset(s) to prepare.",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    p.add_argument(
        "--skip_cifar_download",
        action="store_true",
        help="For Q3: create folder but skip torchvision download.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    root = _project_root()
    cwd = Path.cwd().resolve()
    if cwd != root:
        # Requirement says: run from project root.
        # For robustness, we auto-chdir but still tell the user.
        print(f"[info] Current working directory is:\n  {cwd}")
        print(f"[info] Switching to project root:\n  {root}")
        os.chdir(root)

    data_root = _ensure_dir(_data_root())
    print(f"[info] Using data root: {data_root}")

    task = args.task
    force = bool(args.force)

    if task in {"q1", "all"}:
        prepare_q1(data_root, force=force)
    if task in {"q2", "all"}:
        prepare_q2(data_root, force=force)
    if task in {"q3", "all"}:
        prepare_q3(data_root, force=force, download=not args.skip_cifar_download)

    _print_header("DONE")
    print("Dataset directories:")
    print(f"- Q1: {data_root / 'q1'}")
    print(f"- Q2: {data_root / 'q2'}")
    print(f"- Q3: {data_root / 'q3'}")
    print("\nNext step (recommended):")
    print("  py -3 download_data.py --task all")
    print("then verify data exists under data/q1, data/q2, data/q3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] Interrupted.")
        sys.exit(130)

