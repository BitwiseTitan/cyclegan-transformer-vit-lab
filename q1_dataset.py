"""
Q1 dataset loader for Person Face Sketches (Kaggle).

Expected placement under `data/q1/` is printed by `download_data.py`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Q1FaceSketchDataset(Dataset):
    """
    Loads Person Face Sketches dataset with expected structure:

    data/q1/
      train|val|test/
        photos/
        sketches/

    Pairing strategy:
    - If filenames match between photos and sketches, pair by name.
    - Otherwise, unaligned pairing: photo[i] with a random sketch.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        unaligned: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.photos_dir = self.split_dir / "photos"
        self.sketches_dir = self.split_dir / "sketches"

        if not self.photos_dir.exists() or not self.sketches_dir.exists():
            raise FileNotFoundError(
                f"Missing expected folders. Got photos={self.photos_dir.exists()} sketches={self.sketches_dir.exists()}.\n"
                f"Expected:\n  {self.photos_dir}\n  {self.sketches_dir}"
            )

        self.photo_paths = sorted([p for p in self.photos_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.sketch_paths = sorted([p for p in self.sketches_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(self.photo_paths) == 0 or len(self.sketch_paths) == 0:
            raise RuntimeError(f"No images found in {self.photos_dir} or {self.sketches_dir}")

        sketch_by_name: Dict[str, Path] = {p.name: p for p in self.sketch_paths}
        aligned = []
        for p in self.photo_paths:
            s = sketch_by_name.get(p.name)
            if s is not None:
                aligned.append((p, s))

        self.unaligned = bool(unaligned) or (len(aligned) < max(10, int(0.8 * len(self.photo_paths))))
        if not self.unaligned:
            self.items: List[Tuple[Path, Path]] = aligned
        else:
            # store photos only; choose random sketch at __getitem__
            self.items = [(p, Path("")) for p in self.photo_paths]

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        photo_path, sketch_path = self.items[idx]
        if self.unaligned:
            sketch_path = random.choice(self.sketch_paths)

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")
        return self.tf(photo), self.tf(sketch)

