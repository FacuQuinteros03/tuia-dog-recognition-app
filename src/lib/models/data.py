"""Dataset de PyTorch sobre las carpetas <split>/<raza>/ generadas por 1-pre-processing.ipynb."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.preprocessing import bgr_to_rgb, build_train_augmentation, preprocess_rgb, read_bgr

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_classes(root: Path) -> list[str]:
    """Razas = subcarpetas de train, en orden alfabetico (el mismo orden que los logits)."""
    return sorted(p.name for p in (root / "train").iterdir() if p.is_dir())


def list_split(root: Path, split: str, classes: list[str]) -> tuple[list[str], list[int]]:
    index = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for breed_dir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
        for img in sorted(breed_dir.iterdir()):
            if img.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(img))
                labels.append(index[breed_dir.name])
    return paths, labels


class DogImageDataset(Dataset):
    """Lee BGR -> RGB, aplica augmentation (solo si `augment`) y el preprocesamiento compartido."""

    def __init__(self, paths: list[str], labels: list[int], size: int, augment: bool = False) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.size = size
        self.augmentation = build_train_augmentation(size) if augment else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rgb = bgr_to_rgb(read_bgr(self.paths[i]))
        if self.augmentation is not None:
            rgb = self.augmentation(image=rgb)["image"]
        x = torch.from_numpy(preprocess_rgb(rgb, self.size))
        return x, torch.tensor(self.labels[i], dtype=torch.long)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random

    random.seed(seed)
