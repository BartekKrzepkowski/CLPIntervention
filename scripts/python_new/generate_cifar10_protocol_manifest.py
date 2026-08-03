#!/usr/bin/env python3
"""Generate the versioned CIFAR-10 split and normalization manifest."""

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from scripts.python_new.get_mean_and_std import (
    VisualFieldsDataset,
    streaming_statistics,
)
from src.data.cifar10_protocol import (
    CIFAR10_PROTOCOL_PROFILE,
    array_sha256,
    stratified_cifar10_split,
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", default=os.environ.get("CIFAR10_PATH")
    )
    parser.add_argument(
        "--overlap", type=float, action="append", default=None
    )
    parser.add_argument("--resize-factor", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError("set --dataset-path or CIFAR10_PATH")
    overlaps = args.overlap or [0.0]

    raw_train = datasets.CIFAR10(
        args.dataset_path, train=True, download=False
    )
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    split = stratified_cifar10_split(targets)
    statistics = {}
    for overlap in overlaps:
        fields = VisualFieldsDataset(
            raw_train,
            overlap=overlap,
            resize_factor=args.resize_factor,
        )
        loader = DataLoader(
            Subset(fields, split.train_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        statistics[str(overlap)] = streaming_statistics(loader)

    batch_root = Path(args.dataset_path) / "cifar-10-batches-py"
    dataset_files = {
        name: _file_sha256(batch_root / name)
        for name in (
            *(f"data_batch_{index}" for index in range(1, 6)),
            "test_batch",
            "batches.meta",
        )
    }
    class_counts = {}
    for name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("fim", split.fim_indices),
    ):
        counts = np.bincount(targets[indices], minlength=10)
        class_counts[name] = counts.tolist()

    manifest = {
        "profile": CIFAR10_PROTOCOL_PROFILE,
        "dataset": "CIFAR-10 original training split",
        "created_at": date.today().isoformat(),
        "split": {
            "algorithm": (
                "np.random.default_rng(seed).permutation(class_indices); "
                "validation first, FIM second, remainder train"
            ),
            "seed": split.spec.seed,
            "validation_per_class": split.spec.validation_per_class,
            "fim_per_class": split.spec.fim_per_class,
        },
        "counts": split.state_dict()["counts"],
        "class_counts": class_counts,
        "targets_sha256": array_sha256(targets),
        "indices_sha256": split.state_dict()["indices_sha256"],
        "dataset_file_sha256": dataset_files,
        "normalization": {
            "source": "final protocol train split only",
            "sample_count": int(split.train_indices.size),
            "population_std": True,
            "resize_factor": args.resize_factor,
            "statistics": statistics,
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
