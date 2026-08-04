#!/usr/bin/env python3
"""Compute per-field normalization statistics without materializing the dataset."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


DATASET_ENV = {
    "cifar10": "CIFAR10_PATH",
    "fmnist": "FMNIST_PATH",
    "kmnist": "KMNIST_PATH",
    "mnist": "MNIST_PATH",
    "svhn": "SVHN_PATH",
    "tinyimagenet": "TINYIMAGENET_PATH",
}


def load_training_dataset(name, root):
    if name == "cifar10":
        return datasets.CIFAR10(root, train=True, download=False)
    if name == "fmnist":
        return datasets.FashionMNIST(root, train=True, download=False)
    if name == "kmnist":
        return datasets.KMNIST(root, train=True, download=False)
    if name == "mnist":
        return datasets.MNIST(root, train=True, download=False)
    if name == "svhn":
        return datasets.SVHN(root, split="train", download=False)
    if name == "tinyimagenet":
        return datasets.ImageFolder(Path(root) / "train")
    raise ValueError(f"Unsupported dataset: {name}")


class VisualFieldsDataset(Dataset):
    def __init__(self, dataset, overlap=0.0, resize_factor=0.25):
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap must be in [0, 1]")
        if not 0.0 < resize_factor <= 1.0:
            raise ValueError("resize_factor must be in (0, 1]")
        self.dataset = dataset
        self.with_overlap = 0.5 + overlap / 2
        image, _ = dataset[0]
        width, height = image.size
        field_width = math.ceil(width * self.with_overlap)
        reduced_size = (
            math.ceil(height * resize_factor),
            math.ceil(field_width * resize_factor),
        )
        self.to_tensor = transforms.ToTensor()
        self.blur = transforms.Compose(
            [
                transforms.Resize(
                    reduced_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=None,
                ),
                transforms.Resize(
                    (height, field_width),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=None,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        width, height = image.size
        field_width = math.ceil(width * self.with_overlap)
        left = image.crop((0, 0, field_width, height))
        right = image.crop((width - field_width, 0, width, height))
        return self.to_tensor(left), self.to_tensor(right), self.blur(right)


def streaming_statistics(loader):
    names = ("proper_left", "proper_right", "blurred_right")
    sums = {name: None for name in names}
    squared_sums = {name: None for name in names}
    counts = {name: 0 for name in names}
    for batches in loader:
        for name, batch in zip(names, batches):
            batch = batch.to(dtype=torch.float64)
            batch_sum = batch.sum(dim=(0, 2, 3))
            batch_squared_sum = batch.square().sum(dim=(0, 2, 3))
            sums[name] = batch_sum if sums[name] is None else sums[name] + batch_sum
            squared_sums[name] = (
                batch_squared_sum
                if squared_sums[name] is None
                else squared_sums[name] + batch_squared_sum
            )
            counts[name] += batch.size(0) * batch.size(2) * batch.size(3)

    result = {}
    for name in names:
        mean = sums[name] / counts[name]
        variance = squared_sums[name] / counts[name] - mean.square()
        result[name] = {
            "mean": mean.tolist(),
            "std": variance.clamp_min(0).sqrt().tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASET_ENV))
    parser.add_argument("--dataset-path")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--resize-factor", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.dataset_path or os.environ.get(DATASET_ENV[args.dataset])
    if not root:
        raise ValueError(
            f"set --dataset-path or {DATASET_ENV[args.dataset]} for {args.dataset}"
        )
    dataset = VisualFieldsDataset(
        load_training_dataset(args.dataset, root),
        overlap=args.overlap,
        resize_factor=args.resize_factor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(json.dumps(streaming_statistics(loader), indent=2))


if __name__ == "__main__":
    main()
