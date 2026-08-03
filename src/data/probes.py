from dataclasses import dataclass

import numpy as np
import torch

from src.data import (
    transforms_cifar10,
    transforms_fmnist,
    transforms_kmnist,
    transforms_mnist,
    transforms_svhn,
    transforms_tinyimagenet,
)
from src.data.datasets_class import SplitAndAugmentDataset
from src.utils.utils_data import get_targets


TRANSFORM_MODULES = {
    "mm_cifar10": transforms_cifar10,
    "mm_fmnist": transforms_fmnist,
    "mm_kmnist": transforms_kmnist,
    "mm_mnist": transforms_mnist,
    "mm_svhn": transforms_svhn,
    "mm_tinyimagenet": transforms_tinyimagenet,
}


@dataclass(frozen=True)
class FIMProbe:
    tensors: dict[str, torch.Tensor]
    probe_indices: np.ndarray
    train_indices: np.ndarray


def class_balanced_indices(targets, fraction=0.02, seed=0):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    classes = np.unique(targets)
    samples_per_class = int(targets.size * fraction) // classes.size
    if samples_per_class < 1:
        raise ValueError("fraction selects fewer than one sample per class")

    generator = np.random.default_rng(seed)
    selected = []
    for label in classes:
        candidates = np.flatnonzero(targets == label)
        if candidates.size < samples_per_class:
            raise ValueError(f"class {label} has too few samples")
        selected.append(
            generator.choice(candidates, size=samples_per_class, replace=False)
        )
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _raw_dataset(dataset):
    current = dataset
    while hasattr(current, "dataset") and not (
        hasattr(current, "targets") or hasattr(current, "labels")
    ):
        current = current.dataset
    return current


def build_fim_probe(
    train_dataset,
    dataset_name,
    overlap=0.0,
    resize_factor=0.25,
    fraction=0.02,
    seed=0,
    probe_indices=None,
    train_indices=None,
    normalization_profile=None,
):
    """Build a deterministic FIM probe and an explicitly disjoint train split."""
    try:
        transform_module = TRANSFORM_MODULES[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported FIM probe dataset: {dataset_name}") from error

    raw_dataset = _raw_dataset(train_dataset)
    targets = get_targets(raw_dataset)
    if probe_indices is None:
        probe_indices = class_balanced_indices(
            targets, fraction=fraction, seed=seed
        )
    else:
        probe_indices = np.sort(
            np.ascontiguousarray(probe_indices, dtype=np.int64).reshape(-1)
        )
        if (
            probe_indices.size == 0
            or np.unique(probe_indices).size != probe_indices.size
            or probe_indices.min() < 0
            or probe_indices.max() >= len(raw_dataset)
        ):
            raise ValueError("probe_indices must be non-empty, unique, and valid")
    if train_indices is None:
        keep_mask = np.ones(len(raw_dataset), dtype=bool)
        keep_mask[probe_indices] = False
        train_indices = np.flatnonzero(keep_mask)
    else:
        train_indices = np.sort(
            np.ascontiguousarray(train_indices, dtype=np.int64).reshape(-1)
        )
        if train_indices.size and (
            train_indices.min() < 0
            or train_indices.max() >= len(raw_dataset)
            or np.unique(train_indices).size != train_indices.size
        ):
            raise ValueError("train_indices must be unique and valid")
        if np.intersect1d(probe_indices, train_indices).size:
            raise ValueError("probe_indices and train_indices must be disjoint")

    transform_kwargs = (
        {"normalization_profile": normalization_profile}
        if dataset_name == "mm_cifar10" and normalization_profile is not None
        else {}
    )
    first_image, _ = raw_dataset[0]
    width, height = first_image.size
    transforms_map = transform_module.TRANSFORMS_NAME_MAP
    proper_dataset = SplitAndAugmentDataset(
        raw_dataset,
        transforms_map["transform_eval_proper"](
            overlap, "left", **transform_kwargs
        ),
        transforms_map["transform_eval_proper"](
            overlap, "right", **transform_kwargs
        ),
        overlap=overlap,
        is_train=False,
        reverse=False,
    )
    blurred_dataset = SplitAndAugmentDataset(
        raw_dataset,
        transforms_map["transform_eval_proper"](
            overlap, "left", **transform_kwargs
        ),
        transforms_map["transform_eval_blurred"](
            height, width, resize_factor, overlap, **transform_kwargs
        ),
        overlap=overlap,
        is_train=False,
        reverse=False,
    )

    proper_samples = [proper_dataset[int(index)] for index in probe_indices]
    blurred_samples = [blurred_dataset[int(index)] for index in probe_indices]
    proper_pairs, labels = zip(*proper_samples)
    blurred_pairs, blurred_labels = zip(*blurred_samples)
    if labels != blurred_labels:
        raise RuntimeError("proper and blurred FIM probes are not aligned")

    proper_left, proper_right = zip(*proper_pairs)
    _, blurred_right = zip(*blurred_pairs)
    tensors = {
        "proper_x_left": torch.stack(proper_left),
        "proper_x_right": torch.stack(proper_right),
        "blurred_x_right": torch.stack(blurred_right),
        "y": torch.as_tensor(labels, dtype=torch.long),
    }
    return FIMProbe(
        tensors=tensors,
        probe_indices=probe_indices,
        train_indices=train_indices,
    )
