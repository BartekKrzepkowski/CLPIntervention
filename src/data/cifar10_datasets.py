"""CIFAR-10 datasets for the validation-controlled CLP protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from src.data import transforms_cifar10
from src.data.cifar10_protocol import (
    CIFAR10_PROTOCOL_PROFILE,
    CIFAR10Split,
    load_cifar10_protocol_manifest,
    stratified_cifar10_split,
    validate_cifar10_dataset_files,
    validate_cifar10_protocol_split,
)
from src.data.datasets_class import SplitAndAugmentDataset
from src.data.probes import FIMProbe, build_fim_probe


TRAIN_PROBE_SEED = 1083
TRAIN_PROBE_PER_CLASS = 100


@dataclass(frozen=True)
class CIFAR10ProtocolDatasets:
    train: Dataset
    train_probe: Dataset
    validation_proper: Dataset
    validation_blurred: Dataset
    fim: FIMProbe
    train_probe_indices: np.ndarray
    split: CIFAR10Split
    dataset_root: str


@dataclass(frozen=True)
class CIFAR10ProtocolTestDatasets:
    test_proper: Dataset
    test_blurred: Dataset
    dataset_root: str


def _root(dataset_path):
    root = dataset_path or os.environ.get("CIFAR10_PATH")
    if not root:
        raise ValueError("set dataset_path or CIFAR10_PATH for CIFAR-10")
    return str(root)


def _train_probe_indices(
    targets,
    train_indices,
    per_class=TRAIN_PROBE_PER_CLASS,
    seed=TRAIN_PROBE_SEED,
):
    targets = np.asarray(targets, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    selected = []
    for label in np.unique(targets):
        candidates = train_indices[targets[train_indices] == label]
        if candidates.size < per_class:
            raise ValueError(f"class {label} has too few train-probe samples")
        selected.append(generator.permutation(candidates)[:per_class])
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _field_datasets(
    raw_dataset,
    *,
    overlap,
    resize_factor,
    normalization_profile,
):
    proper = SplitAndAugmentDataset(
        raw_dataset,
        transforms_cifar10.transform_eval_proper(
            overlap, "left", normalization_profile
        ),
        transforms_cifar10.transform_eval_proper(
            overlap, "right", normalization_profile
        ),
        overlap=overlap,
        is_train=False,
        reverse=False,
    )
    blurred = SplitAndAugmentDataset(
        raw_dataset,
        transforms_cifar10.transform_eval_proper(
            overlap, "left", normalization_profile
        ),
        transforms_cifar10.transform_eval_blurred(
            32, 32, resize_factor, overlap, normalization_profile
        ),
        overlap=overlap,
        is_train=False,
        reverse=False,
    )
    return proper, blurred


def build_cifar10_protocol_datasets(
    *,
    dataset_path=None,
    overlap=0.0,
    resize_factor=0.25,
    subset=None,
    normalization_profile=CIFAR10_PROTOCOL_PROFILE,
    verify_dataset_files=True,
) -> CIFAR10ProtocolDatasets:
    if normalization_profile != CIFAR10_PROTOCOL_PROFILE:
        raise ValueError(
            "the validation-controlled CIFAR-10 split requires "
            f"normalization_profile={CIFAR10_PROTOCOL_PROFILE!r}"
        )
    root = _root(dataset_path)
    manifest = load_cifar10_protocol_manifest()
    raw_train = datasets.CIFAR10(root, train=True, download=False)
    split = stratified_cifar10_split(raw_train.targets)
    validate_cifar10_protocol_split(raw_train.targets, split, manifest)
    if verify_dataset_files:
        validate_cifar10_dataset_files(root, manifest)

    proper_subset = None
    if subset is not None:
        proper_subset = np.ascontiguousarray(subset, dtype=np.int64).reshape(-1)
        excluded = np.setdiff1d(
            proper_subset, split.train_indices, assume_unique=False
        )
        if excluded.size:
            raise ValueError(
                "proper_right_subset_path contains validation/FIM or invalid "
                f"raw indices: {excluded[:8].tolist()}"
            )

    train_full = SplitAndAugmentDataset(
        raw_train,
        transforms_cifar10.transform_train_proper(
            overlap, "left", normalization_profile
        ),
        transforms_cifar10.transform_train_proper(
            overlap, "right", normalization_profile
        ),
        transform3=transforms_cifar10.transform_train_proper(
            overlap, "right", normalization_profile
        ),
        subset=proper_subset,
        overlap=overlap,
        is_train=True,
        reverse=True,
    )
    validation_proper_full, validation_blurred_full = _field_datasets(
        raw_train,
        overlap=overlap,
        resize_factor=resize_factor,
        normalization_profile=normalization_profile,
    )
    train_probe_indices = _train_probe_indices(
        raw_train.targets, split.train_indices
    )
    fim = build_fim_probe(
        raw_train,
        "mm_cifar10",
        overlap=overlap,
        resize_factor=resize_factor,
        probe_indices=split.fim_indices,
        train_indices=split.train_indices,
        normalization_profile=normalization_profile,
    )
    return CIFAR10ProtocolDatasets(
        train=Subset(train_full, split.train_indices.tolist()),
        train_probe=Subset(
            validation_proper_full, train_probe_indices.tolist()
        ),
        validation_proper=Subset(
            validation_proper_full, split.validation_indices.tolist()
        ),
        validation_blurred=Subset(
            validation_blurred_full, split.validation_indices.tolist()
        ),
        fim=fim,
        train_probe_indices=train_probe_indices,
        split=split,
        dataset_root=root,
    )


def build_cifar10_protocol_test_datasets(
    *,
    dataset_path=None,
    overlap=0.0,
    resize_factor=0.25,
    normalization_profile=CIFAR10_PROTOCOL_PROFILE,
) -> CIFAR10ProtocolTestDatasets:
    if normalization_profile != CIFAR10_PROTOCOL_PROFILE:
        raise ValueError(
            "the validation-controlled CIFAR-10 test datasets require the "
            "versioned 44k normalization profile"
        )
    root = _root(dataset_path)
    raw_test = datasets.CIFAR10(root, train=False, download=False)
    proper, blurred = _field_datasets(
        raw_test,
        overlap=overlap,
        resize_factor=resize_factor,
        normalization_profile=normalization_profile,
    )
    return CIFAR10ProtocolTestDatasets(
        test_proper=proper,
        test_blurred=blurred,
        dataset_root=root,
    )
