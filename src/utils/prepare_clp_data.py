"""Loader construction for the validation-controlled CLP protocol."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cifar10_datasets import (
    TRAIN_PROBE_PER_CLASS,
    TRAIN_PROBE_SEED,
    build_cifar10_protocol_datasets,
    build_cifar10_protocol_test_datasets,
)
from src.data.cifar10_protocol import CIFAR10_PROTOCOL_PROFILE, array_sha256
from src.data.probes import FIMProbe


def seed_loader_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _evaluation_loader(dataset, loader_params):
    params = dict(loader_params)
    params.pop("shuffle", None)
    params.pop("drop_last", None)
    params.pop("generator", None)
    params.pop("worker_init_fn", None)
    return DataLoader(dataset, shuffle=False, drop_last=False, **params)


@dataclass
class CLPTrainingLoaders(Mapping[str, DataLoader]):
    train: DataLoader
    train_probe: DataLoader
    validation_proper: DataLoader
    validation_blurred: DataLoader
    fim: FIMProbe
    train_generator: torch.Generator
    split_manifest: dict[str, Any]
    dataset_root: str

    def __getitem__(self, key: str) -> DataLoader:
        return {
            "train": self.train,
            "train_probe": self.train_probe,
            "validation_proper": self.validation_proper,
            "validation_blurred": self.validation_blurred,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(
            ("train", "train_probe", "validation_proper", "validation_blurred")
        )

    def __len__(self) -> int:
        return 4

    def state_dict(self):
        return {"train_generator_state": self.train_generator.get_state()}

    def load_state_dict(self, state):
        if state and state.get("train_generator_state") is not None:
            self.train_generator.set_state(
                state["train_generator_state"].detach().cpu()
            )


@dataclass(frozen=True)
class CLPTestLoaders(Mapping[str, DataLoader]):
    test_proper: DataLoader
    test_blurred: DataLoader
    dataset_root: str

    def __getitem__(self, key: str) -> DataLoader:
        return {
            "test_proper": self.test_proper,
            "test_blurred": self.test_blurred,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("test_proper", "test_blurred"))

    def __len__(self) -> int:
        return 2


def prepare_training_loaders_clp(
    dataset_name,
    dataset_params,
    loader_params,
    *,
    split_profile=CIFAR10_PROTOCOL_PROFILE,
    normalization_profile=CIFAR10_PROTOCOL_PROFILE,
    generator_seed=83,
    verify_dataset_files=True,
) -> CLPTrainingLoaders:
    if dataset_name != "mm_cifar10":
        raise ValueError(
            "the validation-controlled loader protocol currently supports "
            "only mm_cifar10"
        )
    if split_profile != CIFAR10_PROTOCOL_PROFILE:
        raise ValueError(f"Unsupported CIFAR-10 split profile: {split_profile}")
    datasets = build_cifar10_protocol_datasets(
        **dataset_params,
        normalization_profile=normalization_profile,
        verify_dataset_files=verify_dataset_files,
    )
    generator = torch.Generator().manual_seed(int(generator_seed))
    train_params = dict(loader_params)
    train_params.pop("shuffle", None)
    train_params.pop("generator", None)
    train_params.pop("worker_init_fn", None)
    train_loader = DataLoader(
        datasets.train,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_loader_worker,
        **train_params,
    )
    split_manifest = datasets.split.state_dict()
    split_manifest["train_probe"] = {
        "count": int(datasets.train_probe_indices.size),
        "indices_sha256": array_sha256(datasets.train_probe_indices),
        "seed": TRAIN_PROBE_SEED,
        "per_class": TRAIN_PROBE_PER_CLASS,
        "part_of_training": True,
    }
    return CLPTrainingLoaders(
        train=train_loader,
        train_probe=_evaluation_loader(datasets.train_probe, loader_params),
        validation_proper=_evaluation_loader(
            datasets.validation_proper, loader_params
        ),
        validation_blurred=_evaluation_loader(
            datasets.validation_blurred, loader_params
        ),
        fim=datasets.fim,
        train_generator=generator,
        split_manifest=split_manifest,
        dataset_root=datasets.dataset_root,
    )


def prepare_test_loaders_clp(
    dataset_name,
    dataset_params,
    loader_params,
    *,
    normalization_profile=CIFAR10_PROTOCOL_PROFILE,
) -> CLPTestLoaders:
    if dataset_name != "mm_cifar10":
        raise ValueError(
            "the validation-controlled loader protocol currently supports "
            "only mm_cifar10"
        )
    test_params = {
        key: value
        for key, value in dataset_params.items()
        if key != "subset"
    }
    datasets = build_cifar10_protocol_test_datasets(
        **test_params,
        normalization_profile=normalization_profile,
    )
    return CLPTestLoaders(
        test_proper=_evaluation_loader(datasets.test_proper, loader_params),
        test_blurred=_evaluation_loader(datasets.test_blurred, loader_params),
        dataset_root=datasets.dataset_root,
    )
