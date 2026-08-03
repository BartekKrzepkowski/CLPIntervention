"""Compute-node integration tests for the versioned CIFAR-10 protocol."""

import os
from itertools import islice

import numpy as np
import pytest
import torch
from torchvision.transforms import RandomAffine

from src.data.cifar10_protocol import CIFAR10_PROTOCOL_PROFILE
from src.utils.prepare_clp_data import prepare_training_loaders_clp
from src.utils.utils_data import get_targets


pytestmark = pytest.mark.compute


def _loaders():
    root = os.environ.get("CIFAR10_PATH")
    if not root:
        pytest.skip("CIFAR10_PATH is required on a compute node")
    return prepare_training_loaders_clp(
        "mm_cifar10",
        dataset_params={
            "dataset_path": root,
            "overlap": 0.0,
            "resize_factor": 0.25,
            "subset": None,
        },
        loader_params={
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        },
        split_profile=CIFAR10_PROTOCOL_PROFILE,
        normalization_profile=CIFAR10_PROTOCOL_PROFILE,
        generator_seed=83,
        verify_dataset_files=True,
    )


def test_real_protocol_loaders_have_disjoint_balanced_sources():
    loaders = _loaders()
    split = loaders.split_manifest
    assert split["counts"] == {
        "train": 44000,
        "validation": 5000,
        "fim": 1000,
    }
    train_targets = get_targets(loaders.train.dataset)
    validation_targets = get_targets(loaders.validation_proper.dataset)
    assert np.bincount(train_targets, minlength=10).tolist() == [4400] * 10
    assert np.bincount(validation_targets, minlength=10).tolist() == [500] * 10
    assert np.bincount(
        loaders.fim.tensors["y"].numpy(), minlength=10
    ).tolist() == [100] * 10


def test_eval_order_and_train_generator_resume_are_deterministic():
    loaders = _loaders()
    first_validation = [
        targets.clone()
        for _inputs, targets in islice(loaders.validation_proper, 2)
    ]
    second_validation = [
        targets.clone()
        for _inputs, targets in islice(loaders.validation_proper, 2)
    ]
    assert len(first_validation) == len(second_validation) == 2
    for first, second in zip(first_validation, second_validation):
        torch.testing.assert_close(first, second)

    state = loaders.state_dict()
    first_order = list(iter(loaders.train.sampler))
    loaders.load_state_dict(state)
    second_order = list(iter(loaders.train.sampler))
    assert first_order == second_order



def test_only_train_transforms_contain_random_augmentation():
    loaders = _loaders()
    train_dataset = loaders.train.dataset
    while not hasattr(train_dataset, "transform1"):
        train_dataset = train_dataset.dataset
    validation_dataset = loaders.validation_proper.dataset
    while not hasattr(validation_dataset, "transform1"):
        validation_dataset = validation_dataset.dataset

    assert any(
        isinstance(transform, RandomAffine)
        for transform in train_dataset.transform1.transforms
    )
    assert not any(
        isinstance(transform, RandomAffine)
        for transform in validation_dataset.transform1.transforms
    )
    assert loaders.validation_proper.drop_last is False
    assert loaders.validation_blurred.drop_last is False


def test_train_probe_is_balanced_deterministic_and_unaugmented():
    loaders = _loaders()
    probe_targets = get_targets(loaders.train_probe.dataset)
    assert np.bincount(probe_targets, minlength=10).tolist() == [100] * 10
    assert loaders.split_manifest["train_probe"]["count"] == 1000
    assert len(
        loaders.split_manifest["train_probe"]["indices_sha256"]
    ) == 64
    probe_dataset = loaders.train_probe.dataset
    while not hasattr(probe_dataset, "transform1"):
        probe_dataset = probe_dataset.dataset
    assert not any(
        isinstance(transform, RandomAffine)
        for transform in probe_dataset.transform1.transforms
    )
    assert loaders.train_probe.drop_last is False
