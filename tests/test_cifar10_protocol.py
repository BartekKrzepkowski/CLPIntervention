import numpy as np
import pytest
from PIL import Image

from src.data import transforms_cifar10
from src.data.cifar10_protocol import (
    CIFAR10_PROTOCOL_PROFILE,
    CIFAR10SplitSpec,
    array_sha256,
    load_cifar10_protocol_manifest,
    stratified_cifar10_split,
    validate_cifar10_protocol_split,
)
from src.data.normalization import normalization_from_transform
from src.data.probes import build_fim_probe


def test_publication_split_has_exact_counts_is_disjoint_and_deterministic():
    targets = np.repeat(np.arange(10, dtype=np.int64), 5000)
    first = stratified_cifar10_split(targets)
    second = stratified_cifar10_split(targets)
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.validation_indices, second.validation_indices)
    assert np.array_equal(first.fim_indices, second.fim_indices)
    assert np.bincount(targets[first.train_indices]).tolist() == [4400] * 10
    assert np.bincount(targets[first.validation_indices]).tolist() == [500] * 10
    assert np.bincount(targets[first.fim_indices]).tolist() == [100] * 10
    combined = np.concatenate(
        (first.train_indices, first.validation_indices, first.fim_indices)
    )
    assert np.unique(combined).size == 50000


def test_publication_manifest_records_fixed_counts_and_hashes():
    manifest = load_cifar10_protocol_manifest()
    assert manifest["counts"] == {
        "train": 44000,
        "validation": 5000,
        "fim": 1000,
    }
    assert manifest["class_counts"]["train"] == [4400] * 10
    assert manifest["class_counts"]["validation"] == [500] * 10
    assert manifest["class_counts"]["fim"] == [100] * 10
    assert all(
        len(digest) == 64 for digest in manifest["indices_sha256"].values()
    )


def test_manifest_validation_fails_closed_on_changed_targets():
    targets = np.repeat(np.arange(2, dtype=np.int64), 8)
    spec = CIFAR10SplitSpec(
        profile=CIFAR10_PROTOCOL_PROFILE,
        seed=83,
        validation_per_class=2,
        fim_per_class=1,
    )
    split = stratified_cifar10_split(targets, spec)
    state = split.state_dict()
    manifest = {
        "profile": spec.profile,
        "split": {
            "seed": spec.seed,
            "validation_per_class": spec.validation_per_class,
            "fim_per_class": spec.fim_per_class,
        },
        "counts": state["counts"],
        "indices_sha256": state["indices_sha256"],
        "targets_sha256": array_sha256(targets),
    }
    validate_cifar10_protocol_split(targets, split, manifest)
    changed = targets.copy()
    changed[0] = 1
    with pytest.raises(ValueError, match="targets"):
        validate_cifar10_protocol_split(changed, split, manifest)


def test_protocol_normalization_uses_44k_manifest_values():
    manifest = load_cifar10_protocol_manifest()
    transform = transforms_cifar10.transform_eval_proper(
        0.0, "left", CIFAR10_PROTOCOL_PROFILE
    )
    actual = normalization_from_transform(transform)
    expected = manifest["normalization"]["statistics"]["0.0"]["proper_left"]
    assert actual == expected
    with pytest.raises(ValueError, match="configured overlaps"):
        transforms_cifar10.transform_eval_proper(
            0.125, "left", CIFAR10_PROTOCOL_PROFILE
        )


class TinyImages:
    classes = ("zero", "one")

    def __init__(self):
        self.targets = [0, 0, 0, 1, 1, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = np.full((8, 8, 3), index * 20, dtype=np.uint8)
        return Image.fromarray(pixels), self.targets[index]


def test_fim_probe_accepts_fixed_indices_disjoint_from_train():
    probe = build_fim_probe(
        TinyImages(),
        "mm_cifar10",
        probe_indices=np.array([0, 3]),
        train_indices=np.array([1, 2, 4, 5]),
    )
    assert probe.probe_indices.tolist() == [0, 3]
    assert probe.train_indices.tolist() == [1, 2, 4, 5]
    assert probe.tensors["y"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="disjoint"):
        build_fim_probe(
            TinyImages(),
            "mm_cifar10",
            probe_indices=np.array([0, 3]),
            train_indices=np.array([0, 1]),
        )
