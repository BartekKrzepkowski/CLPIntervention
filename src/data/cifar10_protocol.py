"""Versioned CIFAR-10 train/validation/FIM protocol splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CIFAR10_PROTOCOL_PROFILE = "cifar10_stratified_44k_5k_1k_seed83_v1"
CIFAR10_LEGACY_NORMALIZATION_PROFILE = "cifar10_train50k_v1"
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "data"
    / f"{CIFAR10_PROTOCOL_PROFILE}.json"
)


def _canonical_indices(indices) -> np.ndarray:
    canonical = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1)
    if canonical.size and (
        canonical.min(initial=0) < 0
        or np.unique(canonical).size != canonical.size
    ):
        raise ValueError("split indices must be unique and non-negative")
    return np.sort(canonical)


def array_sha256(values) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class CIFAR10SplitSpec:
    profile: str = CIFAR10_PROTOCOL_PROFILE
    seed: int = 83
    validation_per_class: int = 500
    fim_per_class: int = 100


@dataclass(frozen=True)
class CIFAR10Split:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    fim_indices: np.ndarray
    spec: CIFAR10SplitSpec

    def __post_init__(self):
        object.__setattr__(
            self, "train_indices", _canonical_indices(self.train_indices)
        )
        object.__setattr__(
            self,
            "validation_indices",
            _canonical_indices(self.validation_indices),
        )
        object.__setattr__(
            self, "fim_indices", _canonical_indices(self.fim_indices)
        )
        groups = (
            self.train_indices,
            self.validation_indices,
            self.fim_indices,
        )
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if np.intersect1d(left, right).size:
                    raise ValueError("CIFAR-10 protocol splits overlap")

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile": self.spec.profile,
            "seed": self.spec.seed,
            "validation_per_class": self.spec.validation_per_class,
            "fim_per_class": self.spec.fim_per_class,
            "counts": {
                "train": int(self.train_indices.size),
                "validation": int(self.validation_indices.size),
                "fim": int(self.fim_indices.size),
            },
            "indices_sha256": {
                "train": array_sha256(self.train_indices),
                "validation": array_sha256(self.validation_indices),
                "fim": array_sha256(self.fim_indices),
            },
        }


def stratified_cifar10_split(
    targets,
    spec: CIFAR10SplitSpec = CIFAR10SplitSpec(),
) -> CIFAR10Split:
    """Generate the fixed class-balanced split from raw CIFAR-10 targets."""
    targets = np.ascontiguousarray(targets, dtype=np.int64).reshape(-1)
    if targets.size == 0:
        raise ValueError("targets must not be empty")
    classes = np.unique(targets)
    generator = np.random.default_rng(spec.seed)
    train_indices = []
    validation_indices = []
    fim_indices = []
    for label in classes:
        candidates = np.flatnonzero(targets == label)
        required = spec.validation_per_class + spec.fim_per_class
        if candidates.size <= required:
            raise ValueError(
                f"class {label} needs more than {required} examples, "
                f"got {candidates.size}"
            )
        shuffled = generator.permutation(candidates)
        validation_indices.append(shuffled[: spec.validation_per_class])
        fim_indices.append(
            shuffled[
                spec.validation_per_class :
                spec.validation_per_class + spec.fim_per_class
            ]
        )
        train_indices.append(shuffled[required:])
    return CIFAR10Split(
        train_indices=np.concatenate(train_indices),
        validation_indices=np.concatenate(validation_indices),
        fim_indices=np.concatenate(fim_indices),
        spec=spec,
    )


@lru_cache(maxsize=4)
def load_cifar10_protocol_manifest(
    path: str | Path = _MANIFEST_PATH,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CIFAR-10 protocol manifest: {path}. "
            "Generate and verify it before using the publication profile."
        )
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("profile") != CIFAR10_PROTOCOL_PROFILE:
        raise ValueError(
            f"Unexpected CIFAR-10 profile in {path}: {manifest.get('profile')}"
        )
    return manifest


def validate_cifar10_protocol_split(
    targets,
    split: CIFAR10Split,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    manifest = dict(manifest or load_cifar10_protocol_manifest())
    expected = {
        "profile": manifest["profile"],
        "seed": int(manifest["split"]["seed"]),
        "validation_per_class": int(
            manifest["split"]["validation_per_class"]
        ),
        "fim_per_class": int(manifest["split"]["fim_per_class"]),
        "counts": {
            key: int(value) for key, value in manifest["counts"].items()
        },
        "indices_sha256": dict(manifest["indices_sha256"]),
    }
    actual = split.state_dict()
    for key in (
        "profile",
        "seed",
        "validation_per_class",
        "fim_per_class",
        "counts",
        "indices_sha256",
    ):
        if actual[key] != expected[key]:
            raise ValueError(
                f"CIFAR-10 split manifest mismatch for {key}: "
                f"expected {expected[key]!r}, got {actual[key]!r}"
            )
    target_hash = array_sha256(targets)
    if target_hash != manifest["targets_sha256"]:
        raise ValueError(
            "CIFAR-10 targets do not match the versioned protocol manifest"
        )
    combined = np.sort(
        np.concatenate(
            (
                split.train_indices,
                split.validation_indices,
                split.fim_indices,
            )
        )
    )
    if not np.array_equal(combined, np.arange(len(targets), dtype=np.int64)):
        raise ValueError("CIFAR-10 protocol split does not cover the train set")


def validate_cifar10_dataset_files(
    dataset_root: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed if the raw CIFAR-10 training files differ from the manifest."""
    manifest = dict(manifest or load_cifar10_protocol_manifest())
    batch_root = Path(dataset_root) / "cifar-10-batches-py"
    file_hashes = manifest.get(
        "dataset_file_sha256", manifest.get("train_file_sha256", {})
    )
    if not file_hashes:
        raise ValueError("CIFAR-10 manifest has no dataset file identities")
    for name, expected_hash in file_hashes.items():
        path = batch_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing CIFAR-10 training file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as batch_file:
            for chunk in iter(lambda: batch_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"CIFAR-10 file hash mismatch: {path}")


def cifar10_protocol_normalization(
    field: str,
    overlap: float,
    manifest: Mapping[str, Any] | None = None,
):
    manifest = dict(manifest or load_cifar10_protocol_manifest())
    try:
        statistics = manifest["normalization"]["statistics"][str(overlap)][
            field
        ]
    except KeyError as error:
        supported = ", ".join(
            sorted(manifest["normalization"]["statistics"])
        )
        raise ValueError(
            f"Unsupported CIFAR-10 protocol normalization: field={field}, "
            f"overlap={overlap}; configured overlaps: {supported}"
        ) from error
    return tuple(statistics["mean"]), tuple(statistics["std"])
