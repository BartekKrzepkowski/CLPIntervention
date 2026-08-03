"""Contracts for seed-paired unimodal validation references."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.utils.utils_trainer import load_checkpoint_metadata


REFERENCE_VERSION = 2
INITIALIZATION_POLICY = "canonical_bimodal_components_v2"
REFERENCE_MODALITIES = {"left_proper", "right_proper"}


@dataclass(frozen=True)
class UnimodalReferenceMetadata:
    modality: str
    validation_accuracy: float
    validation_loss: float
    selected_epoch: int
    seed: int
    model_name: str
    dataset_name: str
    split_profile: str
    normalization_profile: str
    split_manifest: dict
    normalization_manifest: dict
    initialization_policy: str
    source_bimodal_initial_state_sha256: str
    checkpoint_path: str
    version: int = REFERENCE_VERSION

    def __post_init__(self):
        if self.version != REFERENCE_VERSION:
            raise ValueError("unsupported unimodal reference version")
        if self.modality not in REFERENCE_MODALITIES:
            raise ValueError("unknown unimodal reference modality")
        if not 0.0 < self.validation_accuracy <= 1.0:
            raise ValueError("unimodal validation accuracy must be in (0, 1]")
        if self.validation_loss < 0.0:
            raise ValueError("unimodal validation loss must be non-negative")
        if self.selected_epoch < 0:
            raise ValueError("unimodal selected epoch must be non-negative")
        if self.initialization_policy != INITIALIZATION_POLICY:
            raise ValueError("unimodal references are not seed-paired")
        source_hash = self.source_bimodal_initial_state_sha256
        if not (
            isinstance(source_hash, str)
            and len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError(
                "source bimodal initialization hash must be exactly 64 "
                "lowercase hexadecimal characters"
            )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        payload = load_checkpoint_metadata(checkpoint_path)
        metadata = payload["metadata"].get("unimodal_reference")
        if metadata is None:
            raise ValueError(
                f"checkpoint {checkpoint_path!s} lacks unimodal reference metadata"
            )
        values = dict(metadata)
        if values.get("version") != REFERENCE_VERSION:
            raise ValueError("unsupported unimodal reference version")
        values["checkpoint_path"] = str(checkpoint_path)
        return cls(**values)


def validate_unimodal_reference_pair(
    left: UnimodalReferenceMetadata,
    right: UnimodalReferenceMetadata,
    *,
    seed: int,
    model_name: str,
    dataset_name: str,
    split_profile: str,
    normalization_profile: str,
    split_manifest: dict,
    normalization_manifest: dict,
):
    """Reject references incompatible with the current bimodal protocol."""
    expected = {
        "seed": int(seed),
        "model_name": str(model_name),
        "dataset_name": str(dataset_name),
        "split_profile": str(split_profile),
        "normalization_profile": str(normalization_profile),
        "split_manifest": split_manifest,
        "normalization_manifest": normalization_manifest,
    }
    for reference, modality in (
        (left, "left_proper"),
        (right, "right_proper"),
    ):
        if reference.modality != modality:
            raise ValueError(
                f"expected {modality} reference, got {reference.modality}"
            )
        for name, expected_value in expected.items():
            if getattr(reference, name) != expected_value:
                raise ValueError(
                    f"unimodal {modality} reference {name} does not match "
                    "the current protocol"
                )
    if (
        left.source_bimodal_initial_state_sha256
        != right.source_bimodal_initial_state_sha256
    ):
        raise ValueError(
            "unimodal reference source bimodal initialization hashes do not match"
        )
    return left, right


def load_and_validate_unimodal_reference_pair(
    section,
    **expected,
):
    left_path = section.get("left_checkpoint")
    right_path = section.get("right_checkpoint")
    if not left_path or not right_path:
        raise ValueError(
            "relative_unimodal_parity requires left_checkpoint and "
            "right_checkpoint"
        )
    left = UnimodalReferenceMetadata.from_checkpoint(left_path)
    right = UnimodalReferenceMetadata.from_checkpoint(right_path)
    return validate_unimodal_reference_pair(left, right, **expected)
