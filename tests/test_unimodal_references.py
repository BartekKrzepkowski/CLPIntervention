import json

import pytest

from src.trainer.unimodal_references import (
    UnimodalReferenceMetadata,
    validate_unimodal_reference_pair,
)
from src.trainer.validation_control import (
    InterventionCheckpointRecord,
    ModeMetrics,
    ModalityEvaluationResult,
    Phase3StopDecision,
    UnimodalCheckpointSelector,
    phase3_trajectory_record,
)


def reference(modality, **overrides):
    values = {
        "modality": modality,
        "validation_accuracy": 0.8,
        "validation_loss": 0.7,
        "selected_epoch": 100,
        "seed": 83,
        "model_name": "mm_resnet",
        "dataset_name": "mm_cifar10",
        "split_profile": "split-v1",
        "normalization_profile": "norm-v1",
        "split_manifest": {"indices_sha256": "split"},
        "normalization_manifest": {"proper_left": [0.1, 0.2]},
        "initialization_policy": "canonical_bimodal_components_v2",
        "source_bimodal_initial_state_sha256": "a" * 64,
        "checkpoint_path": f"{modality}.pth",
    }
    values.update(overrides)
    return UnimodalReferenceMetadata(**values)


def expected():
    return {
        "seed": 83,
        "model_name": "mm_resnet",
        "dataset_name": "mm_cifar10",
        "split_profile": "split-v1",
        "normalization_profile": "norm-v1",
        "split_manifest": {"indices_sha256": "split"},
        "normalization_manifest": {"proper_left": [0.1, 0.2]},
    }


def test_unimodal_selector_ranks_accuracy_then_loss_then_earlier_epoch():
    selector = UnimodalCheckpointSelector("left_proper")
    selector.update(ModeMetrics(1.0, 0.7), 5, "low_accuracy")
    selector.update(ModeMetrics(1.2, 0.8), 6, "high_accuracy")
    selector.update(ModeMetrics(0.9, 0.8), 7, "low_loss")
    selector.update(ModeMetrics(0.9, 0.8), 8, "later_tie")
    assert selector.best.checkpoint_path == "low_loss"
    assert selector.retained_checkpoint_paths == {"low_loss"}


def test_unimodal_selector_retains_best_after_non_improving_look():
    selector = UnimodalCheckpointSelector("right_proper")
    selector.update(ModeMetrics(0.7, 0.8), 5, "best")
    improved, _ = selector.update(
        ModeMetrics(0.8, 0.7), 10, "not_saved"
    )
    assert not improved
    assert selector.retained_checkpoint_paths == {"best"}


def test_unimodal_selector_state_round_trip():
    selector = UnimodalCheckpointSelector("right_proper")
    selector.update(ModeMetrics(0.7, 0.8), 12, "best")
    restored = UnimodalCheckpointSelector("right_proper")
    restored.load_state_dict(selector.state_dict())
    assert restored.best == selector.best


def test_unimodal_reference_pair_accepts_matching_protocol():
    left, right = validate_unimodal_reference_pair(
        reference("left_proper"),
        reference("right_proper"),
        **expected(),
    )
    assert left.modality == "left_proper"
    assert right.modality == "right_proper"


@pytest.mark.parametrize(
    ("side", "field", "value"),
    [
        ("left", "seed", 184),
        ("right", "model_name", "mm_resnet18"),
        ("right", "split_manifest", {"indices_sha256": "other"}),
        ("left", "normalization_profile", "other-norm"),
    ],
)
def test_unimodal_reference_pair_rejects_protocol_mismatch(
    side, field, value
):
    references = {
        "left": reference("left_proper"),
        "right": reference("right_proper"),
    }
    references[side] = reference(
        f"{side}_proper", **{field: value}
    )
    with pytest.raises(ValueError, match=field):
        validate_unimodal_reference_pair(
            references["left"], references["right"], **expected()
        )


def test_unimodal_reference_rejects_unpaired_initialization():
    with pytest.raises(ValueError, match="not seed-paired"):
        reference(
            "left_proper", initialization_policy="independent"
        )


@pytest.mark.parametrize(
    "source_hash",
    [
        "a" * 63,
        "A" * 64,
        "g" * 64,
    ],
)
def test_unimodal_reference_rejects_malformed_source_model_hash(source_hash):
    with pytest.raises(
        ValueError,
        match=(
            "source bimodal initialization hash must be exactly 64 "
            "lowercase hexadecimal characters"
        ),
    ):
        reference(
            "left_proper",
            source_bimodal_initial_state_sha256=source_hash,
        )


def test_unimodal_reference_rejects_legacy_version():
    with pytest.raises(
        ValueError, match="unsupported unimodal reference version"
    ):
        reference("left_proper", version=1)


def test_unimodal_reference_checkpoint_rejects_legacy_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.trainer.unimodal_references.load_checkpoint_metadata",
        lambda _path: {
            "metadata": {"unimodal_reference": {"version": 1}}
        },
    )
    with pytest.raises(
        ValueError, match="unsupported unimodal reference version"
    ):
        UnimodalReferenceMetadata.from_checkpoint("legacy.pth")


def test_unimodal_reference_pair_rejects_different_source_model_hashes():
    with pytest.raises(ValueError, match="initialization hashes do not match"):
        validate_unimodal_reference_pair(
            reference("left_proper"),
            reference(
                "right_proper",
                source_bimodal_initial_state_sha256="b" * 64,
            ),
            **expected(),
        )


def test_phase3_trajectory_record_keeps_raw_metrics_and_selection():
    metrics = ModalityEvaluationResult(
        full=ModeMetrics(0.5, 0.8),
        dominant_only=ModeMetrics(0.7, 0.7),
        weak_only=ModeMetrics(0.8, 0.6),
        intervention=ModeMetrics(0.8, 0.6),
        phase_epoch=8,
        global_epoch=248,
        global_step=1234,
    )
    current = InterventionCheckpointRecord(
        metrics=metrics,
        checkpoint_path="e8.pth",
        weak_quality_gain=0.1,
        weak_utility_gain=0.2,
        full_loss_increase=0.0,
        dominant_loss_increase=0.0,
        compatibility_drift_accuracy=0.0,
        reactivation_full_loss_gap=0.0,
        is_feasible=True,
        is_safe=True,
        dominant_ratio=0.8,
        weak_ratio=0.82,
        parity_gap=0.02,
    )
    decision = Phase3StopDecision(
        should_stop=True,
        stop_reason="relative_parity_reached",
        selection_status="first_parity_checkpoint",
        selected=current,
        current=current,
        bad_checks=2,
        safety_bad_checks=0,
    )
    record = phase3_trajectory_record(
        metrics,
        decision_rule="relative_unimodal_parity",
        checkpoint_path="e8.pth",
        checkpoint_retained=True,
        current_record=current,
        decision=decision,
        unimodal_references={"left": {"validation_accuracy": 0.85}},
    )
    assert record["version"] == 1
    assert record["metrics"]["weak_only"]["accuracy"] == 0.6
    assert record["controller"]["parity_gap"] == 0.02
    assert record["controller"]["selected_epoch"] == 8
    assert record["checkpoint_retained"] is True
    json.dumps(record)
