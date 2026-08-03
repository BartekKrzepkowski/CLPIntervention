import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.trainer.trainer_validation_clp import ValidationControlledTrainer
from src.trainer.modality_evaluation import (
    WEAK_ONLY_MODE,
    evaluate_modalities,
)
from src.trainer.gradient_diagnostics import (
    evaluate_phase3_gradient_diagnostics,
)
from src.trainer.validation_control import (
    ModalityEvaluationResult,
    ModeMetrics,
    PerExampleModeLosses,
    PerExampleModeCorrectness,
    Phase2CheckpointRecord,
    Phase2CheckpointSelector,
    Phase2PlateauConfig,
    Phase2PlateauDetector,
    Phase3InterventionStopper,
    Phase3LocalAccuracyStopper,
    Phase3RelativeUnimodalStopper,
    should_evaluate_phase_epoch,
    Phase3RecoveryStopper,
    Phase3StopConfig,
    Phase4CheckpointRecord,
    Phase4CheckpointSelector,
)


def parity_result(epoch, *, weak_accuracy, dominant_accuracy=0.72, weak_loss=1.0):
    return ModalityEvaluationResult(
        full=ModeMetrics(0.8, 0.82),
        dominant_only=ModeMetrics(1.0, dominant_accuracy),
        weak_only=ModeMetrics(weak_loss, weak_accuracy),
        intervention=ModeMetrics(weak_loss, weak_accuracy),
        phase_epoch=epoch,
        global_epoch=epoch + 10,
        global_step=epoch * 7,
    )


def parity_stopper(baseline=None, **overrides):
    values = {
        "decision_rule": "relative_unimodal_parity",
        "min_epochs": 1,
        "max_epochs": 200,
        "parity_patience": 2,
        "emergency_stop_mode": "numerical_only",
    }
    values.update(overrides)
    return Phase3RelativeUnimodalStopper(
        Phase3StopConfig(**values),
        baseline or parity_result(0, weak_accuracy=0.2),
        unimodal_left_accuracy=0.9,
        unimodal_right_accuracy=0.8,
    )


def test_relative_parity_uses_normalized_branch_accuracies():
    stopper = parity_stopper()
    baseline = stopper.initialize_baseline("pre")
    assert not baseline.should_stop
    assert baseline.current.dominant_ratio == pytest.approx(0.8)
    assert baseline.current.weak_ratio == pytest.approx(0.25)
    assert not stopper.update(
        parity_result(1, weak_accuracy=0.63), "below"
    ).should_stop
    first = stopper.update(parity_result(2, weak_accuracy=0.65), "first")
    assert not first.should_stop
    confirmed = stopper.update(
        parity_result(3, weak_accuracy=0.64), "confirmation"
    )
    assert confirmed.should_stop
    assert confirmed.stop_reason == "relative_parity_reached"
    assert confirmed.selected.checkpoint_path == "first"


@pytest.mark.parametrize(
    ("threshold", "below_accuracy", "qualifying_accuracy"),
    [
        (0.90, 0.592, 0.600),
        (0.95, 0.616, 0.620),
    ],
)
def test_relative_recovery_fraction_closes_configured_baseline_deficit(
    threshold, below_accuracy, qualifying_accuracy
):
    stopper = parity_stopper(recovery_fraction_threshold=threshold)
    baseline = stopper.initialize_baseline("pre")
    assert baseline.current.recovery_fraction == pytest.approx(0.0)
    assert baseline.current.recovery_fraction_threshold == threshold

    below = stopper.update(
        parity_result(1, weak_accuracy=below_accuracy), "below"
    )
    assert not below.should_stop
    first = stopper.update(
        parity_result(2, weak_accuracy=qualifying_accuracy), "first"
    )
    assert not first.should_stop
    confirmed = stopper.update(
        parity_result(3, weak_accuracy=qualifying_accuracy + 0.008),
        "confirmation",
    )
    assert confirmed.should_stop
    assert confirmed.stop_reason == "relative_recovery_reached"
    assert confirmed.selection_status == "first_recovery_checkpoint"
    assert confirmed.selected.checkpoint_path == "first"


def test_relative_recovery_resume_rejects_changed_threshold():
    stopper = parity_stopper(recovery_fraction_threshold=0.90)
    stopper.initialize_baseline("pre")
    stopper.update(parity_result(1, weak_accuracy=0.60), "first")
    changed = parity_stopper(recovery_fraction_threshold=0.95)
    with pytest.raises(ValueError, match="recovery target changed"):
        changed.load_state_dict(stopper.state_dict())


def test_relative_recovery_threshold_must_be_a_fraction():
    with pytest.raises(ValueError, match="must be in"):
        Phase3StopConfig(
            decision_rule="relative_unimodal_parity",
            recovery_fraction_threshold=0.0,
        )


def test_relative_parity_resets_interrupted_confirmation_streak():
    stopper = parity_stopper()
    stopper.initialize_baseline("pre")
    stopper.update(parity_result(1, weak_accuracy=0.65), "discarded")
    stopper.update(parity_result(2, weak_accuracy=0.60), "below")
    stopper.update(parity_result(3, weak_accuracy=0.66), "new_first")
    decision = stopper.update(parity_result(4, weak_accuracy=0.67), "second")
    assert decision.selected.checkpoint_path == "new_first"


def test_relative_parity_skips_phase3_when_baseline_already_meets_target():
    stopper = parity_stopper(
        baseline=parity_result(0, weak_accuracy=0.64)
    )
    decision = stopper.initialize_baseline("pre")
    assert decision.should_stop
    assert decision.selected.metrics.phase_epoch == 0
    assert decision.selection_status == "first_parity_checkpoint"


def test_relative_parity_max_epoch_selects_best_weak_ratio():
    stopper = parity_stopper(max_epochs=3)
    stopper.initialize_baseline("pre")
    stopper.update(parity_result(1, weak_accuracy=0.50), "one")
    stopper.update(parity_result(2, weak_accuracy=0.60), "best")
    decision = stopper.update(parity_result(3, weak_accuracy=0.55), "last")
    assert decision.stop_reason == "max_epochs"
    assert decision.selected.checkpoint_path == "best"
    assert decision.selection_status == "best_weak_ratio"


def test_relative_parity_numerical_emergency_uses_last_finite_best():
    stopper = parity_stopper()
    stopper.initialize_baseline("pre")
    stopper.update(parity_result(1, weak_accuracy=0.5), "finite")
    decision = stopper.update(
        parity_result(2, weak_accuracy=float("nan")), "nonfinite"
    )
    assert decision.stop_reason == "numerical_emergency"
    assert decision.selected.checkpoint_path == "finite"


def test_relative_parity_resume_preserves_first_confirmation_checkpoint():
    stopper = parity_stopper()
    stopper.initialize_baseline("pre")
    stopper.update(parity_result(1, weak_accuracy=0.65), "first")
    restored = parity_stopper()
    restored.load_state_dict(stopper.state_dict())
    decision = restored.update(
        parity_result(2, weak_accuracy=0.66), "confirmation"
    )
    assert decision.selected.checkpoint_path == "first"


def result(
    epoch,
    *,
    full=1.0,
    dominant=1.2,
    weak=1.5,
    intervention=None,
    accuracy=0.5,
):
    return ModalityEvaluationResult(
        full=ModeMetrics(full, accuracy),
        dominant_only=ModeMetrics(dominant, accuracy - 0.05),
        weak_only=ModeMetrics(weak, accuracy - 0.1),
        intervention=ModeMetrics(
            weak if intervention is None else intervention,
            accuracy - 0.1,
        ),
        phase_epoch=epoch,
        global_epoch=epoch + 10,
        global_step=epoch * 7,
    )


def test_modality_metrics_load_legacy_state_without_calibration_fields():
    state = result(1).state_dict()
    calibration_fields = (
        "nll",
        "brier",
        "ece",
        "mean_confidence",
        "mean_incorrect_confidence",
    )
    for mode in ("full", "dominant_only", "weak_only", "intervention"):
        for field in calibration_fields:
            state[mode].pop(field)
    restored = ModalityEvaluationResult.from_state_dict(state)
    assert restored.full.loss == pytest.approx(1.0)
    assert restored.full.nll is None

def phase2_config(**overrides):
    values = {
        "min_epochs": 4,
        "min_delta_full_loss": 0.01,
        "full_loss_patience": 2,
        "stability_window": 3,
        "max_abs_slope_weak_loss": 0.01,
        "max_abs_slope_weak_utility": 0.01,
        "plateau_confirmations": 2,
        "selection_window": 3,
    }
    values.update(overrides)
    return Phase2PlateauConfig(**values)


def test_phase2_plateau_is_not_detected_before_min_epochs():
    detector = Phase2PlateauDetector(phase2_config(min_epochs=6))
    for epoch in range(1, 6):
        decision = detector.update(result(epoch))
        assert not decision.should_stop


def test_phase2_plateau_waits_while_weak_loss_keeps_improving():
    detector = Phase2PlateauDetector(phase2_config())
    for epoch in range(1, 9):
        decision = detector.update(result(epoch, weak=2.0 - 0.1 * epoch))
    assert not decision.should_stop
    assert abs(decision.weak_loss_slope) > 0.01


def test_phase2_plateau_waits_while_weak_utility_keeps_growing():
    detector = Phase2PlateauDetector(phase2_config())
    for epoch in range(1, 9):
        decision = detector.update(
            result(epoch, dominant=1.2 + 0.1 * epoch)
        )
    assert not decision.should_stop
    assert abs(decision.weak_utility_slope) > 0.01


def test_phase2_plateau_requires_all_signals_and_confirmations():
    detector = Phase2PlateauDetector(phase2_config())
    decisions = [detector.update(result(epoch)) for epoch in range(1, 6)]
    assert not decisions[-2].should_stop
    assert decisions[-1].should_stop
    assert decisions[-1].stop_reason == "plateau_detected"


def test_phase2_selector_uses_only_final_window_and_tie_breakers():
    selector = Phase2CheckpointSelector(3)
    selector.add(Phase2CheckpointRecord(result(1, full=0.1), "old"))
    selector.add(Phase2CheckpointRecord(result(2, full=0.5), "two"))
    selector.add(Phase2CheckpointRecord(result(3, full=0.4), "three"))
    selector.add(Phase2CheckpointRecord(result(4, full=0.3), "four"))
    assert selector.best.checkpoint_path == "four"

    tie = Phase2CheckpointSelector(4)
    tie.add(
        Phase2CheckpointRecord(
            result(1, full=1.0, dominant=1.3, weak=1.2), "utility"
        )
    )
    tie.add(
        Phase2CheckpointRecord(
            result(2, full=1.0, dominant=1.2, weak=0.8), "weak"
        )
    )
    assert tie.best.checkpoint_path == "utility"
    tie.add(
        Phase2CheckpointRecord(
            result(3, full=1.0, dominant=1.3, weak=1.1), "lower_weak"
        )
    )
    assert tie.best.checkpoint_path == "lower_weak"



def test_phase2_global_selector_retains_loss_and_accuracy_best_and_resumes():
    selector = Phase2CheckpointSelector(8, "global")
    assert selector.add(
        Phase2CheckpointRecord(result(1, full=0.2, accuracy=0.5), "loss")
    )
    assert selector.add(
        Phase2CheckpointRecord(result(2, full=0.4, accuracy=0.7), "accuracy")
    )
    assert selector.best_loss.checkpoint_path == "loss"
    assert selector.best_accuracy.checkpoint_path == "accuracy"
    assert len(selector.records) == 2

    restored = Phase2CheckpointSelector(8, "global")
    restored.load_state_dict(selector.state_dict())
    assert restored.best_loss == selector.best_loss
    assert restored.best_accuracy == selector.best_accuracy
    assert restored.selection_scope == "global"

    assert restored.add(
        Phase2CheckpointRecord(
            result(3, full=0.1, accuracy=0.6), "improved_loss"
        )
    )
    assert restored.best_loss.checkpoint_path == "improved_loss"
    assert restored.best_accuracy.checkpoint_path == "accuracy"

def test_phase2_controller_state_round_trip_is_deterministic():
    config = phase2_config()
    detector = Phase2PlateauDetector(config)
    selector = Phase2CheckpointSelector(config.selection_window)
    for epoch in range(1, 4):
        metrics = result(epoch)
        detector.update(metrics)
        selector.add(Phase2CheckpointRecord(metrics, str(epoch)))
    restored_detector = Phase2PlateauDetector(config)
    restored_detector.load_state_dict(detector.state_dict())
    restored_selector = Phase2CheckpointSelector(config.selection_window)
    restored_selector.load_state_dict(selector.state_dict())
    assert restored_detector.update(result(4)) == detector.update(result(4))
    assert restored_selector.best == selector.best


def phase3_config(**overrides):
    values = {
        "min_epochs": 1,
        "max_epochs": 10,
        "patience": 2,
        "safety_patience": 2,
        "min_delta": 0.01,
        "min_weak_quality_gain": 0.0,
        "min_weak_utility_gain": 0.0,
        "max_full_loss_increase": 0.05,
        "max_dominant_loss_increase": 0.05,
        "hard_max_full_loss_increase": 0.2,
        "hard_max_dominant_loss_increase": 0.2,
    }
    values.update(overrides)
    return Phase3StopConfig(**values)


def test_phase3_rejects_weak_gain_that_harms_full_model():
    stopper = Phase3InterventionStopper(phase3_config(), result(0))
    decision = stopper.update(
        result(1, full=1.1, dominant=1.2, weak=1.0), "harm_full"
    )
    assert not decision.current.is_feasible
    assert not decision.current.is_safe


def test_phase3_rejects_dominant_compatibility_drift():
    stopper = Phase3InterventionStopper(phase3_config(), result(0))
    decision = stopper.update(
        result(1, full=1.0, dominant=1.3, weak=1.0), "harm_dominant"
    )
    assert not decision.current.is_feasible
    assert decision.current.dominant_loss_increase == pytest.approx(0.1)


def test_phase3_patience_resets_only_above_min_delta():
    stopper = Phase3InterventionStopper(phase3_config(), result(0))
    first = stopper.update(
        result(1, full=0.95, dominant=1.2, weak=1.3), "first"
    )
    assert first.current.is_feasible
    second = stopper.update(
        result(2, full=0.945, dominant=1.2, weak=1.3), "tiny"
    )
    assert second.bad_checks == 1
    third = stopper.update(
        result(3, full=0.944, dominant=1.2, weak=1.3), "last"
    )
    assert third.should_stop
    assert third.stop_reason == "patience"
    assert third.selected.checkpoint_path == "first"


def test_phase3_hard_safety_stop_uses_consecutive_evaluations():
    stopper = Phase3InterventionStopper(phase3_config(), result(0))
    first = stopper.update(
        result(1, full=1.25, dominant=1.2), "unsafe1"
    )
    assert not first.should_stop
    second = stopper.update(
        result(2, full=1.25, dominant=1.2), "unsafe2"
    )
    assert second.should_stop
    assert second.stop_reason == "hard_safety"


def test_phase3_selects_best_feasible_not_last_checkpoint():
    stopper = Phase3InterventionStopper(
        phase3_config(patience=5), result(0)
    )
    stopper.update(
        result(1, full=0.9, dominant=1.2, weak=1.3), "best"
    )
    stopper.update(
        result(2, full=0.95, dominant=1.2, weak=1.4), "last"
    )
    selected, status = stopper.selection()
    assert selected.checkpoint_path == "best"
    assert status == "best_feasible"


def test_phase3_fallback_selects_safe_checkpoint_without_feasible_gain():
    stopper = Phase3InterventionStopper(
        phase3_config(
            max_epochs=2,
            min_weak_quality_gain=1.0,
            min_weak_utility_gain=1.0,
        ),
        result(0),
    )
    stopper.update(result(1), "safe1")
    final = stopper.update(result(2, full=0.99), "safe2")
    assert final.should_stop
    assert final.selection_status == "best_safe"
    assert final.selected is not None


def test_phase3_rolls_back_when_no_safe_checkpoint_exists():
    stopper = Phase3InterventionStopper(
        phase3_config(max_epochs=1), result(0)
    )
    final = stopper.update(
        result(1, full=1.1, dominant=1.3), "unsafe"
    )
    assert final.should_stop
    assert final.selection_status == "rollback_pre_phase3"
    assert final.selected is None


def test_phase3_state_round_trip_preserves_selection_and_counters():
    config = phase3_config(patience=4)
    baseline = result(0)
    stopper = Phase3InterventionStopper(config, baseline)
    stopper.update(result(1, full=0.9, weak=1.3), "one")
    stopper.update(result(2, full=0.91, weak=1.3), "two")
    restored = Phase3InterventionStopper(config, baseline)
    restored.load_state_dict(stopper.state_dict())
    assert restored.selection() == stopper.selection()
    assert restored.bad_checks == stopper.bad_checks


def test_phase4_selects_loss_and_accuracy_in_both_budgets():
    selector = Phase4CheckpointSelector(200, 198)
    selector.add(
        Phase4CheckpointRecord(result(0, full=1.0, accuracy=0.6), "zero")
    )
    selector.add(
        Phase4CheckpointRecord(result(1, full=0.9, accuracy=0.9), "one")
    )
    selector.add(
        Phase4CheckpointRecord(result(2, full=0.8, accuracy=0.7), "two")
    )
    selector.add(
        Phase4CheckpointRecord(result(3, full=0.1, accuracy=0.8), "three")
    )
    assert selector.best_full.checkpoint_path == "three"
    assert selector.best_budget_matched.checkpoint_path == "two"
    assert selector.best_full_accuracy.checkpoint_path == "one"
    assert selector.best_budget_matched_accuracy.checkpoint_path == "one"
    assert selector.best_full_for("loss") == selector.best_full
    assert selector.best_full_for("accuracy") == selector.best_full_accuracy


def test_phase4_dual_selector_state_round_trip_and_legacy_fallback():
    selector = Phase4CheckpointSelector(20, 5)
    selector.add(
        Phase4CheckpointRecord(result(2, full=0.5, accuracy=0.7), "loss")
    )
    selector.add(
        Phase4CheckpointRecord(
            result(4, full=0.8, accuracy=0.9), "accuracy"
        )
    )
    restored = Phase4CheckpointSelector(20, 5)
    restored.load_state_dict(selector.state_dict())
    assert restored.best_full == selector.best_full
    assert restored.best_full_accuracy == selector.best_full_accuracy

    legacy_state = selector.state_dict()
    legacy_state.pop("best_full_accuracy")
    legacy_state.pop("best_budget_matched_accuracy")
    legacy = Phase4CheckpointSelector(20, 5)
    legacy.load_state_dict(legacy_state)
    assert legacy.best_full_accuracy == legacy.best_full
    assert legacy.best_budget_matched_accuracy == legacy.best_budget_matched


class TinyPairedDataset(Dataset):
    def __init__(self):
        self.left = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
        )
        self.right = torch.flip(self.left, dims=(1,))
        self.targets = torch.tensor([0, 1, 0, 1])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (self.left[index], self.right[index]), self.targets[index]


class TinyBimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left_branch = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2)
        )
        self.right_branch = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2)
        )
        self.main_branch = torch.nn.Linear(2, 2)
        self.calls = []

    def forward(
        self,
        left,
        right,
        *,
        enable_left_branch=True,
        enable_right_branch=True,
        left_branch_intervention=None,
        right_branch_intervention=None,
    ):
        self.calls.append(
            (
                enable_left_branch,
                enable_right_branch,
                left_branch_intervention,
                right_branch_intervention,
            )
        )
        left_features = (
            self.left_branch(left)
            if enable_left_branch
            else torch.zeros_like(left)
        )
        right_features = (
            self.right_branch(right)
            if enable_right_branch
            else torch.zeros_like(right)
        )
        return self.main_branch(left_features + right_features)


def test_modality_evaluation_is_non_invasive_and_uses_all_modes():
    model = TinyBimodalModel()
    model.train()
    model.right_branch.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TinyPairedDataset(), batch_size=2, shuffle=False)
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    modes = {name: module.training for name, module in model.named_modules()}

    metrics = evaluate_modalities(
        model,
        criterion,
        loader,
        torch.device("cpu"),
        intervention_mode=WEAK_ONLY_MODE,
        phase_epoch=3,
        global_epoch=10,
        global_step=20,
    )

    assert metrics.weak_utility_loss == pytest.approx(
        metrics.dominant_only.loss - metrics.full.loss
    )
    assert len(model.calls) == 3 * len(loader)
    assert sum(metrics.per_example_losses.full) / 4 == pytest.approx(
        metrics.full.loss
    )
    assert sum(metrics.per_example_correctness.full) / 4 == pytest.approx(
        metrics.full.accuracy
    )
    assert metrics.per_example_correctness.intervention == (
        metrics.per_example_correctness.weak_only
    )
    for mode_metrics in (
        metrics.full,
        metrics.dominant_only,
        metrics.weak_only,
        metrics.intervention,
    ):
        assert mode_metrics.nll is not None
        assert mode_metrics.brier is not None
        assert 0.0 <= mode_metrics.ece <= 1.0
        assert 0.0 <= mode_metrics.mean_confidence <= 1.0
        assert 0.0 <= mode_metrics.mean_incorrect_confidence <= 1.0
    assert metrics.full.nll == pytest.approx(metrics.full.loss)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name])
        assert parameter.grad is None
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name])
    assert {
        name: module.training for name, module in model.named_modules()
    } == modes
    call_modes = {(left, right) for left, right, _, _ in model.calls}
    assert call_modes == {(True, True), (True, False), (False, True)}


def test_phase3_gradient_diagnostics_are_non_invasive():
    model = TinyBimodalModel()
    model.train()
    model.right_branch.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TinyPairedDataset(), batch_size=2, shuffle=False)
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    modes = {name: module.training for name, module in model.named_modules()}
    gradients = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.ones_like(parameter)
        gradients[name] = parameter.grad.detach().clone()

    measured = evaluate_phase3_gradient_diagnostics(
        model,
        criterion,
        loader,
        torch.device("cpu"),
        max_batches=1,
    )

    assert measured["sample_count"] == 2
    assert measured["batch_count"] == 1
    assert measured["shared_parameter_count"] > 0
    assert measured["weak_parameter_count"] > 0
    assert measured["weak_branch_norm_per_sqrt_parameter"] > 0
    assert -1.0 <= measured["shared_cosine_weak_dominant"] <= 1.0
    assert -1.0 <= measured["shared_cosine_weak_full"] <= 1.0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name])
        torch.testing.assert_close(parameter.grad, gradients[name])
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers[name])
    assert {
        name: module.training for name, module in model.named_modules()
    } == modes


def test_phase2_posthoc_records_survive_phase_state_resume():
    source = ValidationControlledTrainer.__new__(ValidationControlledTrainer)
    source._phase2_posthoc_records = {
        "best_loss": Phase2CheckpointRecord(result(2), "loss")
    }
    state = {"phase_state": source._phase_state(selector={"value": 1})}

    restored = ValidationControlledTrainer.__new__(
        ValidationControlledTrainer
    )
    restored._phase2_posthoc_records = {}
    restored._restore_phase2_posthoc_records(state)

    assert restored._phase2_posthoc_records == source._phase2_posthoc_records


def test_unimodal_reference_metadata_v2_tracks_source_bimodal_hash():
    source_hash = "a" * 64
    config = SimpleNamespace(
        protocol_manifest={
            "model": {"name": "mm_resnet"},
            "dataset": {
                "name": "mm_cifar10",
                "split_profile": "split-v1",
                "normalization_profile": "norm-v1",
                "split": {"indices_sha256": "split"},
                "normalization": {"proper_left": [0.1, 0.2]},
            },
            "training": {
                "seed": 83,
                "source_bimodal_initial_state_sha256": source_hash,
            },
        }
    )
    record = SimpleNamespace(metrics=ModeMetrics(0.7, 0.8), epoch=25)

    metadata = ValidationControlledTrainer._unimodal_reference_metadata(
        config, "left_proper", record
    )

    assert metadata["version"] == 2
    assert (
        metadata["initialization_policy"]
        == "canonical_bimodal_components_v2"
    )
    assert metadata["source_bimodal_initial_state_sha256"] == source_hash


def test_phase2_publication_binary_rejects_diagnostic_duration():
    trainer = ValidationControlledTrainer.__new__(ValidationControlledTrainer)
    with pytest.raises(
        ValueError,
        match="publication_binary phase2 must be configured as 0 or 200",
    ):
        trainer._run_phase2(
            {
                "phase2": 50,
                "phase2_stopping": {"mode": "disabled"},
            }
        )


def test_phase2_diagnostic_fixed_accepts_nonpublication_duration():
    trainer = ValidationControlledTrainer.__new__(ValidationControlledTrainer)

    def accepted(_phase, _config):
        raise RuntimeError("duration accepted")

    trainer._prepare_phase = accepted
    with pytest.raises(RuntimeError, match="duration accepted"):
        trainer._run_phase2(
            {
                "phase2": 50,
                "phase2_stopping": {
                    "mode": "disabled",
                    "duration_policy": "diagnostic_fixed",
                },
            }
        )


def test_phase2_rejects_unknown_duration_policy_even_when_skipped():
    trainer = ValidationControlledTrainer.__new__(ValidationControlledTrainer)
    with pytest.raises(ValueError, match="unsupported phase2 duration policy"):
        trainer._run_phase2(
            {
                "phase2": 0,
                "phase2_stopping": {
                    "mode": "disabled",
                    "duration_policy": "unknown",
                },
            }
        )


def test_phase2_test_is_posthoc_deduplicated_and_restores_final_checkpoint():
    trainer = ValidationControlledTrainer.__new__(ValidationControlledTrainer)
    trainer._phase2_posthoc_records = {
        "best_loss": Phase2CheckpointRecord(result(2), "phase2_loss"),
        "best_accuracy": Phase2CheckpointRecord(
            result(4, accuracy=0.7), "phase2_accuracy"
        ),
    }
    trainer._final_selected_checkpoint_path = "final_phase4"
    trainer.global_step = 99
    loads = []
    logs = []
    summaries = []
    factory_calls = []
    trainer.final_test_loader_factory = lambda: (
        factory_calls.append(True)
        or {"test_proper": "proper", "test_blurred": "blurred"}
    )
    trainer._batch_limit = lambda _name: None
    trainer._load_selected = lambda path: loads.append(path)
    trainer._evaluate_full_loader = lambda loader, max_batches=None: (
        ModeMetrics(0.4, 0.8)
        if loader == "proper"
        else ModeMetrics(0.7, 0.6)
    )
    trainer.logger = SimpleNamespace(
        log_scalars=lambda values, step: logs.append((values, step))
    )
    trainer._write_phase_summary = summaries.append

    trainer._run_phase2_posthoc_test(
        {"phase2_test_policy": "posthoc_final"}
    )

    assert factory_calls == [True]
    assert loads == ["phase2_loss", "phase2_accuracy", "final_phase4"]
    assert len(logs) == 2
    assert all(step == 99 for _values, step in logs)
    assert summaries[0]["status"] == "completed"
    assert set(summaries[0]["checkpoints"]) == {
        "best_loss",
        "best_accuracy",
    }
    assert all(
        key.startswith("posthoc_test/phase2/")
        for values, _step in logs
        for key in values
    )


def test_test_loader_is_not_part_of_stopper_or_evaluator_interfaces():
    assert "test" not in inspect.signature(
        Phase3InterventionStopper.update
    ).parameters
    assert "test" not in inspect.signature(evaluate_modalities).parameters



def test_validation_trainer_phase4_only_uses_validation_controller():
    trainer = object.__new__(ValidationControlledTrainer)
    trainer.loaders = {"train": [1, 2, 3]}
    trainer.logger = SimpleNamespace(close=lambda: None)
    trainer._initialize_run = lambda config: None
    calls = []
    trainer._run_phase4 = lambda config, e3: calls.append(e3)
    config = SimpleNamespace(exp_starts_at_epoch=280)
    config.get = lambda name, default=None: {"phase3": 60}.get(name, default)

    trainer.run_phase(4, config)

    assert calls == [60]
    assert trainer.executed_global_epoch == 280
    assert trainer.global_step == 0


def test_validation_trainer_preserves_actual_update_counter_per_epoch():
    trainer = object.__new__(ValidationControlledTrainer)
    trainer.global_step = 17
    trainer.executed_global_epoch = 280
    trainer.model = SimpleNamespace(train=lambda: None)
    trainer.criterion = SimpleNamespace(train=lambda: None)
    observed = []

    def run_epoch(phase, config):
        observed.append((phase, trainer.resume_global_step))
        trainer.global_step = trainer.resume_global_step + 4
        return {}

    trainer.run_epoch = run_epoch
    trainer._train_one_epoch(SimpleNamespace(), phase_epoch=1)

    assert observed == [("train", 17)]
    assert trainer.global_step == 21
    assert trainer.executed_global_epoch == 281

def test_validation_trainer_reuses_loaded_optimizer_inside_resumed_phase():
    trainer = object.__new__(ValidationControlledTrainer)
    trainer.resume_training_state = {
        "is_training_checkpoint": True,
        "metadata": {"phase": 2},
        "next_epoch": 17,
        "global_step": 101,
        "phase_epoch": 6,
    }
    trainer._apply_stage = lambda config, stage: None
    trainer._start_phase = lambda phase, config: pytest.fail(
        "resume inside a phase must not reset the optimizer"
    )

    phase_epoch, state = trainer._prepare_phase(2, SimpleNamespace())

    assert phase_epoch == 6
    assert state is trainer.resume_training_state
    assert trainer.executed_global_epoch == 17
    assert trainer.global_step == 101


def test_validation_trainer_builds_fresh_optimizer_on_phase_boundaries():
    trainer = object.__new__(ValidationControlledTrainer)
    created = []

    def factory(phase):
        pair = (object(), object())
        created.append((phase, pair))
        return pair

    trainer.optimizer_factory = factory
    trainer.extra_modules = {}
    trainer._reset_optimizer(1, SimpleNamespace())
    first = (trainer.optim, trainer.lr_scheduler)
    trainer._reset_optimizer(2, SimpleNamespace())
    second = (trainer.optim, trainer.lr_scheduler)

    assert [phase for phase, _pair in created] == [1, 2]
    assert first != second


def test_validation_trainer_resume_skips_earlier_completed_phases():
    trainer = object.__new__(ValidationControlledTrainer)
    trainer.resume_training_state = {
        "is_training_checkpoint": True,
        "metadata": {"phase": 3},
        "next_epoch": 30,
        "global_step": 400,
        "phase_epoch": 4,
    }
    trainer.logger = None
    trainer._initialize_run = lambda config: None
    trainer._run_phase1 = lambda config: pytest.fail("phase 1 repeated")
    trainer._run_phase2 = lambda config: pytest.fail("phase 2 repeated")
    trainer._run_phase3 = lambda config: 4
    calls = []
    trainer._run_phase4 = lambda config, e3: calls.append(e3)

    trainer.run_all_at_once(SimpleNamespace())

    assert calls == [4]
    assert trainer.executed_global_epoch == 30
    assert trainer.global_step == 400


def test_phase_summary_metrics_exclude_per_example_payload():
    metrics = paired_result(1, full=0.8, dominant=1.0, weak=1.2)
    summary = ValidationControlledTrainer._summary_metrics(metrics)
    assert summary["full_loss"] == pytest.approx(0.8)
    assert summary["weak_utility_loss"] == pytest.approx(0.2)
    assert "per_example_losses" not in summary
    assert "per_example_correctness" not in summary


def test_checkpoint_metrics_are_available_for_boundary_resume():
    expected = result(8, full=0.7)
    state = {"metadata": {"metrics": expected.state_dict()}}
    assert ValidationControlledTrainer._checkpoint_metrics(state) == expected
    assert ValidationControlledTrainer._checkpoint_metrics(None) is None


def test_phase4_selector_rejects_candidate_beyond_budget():
    selector = Phase4CheckpointSelector(3, 1)
    with pytest.raises(ValueError, match="exceeds"):
        selector.add(Phase4CheckpointRecord(result(4), "too_late"))


def paired_result(epoch, *, full, dominant, weak, sample_count=32):
    full_losses = tuple([float(full)] * sample_count)
    dominant_losses = tuple([float(dominant)] * sample_count)
    weak_losses = tuple([float(weak)] * sample_count)
    return ModalityEvaluationResult(
        full=ModeMetrics(float(full), 0.7),
        dominant_only=ModeMetrics(float(dominant), 0.65),
        weak_only=ModeMetrics(float(weak), 0.55),
        intervention=ModeMetrics(float(weak), 0.55),
        phase_epoch=epoch,
        global_epoch=epoch,
        global_step=epoch * 10,
        per_example_losses=PerExampleModeLosses(
            full=full_losses,
            dominant_only=dominant_losses,
            weak_only=weak_losses,
            intervention=weak_losses,
        ),
    )


def adaptive_phase3_config(**overrides):
    values = {
        "min_epochs": 1,
        "max_epochs": 20,
        "patience": 8,
        "safety_patience": 3,
        "min_delta": 0.0,
        "min_weak_quality_gain": 0.0,
        "min_weak_utility_gain": 0.0,
        "max_full_loss_increase": 0.1,
        "max_dominant_loss_increase": 0.1,
        "hard_max_full_loss_increase": 0.5,
        "hard_max_dominant_loss_increase": 0.5,
        "adaptive_rule": True,
        "confidence_level": 0.95,
        "confidence_family_size": 8,
        "max_looks": 10,
        "minimum_exposure_evaluations": 1,
        "reversal_patience": 2,
        "futility_patience": 2,
        "trend_window": 2,
        "futility_prediction_horizon_epochs": 2,
    }
    values.update(overrides)
    return Phase3StopConfig(**values)


def test_adaptive_phase3_stops_after_confirmed_trend_reversal():
    baseline = paired_result(0, full=1.0, dominant=1.2, weak=1.5)
    stopper = Phase3InterventionStopper(
        adaptive_phase3_config(), baseline
    )
    best = stopper.update(
        paired_result(1, full=1.0, dominant=1.24, weak=1.3), "best"
    )
    assert best.current.is_feasible
    assert best.current.paired_estimates["weak_utility_gain"]["lower"] > 0

    decline = paired_result(2, full=1.02, dominant=1.2, weak=1.35)
    assert not stopper.update(decline, "decline_1").should_stop
    decision = stopper.update(
        paired_result(3, full=1.02, dominant=1.2, weak=1.35),
        "decline_2",
    )
    assert decision.should_stop
    assert decision.stop_reason == "trend_reversal"
    assert decision.selected.checkpoint_path == "best"


def test_adaptive_phase3_futility_stops_without_plausible_gain():
    baseline = paired_result(0, full=1.0, dominant=1.2, weak=1.5)
    stopper = Phase3InterventionStopper(
        adaptive_phase3_config(), baseline
    )
    futile = paired_result(1, full=1.0, dominant=1.15, weak=1.55)
    assert not stopper.update(futile, "futile_1").should_stop
    second = stopper.update(
        paired_result(2, full=1.0, dominant=1.15, weak=1.55),
        "futile_2",
    )
    assert not second.should_stop
    decision = stopper.update(
        paired_result(3, full=1.0, dominant=1.15, weak=1.55),
        "futile_3",
    )
    assert decision.should_stop
    assert decision.stop_reason == "futility"
    assert decision.selection_status == "best_safe"


def test_adaptive_phase3_futility_waits_for_plausible_positive_trend():
    baseline = paired_result(0, full=1.0, dominant=1.2, weak=1.5)
    stopper = Phase3InterventionStopper(
        adaptive_phase3_config(
            futility_patience=1,
            futility_prediction_horizon_epochs=10,
        ),
        baseline,
    )
    stopper.update(
        paired_result(1, full=1.0, dominant=1.1, weak=1.6),
        "slow_1",
    )
    decision = stopper.update(
        paired_result(2, full=1.0, dominant=1.15, weak=1.55),
        "slow_2",
    )
    assert not decision.should_stop
    assert stopper.last_optimistic_bounds["weak_utility_gain"] > 0.0
    assert stopper.last_optimistic_bounds["weak_quality_gain"] > 0.0


def test_adaptive_phase3_futility_stops_when_quality_plateaus_without_utility():
    baseline = paired_result(0, full=1.0, dominant=1.2, weak=1.5)
    stopper = Phase3InterventionStopper(
        adaptive_phase3_config(futility_patience=1), baseline
    )
    stopper.update(
        paired_result(1, full=1.0, dominant=1.15, weak=1.3),
        "quality_only_1",
    )
    decision = stopper.update(
        paired_result(2, full=1.0, dominant=1.15, weak=1.3),
        "quality_only_2",
    )
    assert decision.should_stop
    assert decision.stop_reason == "futility"


def test_adaptive_phase3_state_round_trip_preserves_local_counters():
    baseline = paired_result(0, full=1.0, dominant=1.2, weak=1.5)
    config = adaptive_phase3_config(reversal_patience=3)
    stopper = Phase3InterventionStopper(config, baseline)
    stopper.update(
        paired_result(1, full=1.0, dominant=1.24, weak=1.3), "best"
    )
    stopper.update(
        paired_result(2, full=1.02, dominant=1.2, weak=1.35), "decline"
    )
    restored = Phase3InterventionStopper(config, baseline)
    restored.load_state_dict(stopper.state_dict())
    assert restored.reversal_bad_checks == 1
    assert restored.evaluation_count == 2
    assert len(restored.history) == 2
    decision = restored.update(
        paired_result(3, full=1.02, dominant=1.2, weak=1.35),
        "decline_2",
    )
    assert not decision.should_stop


def recovery_result(
    epoch,
    *,
    weak_loss,
    weak_correct,
    full_loss=1.0,
    dominant_loss=1.2,
    full_correct=70,
    dominant_correct=65,
    sample_count=100,
):
    def correctness(count):
        return tuple([1] * count + [0] * (sample_count - count))

    full_correct = correctness(full_correct)
    dominant_correct = correctness(dominant_correct)
    weak_values = correctness(weak_correct)
    full_losses = tuple([float(full_loss)] * sample_count)
    dominant_losses = tuple([float(dominant_loss)] * sample_count)
    weak_losses = tuple([float(weak_loss)] * sample_count)
    return ModalityEvaluationResult(
        full=ModeMetrics(
            float(full_loss), sum(full_correct) / sample_count
        ),
        dominant_only=ModeMetrics(
            float(dominant_loss), sum(dominant_correct) / sample_count
        ),
        weak_only=ModeMetrics(float(weak_loss), weak_correct / sample_count),
        intervention=ModeMetrics(float(weak_loss), weak_correct / sample_count),
        phase_epoch=epoch,
        global_epoch=epoch,
        global_step=epoch * 10,
        per_example_losses=PerExampleModeLosses(
            full=full_losses,
            dominant_only=dominant_losses,
            weak_only=weak_losses,
            intervention=weak_losses,
        ),
        per_example_correctness=PerExampleModeCorrectness(
            full=full_correct,
            dominant_only=dominant_correct,
            weak_only=weak_values,
            intervention=weak_values,
        ),
    )


def recovery_config(**overrides):
    values = {
        "min_epochs": 1,
        "max_epochs": 20,
        "safety_patience": 1,
        "min_delta": 0.001,
        "adaptive_rule": True,
        "confidence_level": 0.95,
        "confidence_family_size": 8,
        "max_looks": 10,
        "minimum_exposure_evaluations": 2,
        "reversal_patience": 2,
        "futility_patience": 2,
        "trend_window": 2,
        "decision_rule": "weak_recovery",
        "emergency_stop_mode": "numerical_only",
        "min_weak_quality_gain": 0.0,
        "min_weak_accuracy_gain": 0.0,
        "recovery_primary_metric": "accuracy",
        "max_weak_accuracy_slope": 0.0005,
        "max_weak_quality_slope": 0.001,
        "plateau_patience": 2,
    }
    values.update(overrides)
    return Phase3StopConfig(**values)


def local_accuracy_config(**overrides):
    values = {
        **recovery_config().__dict__,
        "decision_rule": "local_accuracy",
        "minimum_exposure_evaluations": 2,
        "trend_window": 2,
        "target_patience": 2,
        "pareto_patience": 2,
        "futility_harm_patience": 2,
        "max_weak_accuracy_slope": 0.0,
    }
    values.update(overrides)
    return Phase3StopConfig(**values)


def test_phase3_evaluation_schedule_has_dense_four_epoch_prefix():
    observed = [
        epoch
        for epoch in range(1, 31)
        if should_evaluate_phase_epoch(epoch, 30, 4, 4)
    ]
    assert observed == [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 30]


def test_phase3_evaluation_schedule_preserves_legacy_cadence_by_default():
    observed = [
        epoch
        for epoch in range(1, 13)
        if should_evaluate_phase_epoch(epoch, 12, 5)
    ]
    assert observed == [5, 10, 12]


def test_local_accuracy_stops_when_weak_reliably_reaches_dominant():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3LocalAccuracyStopper(
        local_accuracy_config(), baseline
    )
    assert not stopper.update(
        recovery_result(
            1,
            weak_loss=1.2,
            weak_correct=75,
            full_correct=75,
            dominant_correct=65,
        ),
        "one",
    ).should_stop
    assert not stopper.update(
        recovery_result(
            2,
            weak_loss=1.1,
            weak_correct=85,
            full_correct=80,
            dominant_correct=60,
        ),
        "two",
    ).should_stop
    decision = stopper.update(
        recovery_result(
            3,
            weak_loss=1.0,
            weak_correct=88,
            full_correct=82,
            dominant_correct=62,
        ),
        "three",
    )
    assert decision.should_stop
    assert decision.stop_reason == "target_reached"
    assert decision.selected.checkpoint_path == "three"


def test_local_accuracy_stops_after_confirmed_pareto_reversal():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3LocalAccuracyStopper(
        local_accuracy_config(), baseline
    )
    stopper.update(
        recovery_result(
            1,
            weak_loss=1.0,
            weak_correct=80,
            full_correct=80,
            dominant_correct=75,
        ),
        "pareto_best",
    )
    assert not stopper.update(
        recovery_result(
            2,
            weak_loss=1.1,
            weak_correct=60,
            full_correct=60,
            dominant_correct=55,
        ),
        "dominated_1",
    ).should_stop
    decision = stopper.update(
        recovery_result(
            3,
            weak_loss=1.2,
            weak_correct=58,
            full_correct=58,
            dominant_correct=53,
        ),
        "dominated_2",
    )
    assert decision.should_stop
    assert decision.stop_reason == "pareto_reversal"
    assert decision.selected.checkpoint_path == "pareto_best"


def test_local_accuracy_futility_requires_weak_plateau_and_harm():
    baseline = recovery_result(
        0,
        weak_loss=2.0,
        weak_correct=200,
        full_correct=700,
        dominant_correct=650,
        sample_count=1000,
    )
    stopper = Phase3LocalAccuracyStopper(
        local_accuracy_config(max_weak_accuracy_slope=1.0), baseline
    )
    stopper.update(
        recovery_result(
            1,
            weak_loss=1.2,
            weak_correct=600,
            full_correct=850,
            dominant_correct=800,
            sample_count=1000,
        ),
        "anchor",
    )
    assert not stopper.update(
        recovery_result(
            2,
            weak_loss=1.1,
            weak_correct=610,
            full_correct=450,
            dominant_correct=400,
            sample_count=1000,
        ),
        "harm_1",
    ).should_stop
    decision = stopper.update(
        recovery_result(
            3,
            weak_loss=1.0,
            weak_correct=620,
            full_correct=350,
            dominant_correct=300,
            sample_count=1000,
        ),
        "harm_2",
    )
    assert decision.should_stop
    assert decision.stop_reason == "futility_with_harm"
    assert decision.selected.checkpoint_path == "anchor"


def test_local_accuracy_gradient_conflict_confirms_futility():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3LocalAccuracyStopper(
        local_accuracy_config(max_weak_accuracy_slope=1.0), baseline
    )
    stable = dict(
        weak_loss=1.0,
        weak_correct=60,
        full_correct=75,
        dominant_correct=70,
    )
    stopper.update(recovery_result(1, **stable), "anchor")
    diagnostics = {
        "shared_cosine_weak_dominant": -0.5,
        "shared_cosine_weak_full": 0.2,
    }
    assert not stopper.update(
        recovery_result(2, **stable),
        "conflict_1",
        diagnostics=diagnostics,
    ).should_stop
    decision = stopper.update(
        recovery_result(3, **stable),
        "conflict_2",
    )
    assert decision.should_stop
    assert decision.stop_reason == "futility_with_harm"
    assert decision.selected.checkpoint_path == "anchor"


def test_local_accuracy_state_round_trip_preserves_local_frontier():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    config = local_accuracy_config(pareto_patience=3)
    stopper = Phase3LocalAccuracyStopper(config, baseline)
    stopper.update(
        recovery_result(
            1,
            weak_loss=1.0,
            weak_correct=80,
            full_correct=80,
            dominant_correct=75,
        ),
        "best",
    )
    stopper.update(
        recovery_result(
            2,
            weak_loss=1.1,
            weak_correct=60,
            full_correct=60,
            dominant_correct=55,
        ),
        "dominated",
    )
    restored = Phase3LocalAccuracyStopper(config, baseline)
    restored.load_state_dict(stopper.state_dict())
    assert restored.pareto_bad_checks == 1
    assert restored.retained_checkpoint_paths == {"best"}


def test_recovery_rule_requires_accuracy_compatibility():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(recovery_config(), baseline)
    decision = stopper.update(
        recovery_result(
            1,
            weak_loss=1.0,
            weak_correct=70,
            full_loss=2.0,
            dominant_loss=2.0,
            full_correct=50,
            dominant_correct=45,
        ),
        "recovered_but_incompatible",
    )
    assert not decision.current.is_feasible
    assert not decision.current.is_safe
    assert decision.current.weak_accuracy_gain == pytest.approx(0.5)
    assert stopper.best_feasible is None
    assert stopper.best_safe is None


def test_recovery_rule_stops_on_confirmed_accuracy_compatibility_breach():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(
        recovery_config(
            safety_patience=2,
            hard_max_full_accuracy_drop=0.10,
            hard_max_dominant_accuracy_drop=0.10,
        ),
        baseline,
    )
    first = stopper.update(
        recovery_result(
            1,
            weak_loss=1.0,
            weak_correct=70,
            full_correct=45,
            dominant_correct=40,
        ),
        "breach_1",
    )
    assert not first.should_stop
    second = stopper.update(
        recovery_result(
            2,
            weak_loss=0.9,
            weak_correct=72,
            full_correct=44,
            dominant_correct=39,
        ),
        "breach_2",
    )
    assert second.should_stop
    assert second.stop_reason == "compatibility_breach"


def test_accuracy_primary_recovery_plateau_ignores_rising_loss():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(recovery_config(), baseline)
    assert not stopper.update(
        recovery_result(1, weak_loss=1.0, weak_correct=70), "best_loss"
    ).should_stop
    assert not stopper.update(
        recovery_result(2, weak_loss=1.5, weak_correct=70), "rising_loss"
    ).should_stop
    decision = stopper.update(
        recovery_result(3, weak_loss=2.5, weak_correct=70), "higher_loss"
    )
    assert decision.should_stop
    assert decision.stop_reason == "recovery_plateau"
    assert decision.selected.checkpoint_path == "best_loss"


def test_recovery_rule_stops_on_validation_plateau():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(recovery_config(), baseline)
    assert not stopper.update(
        recovery_result(1, weak_loss=1.0, weak_correct=70), "one"
    ).should_stop
    assert not stopper.update(
        recovery_result(2, weak_loss=1.0, weak_correct=70), "two"
    ).should_stop
    decision = stopper.update(
        recovery_result(3, weak_loss=1.0, weak_correct=70), "three"
    )
    assert decision.should_stop
    assert decision.stop_reason == "recovery_plateau"
    assert decision.selected.checkpoint_path == "one"


def test_recovery_rule_uses_numerical_emergency_stop_only():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(recovery_config(), baseline)
    decision = stopper.update(
        recovery_result(1, weak_loss=float("inf"), weak_correct=20),
        "nonfinite",
    )
    assert decision.should_stop
    assert decision.stop_reason == "emergency_stop"


def test_recovery_shadow_continues_and_records_independent_triggers():
    baseline = recovery_result(0, weak_loss=2.0, weak_correct=20)
    stopper = Phase3RecoveryStopper(
        recovery_config(
            shadow_continue_after_stop=True,
            reversal_patience=1,
        ),
        baseline,
    )
    stopper.update(recovery_result(1, weak_loss=1.0, weak_correct=70), "best")
    stopper.update(recovery_result(2, weak_loss=1.0, weak_correct=70), "flat1")
    first = stopper.update(
        recovery_result(3, weak_loss=1.0, weak_correct=70), "flat2"
    )
    assert first.stop_reason == "recovery_plateau"
    later = stopper.update(
        recovery_result(4, weak_loss=1.5, weak_correct=40), "decline"
    )
    assert later.stop_reason == "recovery_plateau"
    assert stopper.evaluation_count == 4
    assert stopper.first_trigger_epochs["recovery_plateau"] == 3
    assert stopper.first_trigger_epochs["trend_reversal"] == 4

    restored = Phase3RecoveryStopper(stopper.config, baseline)
    restored.load_state_dict(stopper.state_dict())
    assert restored.first_trigger_epochs == stopper.first_trigger_epochs
    assert restored.evaluation_count == 4
