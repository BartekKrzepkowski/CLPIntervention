"""Pure validation controllers for phases 2-4 of the CLP protocol."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from math import inf
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModeMetrics:
    loss: float
    accuracy: float
    nll: float | None = None
    brier: float | None = None
    ece: float | None = None
    mean_confidence: float | None = None
    mean_incorrect_confidence: float | None = None

    def state_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class UnimodalCheckpointRecord:
    modality: str
    metrics: ModeMetrics
    epoch: int
    checkpoint_path: str

    def __post_init__(self):
        if self.modality not in {"left_proper", "right_proper"}:
            raise ValueError("unknown unimodal reference modality")
        if self.epoch < 0:
            raise ValueError("unimodal checkpoint epoch must be non-negative")

    def state_dict(self):
        return {
            "modality": self.modality,
            "metrics": self.metrics.state_dict(),
            "epoch": self.epoch,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            modality=str(state["modality"]),
            metrics=ModeMetrics(**state["metrics"]),
            epoch=int(state["epoch"]),
            checkpoint_path=str(state["checkpoint_path"]),
        )


class UnimodalCheckpointSelector:
    """Select maximum validation accuracy, then loss and earlier epoch."""

    def __init__(self, modality):
        if modality not in {"left_proper", "right_proper"}:
            raise ValueError("unknown unimodal reference modality")
        self.modality = modality
        self.best = None

    @staticmethod
    def _better(candidate, incumbent):
        if incumbent is None:
            return True
        if candidate.metrics.accuracy != incumbent.metrics.accuracy:
            return candidate.metrics.accuracy > incumbent.metrics.accuracy
        if candidate.metrics.loss != incumbent.metrics.loss:
            return candidate.metrics.loss < incumbent.metrics.loss
        return candidate.epoch < incumbent.epoch

    def update(self, metrics, epoch, checkpoint_path):
        record = UnimodalCheckpointRecord(
            modality=self.modality,
            metrics=metrics,
            epoch=int(epoch),
            checkpoint_path=str(checkpoint_path),
        )
        if self._better(record, self.best):
            self.best = record
            return True, record
        return False, record

    def state_dict(self):
        return {
            "modality": self.modality,
            "best": self.best.state_dict() if self.best is not None else None,
        }

    @property
    def retained_checkpoint_paths(self):
        """Paths required to preserve the current validation selection."""
        if self.best is None:
            return set()
        return {self.best.checkpoint_path}

    def load_state_dict(self, state):
        if str(state["modality"]) != self.modality:
            raise ValueError("unimodal modality changed across resume")
        self.best = (
            UnimodalCheckpointRecord.from_state_dict(state["best"])
            if state.get("best") is not None
            else None
        )


@dataclass(frozen=True)
class PerExampleModeLosses:
    full: tuple[float, ...]
    dominant_only: tuple[float, ...]
    weak_only: tuple[float, ...]
    intervention: tuple[float, ...]

    def __post_init__(self):
        sizes = {
            len(self.full),
            len(self.dominant_only),
            len(self.weak_only),
            len(self.intervention),
        }
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError(
                "per-example mode losses must be aligned and non-empty"
            )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            **{
                key: tuple(float(value) for value in state[key])
                for key in (
                    "full",
                    "dominant_only",
                    "weak_only",
                    "intervention",
                )
            }
        )


@dataclass(frozen=True)
class PerExampleModeCorrectness:
    full: tuple[int, ...]
    dominant_only: tuple[int, ...]
    weak_only: tuple[int, ...]
    intervention: tuple[int, ...]

    def __post_init__(self):
        fields = (
            self.full,
            self.dominant_only,
            self.weak_only,
            self.intervention,
        )
        sizes = {len(values) for values in fields}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError(
                "per-example mode correctness must be aligned and non-empty"
            )
        if any(value not in (0, 1) for values in fields for value in values):
            raise ValueError("per-example correctness values must be binary")

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            **{
                key: tuple(int(value) for value in state[key])
                for key in (
                    "full",
                    "dominant_only",
                    "weak_only",
                    "intervention",
                )
            }
        )


@dataclass(frozen=True)
class ModalityEvaluationResult:
    full: ModeMetrics
    dominant_only: ModeMetrics
    weak_only: ModeMetrics
    intervention: ModeMetrics
    phase_epoch: int
    global_epoch: int
    global_step: int
    per_example_losses: PerExampleModeLosses | None = None
    per_example_correctness: PerExampleModeCorrectness | None = None

    @property
    def weak_utility_loss(self):
        return self.dominant_only.loss - self.full.loss

    @property
    def weak_utility_accuracy(self):
        return self.full.accuracy - self.dominant_only.accuracy

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            full=ModeMetrics(**state["full"]),
            dominant_only=ModeMetrics(**state["dominant_only"]),
            weak_only=ModeMetrics(**state["weak_only"]),
            intervention=ModeMetrics(**state["intervention"]),
            phase_epoch=int(state["phase_epoch"]),
            global_epoch=int(state["global_epoch"]),
            global_step=int(state["global_step"]),
            per_example_losses=(
                PerExampleModeLosses.from_state_dict(
                    state["per_example_losses"]
                )
                if state.get("per_example_losses") is not None
                else None
            ),
            per_example_correctness=(
                PerExampleModeCorrectness.from_state_dict(
                    state["per_example_correctness"]
                )
                if state.get("per_example_correctness") is not None
                else None
            ),
        )


def phase3_trajectory_record(
    metrics,
    *,
    decision_rule,
    checkpoint_path,
    checkpoint_retained,
    current_record=None,
    decision=None,
    unimodal_references=None,
):
    """Build one versioned, W&B-independent Phase-3 replay record."""
    selected = getattr(decision, "selected", None)
    return {
        "version": 1,
        "decision_rule": str(decision_rule),
        "phase_epoch": int(metrics.phase_epoch),
        "global_epoch": int(metrics.global_epoch),
        "global_step": int(metrics.global_step),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_retained": bool(checkpoint_retained),
        "metrics": metrics.state_dict(),
        "controller": {
            "dominant_ratio": getattr(current_record, "dominant_ratio", None),
            "weak_ratio": getattr(current_record, "weak_ratio", None),
            "parity_gap": getattr(current_record, "parity_gap", None),
            "recovery_fraction": getattr(
                current_record, "recovery_fraction", None
            ),
            "recovery_fraction_threshold": getattr(
                current_record, "recovery_fraction_threshold", None
            ),
            "is_feasible": getattr(current_record, "is_feasible", None),
            "is_safe": getattr(current_record, "is_safe", None),
            "should_stop": getattr(decision, "should_stop", False),
            "stop_reason": getattr(decision, "stop_reason", None),
            "selection_status": getattr(decision, "selection_status", None),
            "selected_epoch": (
                int(selected.metrics.phase_epoch) if selected is not None else None
            ),
        },
        "unimodal_references": unimodal_references,
    }


@dataclass(frozen=True)
class Phase2PlateauConfig:
    min_epochs: int = 20
    min_delta_full_loss: float = 0.001
    full_loss_patience: int = 10
    stability_window: int = 8
    max_abs_slope_weak_loss: float = 0.001
    max_abs_slope_weak_utility: float = 0.001
    plateau_confirmations: int = 3
    selection_window: int = 8
    selection_scope: str = "final_window"
    primary_metric: str = "loss"

    def __post_init__(self):
        integer_fields = (
            self.min_epochs,
            self.full_loss_patience,
            self.stability_window,
            self.plateau_confirmations,
            self.selection_window,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("phase 2 count parameters must be positive")
        if any(
            value < 0
            for value in (
                self.min_delta_full_loss,
                self.max_abs_slope_weak_loss,
                self.max_abs_slope_weak_utility,
            )
        ):
            raise ValueError("phase 2 deltas and slope limits must be non-negative")
        if self.selection_scope not in {"final_window", "global"}:
            raise ValueError("unknown phase 2 selection_scope")
        if self.primary_metric not in {"loss", "accuracy"}:
            raise ValueError("unknown phase 2 primary_metric")


@dataclass(frozen=True)
class Phase2CheckpointRecord:
    metrics: ModalityEvaluationResult
    checkpoint_path: str

    def state_dict(self):
        return {
            "metrics": self.metrics.state_dict(),
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            metrics=ModalityEvaluationResult.from_state_dict(state["metrics"]),
            checkpoint_path=str(state["checkpoint_path"]),
        )


@dataclass(frozen=True)
class Phase2PlateauDecision:
    should_stop: bool
    stop_reason: str | None
    full_loss_bad_checks: int
    weak_loss_slope: float | None
    weak_utility_slope: float | None
    plateau_confirmations: int


def _linear_slope(points):
    if len(points) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 0.0
    return float(np.dot(centered, y - y.mean()) / denominator)


class Phase2PlateauDetector:
    def __init__(self, config: Phase2PlateauConfig):
        self.config = config
        self.best_full_loss = inf
        self.full_loss_bad_checks = 0
        self.history = []
        self.plateau_confirmations = 0
        self.stop_epoch = None

    def update(self, metrics: ModalityEvaluationResult):
        if self.stop_epoch is not None:
            return self._decision(True, "plateau_detected")
        if metrics.full.loss <= (
            self.best_full_loss - self.config.min_delta_full_loss
        ):
            self.best_full_loss = metrics.full.loss
            self.full_loss_bad_checks = 0
        else:
            self.full_loss_bad_checks += 1
        self.history.append(
            (
                metrics.phase_epoch,
                metrics.weak_only.loss,
                metrics.weak_utility_loss,
            )
        )
        window = self.history[-self.config.stability_window :]
        weak_slope = _linear_slope(
            [(epoch, weak_loss) for epoch, weak_loss, _ in window]
        )
        utility_slope = _linear_slope(
            [(epoch, utility) for epoch, _, utility in window]
        )
        stable = (
            metrics.phase_epoch >= self.config.min_epochs
            and self.full_loss_bad_checks >= self.config.full_loss_patience
            and len(window) >= self.config.stability_window
            and abs(weak_slope) <= self.config.max_abs_slope_weak_loss
            and abs(utility_slope)
            <= self.config.max_abs_slope_weak_utility
        )
        self.plateau_confirmations = (
            self.plateau_confirmations + 1 if stable else 0
        )
        should_stop = (
            self.plateau_confirmations >= self.config.plateau_confirmations
        )
        if should_stop:
            self.stop_epoch = metrics.phase_epoch
        return self._decision(
            should_stop,
            "plateau_detected" if should_stop else None,
            weak_slope,
            utility_slope,
        )

    def _decision(
        self, should_stop, reason, weak_slope=None, utility_slope=None
    ):
        return Phase2PlateauDecision(
            should_stop=should_stop,
            stop_reason=reason,
            full_loss_bad_checks=self.full_loss_bad_checks,
            weak_loss_slope=weak_slope,
            weak_utility_slope=utility_slope,
            plateau_confirmations=self.plateau_confirmations,
        )

    def state_dict(self):
        return {
            "best_full_loss": self.best_full_loss,
            "full_loss_bad_checks": self.full_loss_bad_checks,
            "history": list(self.history),
            "plateau_confirmations": self.plateau_confirmations,
            "stop_epoch": self.stop_epoch,
        }

    def load_state_dict(self, state):
        self.best_full_loss = float(state["best_full_loss"])
        self.full_loss_bad_checks = int(state["full_loss_bad_checks"])
        self.history = [tuple(item) for item in state["history"]]
        self.plateau_confirmations = int(state["plateau_confirmations"])
        self.stop_epoch = state.get("stop_epoch")


class Phase2CheckpointSelector:
    def __init__(self, selection_window, selection_scope="final_window"):
        if int(selection_window) < 1:
            raise ValueError("selection_window must be positive")
        if selection_scope not in {"final_window", "global"}:
            raise ValueError("unknown phase 2 selection_scope")
        self.selection_window = int(selection_window)
        self.selection_scope = str(selection_scope)
        maxlen = (
            self.selection_window
            if self.selection_scope == "final_window"
            else 2
        )
        self.records = deque(maxlen=maxlen)

    @staticmethod
    def _loss_rank_key(record):
        return (
            record.metrics.full.loss,
            -record.metrics.weak_utility_loss,
            record.metrics.weak_only.loss,
            record.metrics.phase_epoch,
        )

    @staticmethod
    def _accuracy_rank_key(record):
        return (
            -record.metrics.full.accuracy,
            record.metrics.full.loss,
            record.metrics.phase_epoch,
        )

    def add(self, record: Phase2CheckpointRecord):
        if self.selection_scope == "global":
            previous_paths = {
                candidate.checkpoint_path for candidate in self.records
            }
            candidates = [*self.records, record]
            best_loss = min(candidates, key=self._loss_rank_key)
            best_accuracy = min(candidates, key=self._accuracy_rank_key)
            retained = {
                candidate.checkpoint_path: candidate
                for candidate in (best_loss, best_accuracy)
            }
            self.records.clear()
            self.records.extend(retained.values())
            return (
                record.checkpoint_path in retained
                and record.checkpoint_path not in previous_paths
            )
        self.records.append(record)
        return True

    @property
    def best_loss(self):
        if not self.records:
            return None
        return min(self.records, key=self._loss_rank_key)

    @property
    def best_accuracy(self):
        if not self.records:
            return None
        return min(self.records, key=self._accuracy_rank_key)

    @property
    def best(self):
        """Backward-compatible alias for the historical loss selector."""
        return self.best_loss

    def best_for(self, metric):
        if metric == "loss":
            return self.best_loss
        if metric == "accuracy":
            return self.best_accuracy
        raise ValueError("unknown phase 2 primary metric")

    def state_dict(self):
        return {
            "selection_window": self.selection_window,
            "selection_scope": self.selection_scope,
            "records": [record.state_dict() for record in self.records],
        }

    def load_state_dict(self, state):
        if int(state["selection_window"]) != self.selection_window:
            raise ValueError("phase 2 selection window changed across resume")
        saved_scope = str(state.get("selection_scope", "final_window"))
        if saved_scope != self.selection_scope:
            raise ValueError("phase 2 selection scope changed across resume")
        self.records.clear()
        self.records.extend(
            Phase2CheckpointRecord.from_state_dict(item)
            for item in state["records"]
        )


@dataclass(frozen=True)
class PairedEstimate:
    mean: float
    standard_error: float
    lower: float
    upper: float
    sample_count: int

    def state_dict(self):
        return asdict(self)


def _paired_estimate(
    values, confidence_level, max_looks, confidence_family_size
):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired uncertainty requires at least two samples")
    if not np.isfinite(values).all():
        return PairedEstimate(
            mean=float(values.mean()),
            standard_error=inf,
            lower=-inf,
            upper=inf,
            sample_count=int(values.size),
        )
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    alpha = (1.0 - float(confidence_level)) / (
        int(max_looks) * int(confidence_family_size)
    )
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    margin = z_value * standard_error
    return PairedEstimate(
        mean=mean,
        standard_error=standard_error,
        lower=mean - margin,
        upper=mean + margin,
        sample_count=int(values.size),
    )


def _loss_arrays(metrics):
    losses = metrics.per_example_losses
    if losses is None:
        raise ValueError(
            "adaptive Phase 3 stopping requires per-example validation losses"
        )
    return {
        name: np.asarray(getattr(losses, name), dtype=np.float64)
        for name in ("full", "dominant_only", "weak_only", "intervention")
    }

def _correctness_arrays(metrics):
    correctness = metrics.per_example_correctness
    if correctness is None:
        raise ValueError(
            "weak-recovery stopping requires per-example correctness"
        )
    return {
        name: np.asarray(getattr(correctness, name), dtype=np.float64)
        for name in ("full", "dominant_only", "weak_only", "intervention")
    }


def should_evaluate_phase_epoch(
    phase_epoch: int,
    phase_duration: int,
    interval_epochs: int,
    initial_dense_epochs: int = 0,
) -> bool:
    """Return whether an epoch is a scheduled validation look.

    The optional dense prefix makes early intervention dynamics observable
    without paying for per-epoch validation throughout the entire phase.  The
    final epoch is always evaluated, including when it is off cadence.
    """
    if phase_epoch < 1 or phase_duration < 1:
        raise ValueError("phase epochs and duration must be positive")
    if interval_epochs < 1:
        raise ValueError("evaluation interval must be positive")
    if initial_dense_epochs < 0:
        raise ValueError("initial dense evaluation epochs must be non-negative")
    return bool(
        phase_epoch <= initial_dense_epochs
        or phase_epoch % interval_epochs == 0
        or phase_epoch == phase_duration
    )


@dataclass(frozen=True)
class Phase3StopConfig:
    min_epochs: int = 5
    max_epochs: int = 200
    patience: int = 8
    safety_patience: int = 3
    min_delta: float = 0.001
    min_weak_quality_gain: float = 0.0
    min_weak_utility_gain: float = 0.0
    max_full_loss_increase: float = 0.05
    max_dominant_loss_increase: float = 0.05
    hard_max_full_loss_increase: float = 0.20
    hard_max_dominant_loss_increase: float = 0.20
    adaptive_rule: bool = False
    confidence_level: float = 0.95
    confidence_family_size: int = 8
    max_looks: int = 200
    minimum_exposure_evaluations: int = 5
    reversal_patience: int = 3
    futility_patience: int = 3
    trend_window: int = 5
    futility_prediction_horizon_epochs: int = 10
    decision_rule: str = "legacy"
    emergency_stop_mode: str = "loss_limits"
    shadow_continue_after_stop: bool = False
    min_weak_accuracy_gain: float = 0.0
    recovery_primary_metric: str = "accuracy"
    max_weak_accuracy_slope: float = 0.0005
    max_weak_quality_slope: float = 0.001
    plateau_patience: int = 3
    max_full_accuracy_drop: float = 0.05
    max_dominant_accuracy_drop: float = 0.05
    hard_max_full_accuracy_drop: float = 0.10
    hard_max_dominant_accuracy_drop: float = 0.10
    target_patience: int = 2
    pareto_patience: int = 2
    futility_harm_patience: int = 2
    gradient_conflict_threshold: float = 0.0
    parity_patience: int = 2
    recovery_fraction_threshold: float = 1.0

    def __post_init__(self):
        if min(
            self.min_epochs,
            self.max_epochs,
            self.patience,
            self.safety_patience,
            self.max_looks,
            self.confidence_family_size,
            self.minimum_exposure_evaluations,
            self.reversal_patience,
            self.futility_patience,
            self.trend_window,
            self.futility_prediction_horizon_epochs,
            self.plateau_patience,
            self.target_patience,
            self.pareto_patience,
            self.futility_harm_patience,
            self.parity_patience,
        ) < 1:
            raise ValueError("phase 3 count parameters must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("phase 3 min_epochs exceeds max_epochs")
        if self.min_delta < 0:
            raise ValueError("phase 3 min_delta must be non-negative")
        if not 0.0 < self.recovery_fraction_threshold <= 1.0:
            raise ValueError(
                "recovery_fraction_threshold must be in (0, 1]"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("phase 3 confidence_level must be in (0, 1)")
        if self.decision_rule not in {
            "legacy",
            "weak_recovery",
            "local_accuracy",
            "relative_unimodal_parity",
        }:
            raise ValueError("unknown phase 3 decision_rule")
        if self.recovery_primary_metric not in {
            "accuracy",
            "accuracy_and_loss",
        }:
            raise ValueError("unknown phase 3 recovery_primary_metric")
        if self.emergency_stop_mode not in {
            "loss_limits",
            "numerical_only",
            "disabled",
        }:
            raise ValueError("unknown phase 3 emergency_stop_mode")
        if (
            self.max_weak_accuracy_slope < 0
            or self.max_weak_quality_slope < 0
        ):
            raise ValueError(
                "weak-recovery slope tolerances must be non-negative"
            )
        if min(
            self.max_full_accuracy_drop,
            self.max_dominant_accuracy_drop,
            self.hard_max_full_accuracy_drop,
            self.hard_max_dominant_accuracy_drop,
        ) < 0:
            raise ValueError("accuracy-drop tolerances must be non-negative")


@dataclass(frozen=True)
class InterventionCheckpointRecord:
    metrics: ModalityEvaluationResult
    checkpoint_path: str
    weak_quality_gain: float
    weak_utility_gain: float
    full_loss_increase: float
    dominant_loss_increase: float
    compatibility_drift_accuracy: float
    reactivation_full_loss_gap: float
    is_feasible: bool
    is_safe: bool
    paired_estimates: dict[str, dict[str, float]] | None = None
    weak_accuracy_gain: float = 0.0
    dominant_ratio: float | None = None
    weak_ratio: float | None = None
    parity_gap: float | None = None
    recovery_fraction: float | None = None
    recovery_fraction_threshold: float | None = None

    def state_dict(self):
        result = asdict(self)
        result["metrics"] = self.metrics.state_dict()
        return result

    @classmethod
    def from_state_dict(cls, state):
        values = dict(state)
        values.setdefault("weak_accuracy_gain", 0.0)
        values.setdefault("dominant_ratio", None)
        values.setdefault("weak_ratio", None)
        values.setdefault("parity_gap", None)
        values.setdefault("recovery_fraction", None)
        values.setdefault("recovery_fraction_threshold", None)
        values["metrics"] = ModalityEvaluationResult.from_state_dict(
            values["metrics"]
        )
        return cls(**values)


@dataclass(frozen=True)
class Phase3StopDecision:
    should_stop: bool
    stop_reason: str | None
    selection_status: str | None
    selected: InterventionCheckpointRecord | None
    current: InterventionCheckpointRecord
    bad_checks: int
    safety_bad_checks: int


def _tolerant_lexicographic_better(candidate, incumbent, min_delta):
    candidate_score = (
        candidate.weak_utility_gain,
        candidate.weak_quality_gain,
        -candidate.full_loss_increase,
        -candidate.dominant_loss_increase,
    )
    incumbent_score = (
        incumbent.weak_utility_gain,
        incumbent.weak_quality_gain,
        -incumbent.full_loss_increase,
        -incumbent.dominant_loss_increase,
    )
    for candidate_value, incumbent_value in zip(
        candidate_score, incumbent_score
    ):
        if candidate_value > incumbent_value + min_delta:
            return True
        if candidate_value < incumbent_value - min_delta:
            return False
    return candidate.metrics.phase_epoch < incumbent.metrics.phase_epoch


class Phase3InterventionStopper:
    def __init__(
        self,
        config: Phase3StopConfig,
        baseline: ModalityEvaluationResult,
    ):
        self.config = config
        self.baseline = baseline
        self.best_feasible = None
        self.best_safe = None
        self.bad_checks = 0
        self.safety_bad_checks = 0
        self.reversal_bad_checks = 0
        self.futility_bad_checks = 0
        self.evaluation_count = 0
        self.history = deque(maxlen=config.trend_window)
        self.last_trend_estimates = None
        self.last_optimistic_bounds = None
        self.final_decision = None

    def _estimate(self, values):
        return _paired_estimate(
            values,
            self.config.confidence_level,
            self.config.max_looks,
            self.config.confidence_family_size,
        )

    def _baseline_estimates(self, metrics):
        baseline = _loss_arrays(self.baseline)
        current = _loss_arrays(metrics)
        baseline_utility = baseline["dominant_only"] - baseline["full"]
        current_utility = current["dominant_only"] - current["full"]
        return {
            "weak_quality_gain": self._estimate(
                baseline["weak_only"] - current["weak_only"]
            ),
            "weak_utility_gain": self._estimate(
                current_utility - baseline_utility
            ),
            "full_loss_increase": self._estimate(
                current["full"] - baseline["full"]
            ),
            "dominant_loss_increase": self._estimate(
                current["dominant_only"] - baseline["dominant_only"]
            ),
        }

    def _comparison_to_best(self, metrics):
        if self.best_feasible is None:
            return None
        best = _loss_arrays(self.best_feasible.metrics)
        current = _loss_arrays(metrics)
        best_utility = best["dominant_only"] - best["full"]
        current_utility = current["dominant_only"] - current["full"]
        return {
            "utility_change_from_best": self._estimate(
                current_utility - best_utility
            ),
            "weak_quality_change_from_best": self._estimate(
                best["weak_only"] - current["weak_only"]
            ),
        }

    def _trend_estimates(self):
        if len(self.history) < self.config.trend_window:
            return None
        epochs = np.asarray(
            [metrics.phase_epoch for metrics in self.history],
            dtype=np.float64,
        )
        centered = epochs - epochs.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            return None
        baseline = _loss_arrays(self.baseline)
        baseline_utility = baseline["dominant_only"] - baseline["full"]
        utility_gains = []
        quality_gains = []
        for metrics in self.history:
            current = _loss_arrays(metrics)
            utility_gains.append(
                current["dominant_only"] - current["full"]
                - baseline_utility
            )
            quality_gains.append(
                baseline["weak_only"] - current["weak_only"]
            )
        utility_matrix = np.stack(utility_gains, axis=0)
        quality_matrix = np.stack(quality_gains, axis=0)
        utility_slopes = (
            centered[:, None] * utility_matrix
        ).sum(axis=0) / denominator
        quality_slopes = (
            centered[:, None] * quality_matrix
        ).sum(axis=0) / denominator
        return {
            "weak_utility_slope": self._estimate(utility_slopes),
            "weak_quality_slope": self._estimate(quality_slopes),
        }

    def _optimistic_bounds(self, record, trend_estimates):
        horizon = self.config.futility_prediction_horizon_epochs
        estimates = record.paired_estimates
        return {
            "weak_utility_gain": (
                estimates["weak_utility_gain"]["upper"]
                + max(0.0, trend_estimates["weak_utility_slope"].upper)
                * horizon
            ),
            "weak_quality_gain": (
                estimates["weak_quality_gain"]["upper"]
                + max(0.0, trend_estimates["weak_quality_slope"].upper)
                * horizon
            ),
        }

    def _record(self, metrics, checkpoint_path):
        weak_quality_gain = (
            self.baseline.weak_only.loss - metrics.weak_only.loss
        )
        weak_utility_gain = (
            metrics.weak_utility_loss - self.baseline.weak_utility_loss
        )
        full_loss_increase = metrics.full.loss - self.baseline.full.loss
        dominant_loss_increase = (
            metrics.dominant_only.loss - self.baseline.dominant_only.loss
        )
        estimates = None
        if self.config.adaptive_rule:
            estimates = self._baseline_estimates(metrics)
            is_safe = (
                estimates["full_loss_increase"].upper
                <= self.config.max_full_loss_increase
                and estimates["dominant_loss_increase"].upper
                <= self.config.max_dominant_loss_increase
            )
            is_feasible = (
                is_safe
                and estimates["weak_quality_gain"].lower
                >= self.config.min_weak_quality_gain
                and estimates["weak_utility_gain"].lower
                >= self.config.min_weak_utility_gain
            )
        else:
            is_safe = (
                full_loss_increase <= self.config.max_full_loss_increase
                and dominant_loss_increase
                <= self.config.max_dominant_loss_increase
            )
            is_feasible = (
                is_safe
                and weak_quality_gain >= self.config.min_weak_quality_gain
                and weak_utility_gain >= self.config.min_weak_utility_gain
            )
        return InterventionCheckpointRecord(
            metrics=metrics,
            checkpoint_path=str(checkpoint_path),
            weak_quality_gain=weak_quality_gain,
            weak_utility_gain=weak_utility_gain,
            full_loss_increase=full_loss_increase,
            dominant_loss_increase=dominant_loss_increase,
            compatibility_drift_accuracy=(
                self.baseline.dominant_only.accuracy
                - metrics.dominant_only.accuracy
            ),
            reactivation_full_loss_gap=(
                metrics.full.loss - metrics.intervention.loss
            ),
            is_feasible=is_feasible,
            is_safe=is_safe,
            paired_estimates=(
                {
                    name: estimate.state_dict()
                    for name, estimate in estimates.items()
                }
                if estimates is not None
                else None
            ),
        )

    def update(self, metrics, checkpoint_path):
        if self.final_decision is not None:
            return self.final_decision
        self.evaluation_count += 1
        record = self._record(metrics, checkpoint_path)
        if self.config.adaptive_rule:
            self.history.append(metrics)
        if record.is_safe and (
            self.best_safe is None
            or _tolerant_lexicographic_better(
                record, self.best_safe, self.config.min_delta
            )
        ):
            self.best_safe = record

        best_improved = False
        if record.is_feasible:
            if self.best_feasible is None or _tolerant_lexicographic_better(
                record, self.best_feasible, self.config.min_delta
            ):
                self.best_feasible = record
                self.bad_checks = 0
                best_improved = True
            else:
                self.bad_checks += 1
        elif self.best_feasible is not None:
            self.bad_checks += 1

        if self.config.adaptive_rule:
            estimates = record.paired_estimates
            hard_violation = (
                estimates["full_loss_increase"]["lower"]
                > self.config.hard_max_full_loss_increase
                or estimates["dominant_loss_increase"]["lower"]
                > self.config.hard_max_dominant_loss_increase
            )
        else:
            hard_violation = (
                record.full_loss_increase
                > self.config.hard_max_full_loss_increase
                or record.dominant_loss_increase
                > self.config.hard_max_dominant_loss_increase
            )
        self.safety_bad_checks = (
            self.safety_bad_checks + 1 if hard_violation else 0
        )

        enough_exposure = (
            metrics.phase_epoch >= self.config.min_epochs
            and self.evaluation_count
            >= self.config.minimum_exposure_evaluations
        )
        if self.config.adaptive_rule and enough_exposure:
            if best_improved:
                self.reversal_bad_checks = 0
            elif self.best_feasible is not None:
                comparison = self._comparison_to_best(metrics)
                reversal = (
                    comparison["utility_change_from_best"].upper < 0.0
                    and comparison[
                        "weak_quality_change_from_best"
                    ].upper <= 0.0
                )
                self.reversal_bad_checks = (
                    self.reversal_bad_checks + 1 if reversal else 0
                )
            else:
                self.reversal_bad_checks = 0

            if self.best_feasible is None:
                self.last_trend_estimates = self._trend_estimates()
                if self.last_trend_estimates is None:
                    self.last_optimistic_bounds = None
                    futile = False
                else:
                    self.last_optimistic_bounds = self._optimistic_bounds(
                        record, self.last_trend_estimates
                    )
                    weak_quality_slope_upper = (
                        self.last_trend_estimates[
                            "weak_quality_slope"
                        ].upper
                    )
                    futile = (
                        self.last_optimistic_bounds["weak_utility_gain"]
                        <= self.config.min_weak_utility_gain
                        and (
                            self.last_optimistic_bounds["weak_quality_gain"]
                            <= self.config.min_weak_quality_gain
                            or weak_quality_slope_upper <= 0.0
                        )
                    )
                self.futility_bad_checks = (
                    self.futility_bad_checks + 1 if futile else 0
                )
            else:
                self.futility_bad_checks = 0
                self.last_trend_estimates = self._trend_estimates()
                self.last_optimistic_bounds = None
        else:
            self.reversal_bad_checks = 0
            self.futility_bad_checks = 0

        stop_reason = None
        if self.safety_bad_checks >= self.config.safety_patience:
            stop_reason = "hard_safety"
        elif (
            self.config.adaptive_rule
            and self.reversal_bad_checks >= self.config.reversal_patience
        ):
            stop_reason = "trend_reversal"
        elif (
            self.config.adaptive_rule
            and self.futility_bad_checks >= self.config.futility_patience
        ):
            stop_reason = "futility"
        elif (
            not self.config.adaptive_rule
            and metrics.phase_epoch >= self.config.min_epochs
            and self.best_feasible is not None
            and self.bad_checks >= self.config.patience
        ):
            stop_reason = "patience"
        elif metrics.phase_epoch >= self.config.max_epochs:
            stop_reason = "max_epochs"

        selected, selection_status = self.selection()
        decision = Phase3StopDecision(
            should_stop=stop_reason is not None,
            stop_reason=stop_reason,
            selection_status=selection_status if stop_reason else None,
            selected=selected if stop_reason else None,
            current=record,
            bad_checks=self.bad_checks,
            safety_bad_checks=self.safety_bad_checks,
        )
        if decision.should_stop:
            self.final_decision = decision
        return decision

    def selection(self):
        if self.best_feasible is not None:
            return self.best_feasible, "best_feasible"
        if self.best_safe is not None:
            return self.best_safe, "best_safe"
        return None, "rollback_pre_phase3"

    def state_dict(self):
        return {
            "baseline": self.baseline.state_dict(),
            "best_feasible": (
                self.best_feasible.state_dict()
                if self.best_feasible is not None
                else None
            ),
            "best_safe": (
                self.best_safe.state_dict()
                if self.best_safe is not None
                else None
            ),
            "bad_checks": self.bad_checks,
            "safety_bad_checks": self.safety_bad_checks,
            "reversal_bad_checks": self.reversal_bad_checks,
            "futility_bad_checks": self.futility_bad_checks,
            "evaluation_count": self.evaluation_count,
            "history": [
                metrics.state_dict() for metrics in self.history
            ],
            "last_optimistic_bounds": self.last_optimistic_bounds,
            "final_decision": (
                {
                    "stop_reason": self.final_decision.stop_reason,
                    "selection_status": self.final_decision.selection_status,
                    "selected": (
                        self.final_decision.selected.state_dict()
                        if self.final_decision.selected is not None
                        else None
                    ),
                    "current": self.final_decision.current.state_dict(),
                    "bad_checks": self.final_decision.bad_checks,
                    "safety_bad_checks": self.final_decision.safety_bad_checks,
                }
                if self.final_decision is not None
                else None
            ),
        }

    def load_state_dict(self, state):
        if ModalityEvaluationResult.from_state_dict(
            state["baseline"]
        ) != self.baseline:
            raise ValueError("phase 3 baseline changed across resume")
        self.best_feasible = (
            InterventionCheckpointRecord.from_state_dict(
                state["best_feasible"]
            )
            if state.get("best_feasible") is not None
            else None
        )
        self.best_safe = (
            InterventionCheckpointRecord.from_state_dict(state["best_safe"])
            if state.get("best_safe") is not None
            else None
        )
        self.bad_checks = int(state["bad_checks"])
        self.safety_bad_checks = int(state["safety_bad_checks"])
        self.reversal_bad_checks = int(state.get("reversal_bad_checks", 0))
        self.futility_bad_checks = int(state.get("futility_bad_checks", 0))
        self.evaluation_count = int(state.get("evaluation_count", 0))
        self.history.clear()
        self.history.extend(
            ModalityEvaluationResult.from_state_dict(item)
            for item in state.get("history", [])
        )
        self.last_trend_estimates = (
            self._trend_estimates() if self.config.adaptive_rule else None
        )
        self.last_optimistic_bounds = state.get(
            "last_optimistic_bounds"
        )
        final = state.get("final_decision")
        if final is not None:
            self.final_decision = Phase3StopDecision(
                should_stop=True,
                stop_reason=final["stop_reason"],
                selection_status=final["selection_status"],
                selected=(
                    InterventionCheckpointRecord.from_state_dict(
                        final["selected"]
                    )
                    if final["selected"] is not None
                    else None
                ),
                current=InterventionCheckpointRecord.from_state_dict(
                    final["current"]
                ),
                bad_checks=int(final["bad_checks"]),
                safety_bad_checks=int(final["safety_bad_checks"]),
            )


class Phase3RelativeUnimodalStopper(Phase3InterventionStopper):
    """Stop after recovering a configured fraction of the relative deficit.

    The dominant target is frozen from the pre-intervention validation look.
    A threshold of 1.0 reproduces exact relative parity. Lower thresholds stop
    after closing the requested fraction of the baseline weak-to-dominant gap.
    Consecutive qualifying looks confirm the stop, while the first checkpoint
    in that uninterrupted streak is selected for Phase 4.
    """

    def __init__(
        self,
        config: Phase3StopConfig,
        baseline: ModalityEvaluationResult,
        *,
        unimodal_left_accuracy: float,
        unimodal_right_accuracy: float,
    ):
        if config.decision_rule != "relative_unimodal_parity":
            raise ValueError(
                "Phase3RelativeUnimodalStopper requires "
                "relative_unimodal_parity"
            )
        super().__init__(config, baseline)
        self.unimodal_left_accuracy = float(unimodal_left_accuracy)
        self.unimodal_right_accuracy = float(unimodal_right_accuracy)
        if not (
            np.isfinite(self.unimodal_left_accuracy)
            and 0.0 < self.unimodal_left_accuracy <= 1.0
            and np.isfinite(self.unimodal_right_accuracy)
            and 0.0 < self.unimodal_right_accuracy <= 1.0
        ):
            raise ValueError(
                "unimodal reference accuracies must be finite and in (0, 1]"
            )
        self.dominant_ratio = (
            baseline.dominant_only.accuracy / self.unimodal_left_accuracy
        )
        self.baseline_weak_ratio = (
            baseline.weak_only.accuracy / self.unimodal_right_accuracy
        )
        self.recovery_denominator = (
            self.dominant_ratio - self.baseline_weak_ratio
        )
        self.recovery_fraction_threshold = float(
            config.recovery_fraction_threshold
        )
        self.parity_confirmations = 0
        self.first_parity_candidate = None
        self.best_ratio = None
        self.parity_selected = None
        self.first_trigger_epochs = {}

    @property
    def _target_stop_reason(self):
        if self.recovery_fraction_threshold == 1.0:
            return "relative_parity_reached"
        return "relative_recovery_reached"

    @property
    def _target_selection_status(self):
        if self.recovery_fraction_threshold == 1.0:
            return "first_parity_checkpoint"
        return "first_recovery_checkpoint"

    def _record(self, metrics, checkpoint_path):
        record = super()._record(metrics, checkpoint_path)
        weak_ratio = (
            metrics.weak_only.accuracy / self.unimodal_right_accuracy
        )
        finite = all(
            np.isfinite(float(value))
            for value in (
                metrics.weak_only.accuracy,
                metrics.weak_only.loss,
                metrics.full.accuracy,
                metrics.dominant_only.accuracy,
                weak_ratio,
                self.dominant_ratio,
            )
        )
        parity_gap = weak_ratio - self.dominant_ratio
        if self.recovery_denominator <= 0.0:
            recovery_fraction = 1.0
        else:
            recovery_fraction = (
                weak_ratio - self.baseline_weak_ratio
            ) / self.recovery_denominator
        finite = bool(finite and np.isfinite(recovery_fraction))
        return replace(
            record,
            is_feasible=bool(
                finite
                and recovery_fraction >= self.recovery_fraction_threshold
            ),
            is_safe=bool(finite),
            dominant_ratio=self.dominant_ratio,
            weak_ratio=weak_ratio,
            parity_gap=parity_gap,
            recovery_fraction=recovery_fraction,
            recovery_fraction_threshold=self.recovery_fraction_threshold,
        )

    @staticmethod
    def _better_ratio(candidate, incumbent):
        if incumbent is None:
            return True
        if candidate.weak_ratio != incumbent.weak_ratio:
            return candidate.weak_ratio > incumbent.weak_ratio
        if candidate.metrics.weak_only.loss != incumbent.metrics.weak_only.loss:
            return (
                candidate.metrics.weak_only.loss
                < incumbent.metrics.weak_only.loss
            )
        return candidate.metrics.phase_epoch < incumbent.metrics.phase_epoch

    @property
    def retained_checkpoint_paths(self):
        return {
            record.checkpoint_path
            for record in (
                self.first_parity_candidate,
                self.best_ratio,
                self.parity_selected,
            )
            if record is not None
        }

    def _remember_trigger(self, name, epoch):
        self.first_trigger_epochs.setdefault(str(name), int(epoch))

    def _decision(self, record, stop_reason=None, selected=None):
        selection_status = None
        if stop_reason is not None:
            if selected is None:
                selection_status = "rollback_pre_phase3"
            elif stop_reason == self._target_stop_reason:
                selection_status = self._target_selection_status
            else:
                selection_status = "best_weak_ratio"
        decision = Phase3StopDecision(
            should_stop=stop_reason is not None,
            stop_reason=stop_reason,
            selection_status=selection_status,
            selected=selected,
            current=record,
            bad_checks=self.parity_confirmations,
            safety_bad_checks=0,
        )
        if decision.should_stop:
            self.final_decision = decision
        return decision

    def initialize_baseline(self, checkpoint_path):
        """Register e3=0 and skip intervention when parity already holds."""
        if self.evaluation_count:
            raise RuntimeError("relative parity baseline was already initialized")
        record = self._record(self.baseline, checkpoint_path)
        self.best_ratio = record if record.is_safe else None
        self.best_safe = self.best_ratio
        self.best_feasible = record if record.is_feasible else None
        if not record.is_safe:
            self._remember_trigger("numerical_emergency", 0)
            return self._decision(record, "numerical_emergency", None)
        if record.is_feasible:
            self.parity_selected = record
            self._remember_trigger(self._target_stop_reason, 0)
            return self._decision(
                record,
                self._target_stop_reason,
                record,
            )
        return self._decision(record)

    def update(self, metrics, checkpoint_path, diagnostics=None):
        del diagnostics
        if self.final_decision is not None:
            return self.final_decision
        self.evaluation_count += 1
        record = self._record(metrics, checkpoint_path)
        if not record.is_safe:
            self._remember_trigger("numerical_emergency", metrics.phase_epoch)
            selected = self.best_ratio
            return self._decision(record, "numerical_emergency", selected)

        if self._better_ratio(record, self.best_ratio):
            self.best_ratio = record
            self.best_safe = record

        if record.is_feasible:
            if self.parity_confirmations == 0:
                self.first_parity_candidate = record
                trigger_name = (
                    "first_parity"
                    if self.recovery_fraction_threshold == 1.0
                    else "first_recovery"
                )
                self._remember_trigger(trigger_name, metrics.phase_epoch)
            self.parity_confirmations += 1
            self.best_feasible = self.first_parity_candidate
        else:
            self.parity_confirmations = 0
            self.first_parity_candidate = None
            self.best_feasible = None

        if self.parity_confirmations >= self.config.parity_patience:
            self.parity_selected = self.first_parity_candidate
            self._remember_trigger(
                self._target_stop_reason, metrics.phase_epoch
            )
            return self._decision(
                record,
                self._target_stop_reason,
                self.parity_selected,
            )
        if metrics.phase_epoch >= self.config.max_epochs:
            self._remember_trigger("max_epochs", metrics.phase_epoch)
            return self._decision(record, "max_epochs", self.best_ratio)
        return self._decision(record)

    def selection(self):
        if self.parity_selected is not None:
            return self.parity_selected, self._target_selection_status
        if self.best_ratio is not None:
            return self.best_ratio, "best_weak_ratio"
        return None, "rollback_pre_phase3"

    def state_dict(self):
        state = super().state_dict()
        state["relative_unimodal_parity"] = {
            "unimodal_left_accuracy": self.unimodal_left_accuracy,
            "unimodal_right_accuracy": self.unimodal_right_accuracy,
            "dominant_ratio": self.dominant_ratio,
            "baseline_weak_ratio": self.baseline_weak_ratio,
            "recovery_denominator": self.recovery_denominator,
            "recovery_fraction_threshold": self.recovery_fraction_threshold,
            "parity_confirmations": self.parity_confirmations,
            "first_parity_candidate": (
                self.first_parity_candidate.state_dict()
                if self.first_parity_candidate is not None
                else None
            ),
            "best_ratio": (
                self.best_ratio.state_dict()
                if self.best_ratio is not None
                else None
            ),
            "parity_selected": (
                self.parity_selected.state_dict()
                if self.parity_selected is not None
                else None
            ),
            "first_trigger_epochs": dict(self.first_trigger_epochs),
        }
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        parity = state.get("relative_unimodal_parity")
        if parity is None:
            raise ValueError("relative parity checkpoint lacks controller state")
        if (
            float(parity["unimodal_left_accuracy"])
            != self.unimodal_left_accuracy
            or float(parity["unimodal_right_accuracy"])
            != self.unimodal_right_accuracy
            or float(parity["dominant_ratio"]) != self.dominant_ratio
            or float(
                parity.get(
                    "baseline_weak_ratio", self.baseline_weak_ratio
                )
            )
            != self.baseline_weak_ratio
            or float(
                parity.get(
                    "recovery_denominator", self.recovery_denominator
                )
            )
            != self.recovery_denominator
            or float(
                parity.get("recovery_fraction_threshold", 1.0)
            )
            != self.recovery_fraction_threshold
        ):
            raise ValueError(
                "unimodal recovery target changed across resume"
            )
        self.parity_confirmations = int(parity["parity_confirmations"])
        for name in (
            "first_parity_candidate",
            "best_ratio",
            "parity_selected",
        ):
            value = parity.get(name)
            setattr(
                self,
                name,
                (
                    InterventionCheckpointRecord.from_state_dict(value)
                    if value is not None
                    else None
                ),
            )
        self.best_safe = self.best_ratio
        self.best_feasible = self.first_parity_candidate
        self.first_trigger_epochs = {
            str(name): int(epoch)
            for name, epoch in parity.get("first_trigger_epochs", {}).items()
        }



def _weak_recovery_better(candidate, incumbent, min_delta):
    candidate_score = (
        candidate.metrics.weak_only.accuracy,
        candidate.metrics.full.accuracy,
        candidate.metrics.dominant_only.accuracy,
        -candidate.metrics.weak_only.loss,
    )
    incumbent_score = (
        incumbent.metrics.weak_only.accuracy,
        incumbent.metrics.full.accuracy,
        incumbent.metrics.dominant_only.accuracy,
        -incumbent.metrics.weak_only.loss,
    )
    for candidate_value, incumbent_value in zip(
        candidate_score, incumbent_score
    ):
        if candidate_value > incumbent_value + min_delta:
            return True
        if candidate_value < incumbent_value - min_delta:
            return False
    return candidate.metrics.phase_epoch < incumbent.metrics.phase_epoch


class Phase3RecoveryStopper(Phase3InterventionStopper):
    """Validation-only weak-branch recovery rule with diagnostic compatibility."""

    def __init__(self, config, baseline):
        if config.decision_rule not in {"weak_recovery", "local_accuracy"}:
            raise ValueError(
                "Phase3RecoveryStopper requires an accuracy decision rule"
            )
        super().__init__(config, baseline)
        _correctness_arrays(baseline)
        self.first_trigger_epochs = {}

    def _baseline_estimates(self, metrics):
        estimates = super()._baseline_estimates(metrics)
        baseline = _correctness_arrays(self.baseline)
        current = _correctness_arrays(metrics)
        estimates["weak_accuracy_gain"] = self._estimate(
            current["weak_only"] - baseline["weak_only"]
        )
        estimates["full_accuracy_change"] = self._estimate(
            current["full"] - baseline["full"]
        )
        estimates["dominant_accuracy_change"] = self._estimate(
            current["dominant_only"] - baseline["dominant_only"]
        )
        estimates["weak_minus_dominant_accuracy"] = self._estimate(
            current["weak_only"] - current["dominant_only"]
        )
        estimates["full_minus_weak_accuracy"] = self._estimate(
            current["full"] - current["weak_only"]
        )
        return estimates

    def _comparison_to_best(self, metrics):
        if self.best_feasible is None:
            return None
        best_losses = _loss_arrays(self.best_feasible.metrics)
        current_losses = _loss_arrays(metrics)
        best_correct = _correctness_arrays(self.best_feasible.metrics)
        current_correct = _correctness_arrays(metrics)
        return {
            "weak_accuracy_change_from_best": self._estimate(
                current_correct["weak_only"] - best_correct["weak_only"]
            ),
            "weak_quality_change_from_best": self._estimate(
                best_losses["weak_only"] - current_losses["weak_only"]
            ),
        }

    def _trend_estimates(self):
        if len(self.history) < self.config.trend_window:
            return None
        epochs = np.asarray(
            [metrics.phase_epoch for metrics in self.history],
            dtype=np.float64,
        )
        centered = epochs - epochs.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            return None
        baseline_losses = _loss_arrays(self.baseline)["weak_only"]
        baseline_correct = _correctness_arrays(self.baseline)["weak_only"]
        quality_gains = []
        accuracy_gains = []
        full_accuracies = []
        dominant_accuracies = []
        for metrics in self.history:
            current_correct = _correctness_arrays(metrics)
            quality_gains.append(
                baseline_losses - _loss_arrays(metrics)["weak_only"]
            )
            accuracy_gains.append(
                current_correct["weak_only"] - baseline_correct
            )
            full_accuracies.append(current_correct["full"])
            dominant_accuracies.append(current_correct["dominant_only"])
        quality_matrix = np.stack(quality_gains, axis=0)
        accuracy_matrix = np.stack(accuracy_gains, axis=0)
        quality_slopes = (
            centered[:, None] * quality_matrix
        ).sum(axis=0) / denominator
        accuracy_slopes = (
            centered[:, None] * accuracy_matrix
        ).sum(axis=0) / denominator
        full_slopes = (
            centered[:, None] * np.stack(full_accuracies, axis=0)
        ).sum(axis=0) / denominator
        dominant_slopes = (
            centered[:, None] * np.stack(dominant_accuracies, axis=0)
        ).sum(axis=0) / denominator
        return {
            "weak_quality_slope": self._estimate(quality_slopes),
            "weak_accuracy_slope": self._estimate(accuracy_slopes),
            "full_accuracy_slope": self._estimate(full_slopes),
            "dominant_accuracy_slope": self._estimate(dominant_slopes),
        }

    def _record(self, metrics, checkpoint_path):
        estimates = self._baseline_estimates(metrics)
        weak_quality_gain = (
            self.baseline.weak_only.loss - metrics.weak_only.loss
        )
        weak_accuracy_gain = (
            metrics.weak_only.accuracy - self.baseline.weak_only.accuracy
        )
        weak_utility_gain = (
            metrics.weak_utility_loss - self.baseline.weak_utility_loss
        )
        full_loss_increase = metrics.full.loss - self.baseline.full.loss
        dominant_loss_increase = (
            metrics.dominant_only.loss - self.baseline.dominant_only.loss
        )
        compatibility_safe = (
            estimates["full_accuracy_change"].lower
            >= -self.config.max_full_accuracy_drop
            and estimates["dominant_accuracy_change"].lower
            >= -self.config.max_dominant_accuracy_drop
        )
        recovery_feasible = (
            estimates["weak_accuracy_gain"].lower
            >= self.config.min_weak_accuracy_gain
        )
        if self.config.recovery_primary_metric == "accuracy_and_loss":
            recovery_feasible = (
                recovery_feasible
                and estimates["weak_quality_gain"].lower
                >= self.config.min_weak_quality_gain
            )
        return InterventionCheckpointRecord(
            metrics=metrics,
            checkpoint_path=str(checkpoint_path),
            weak_quality_gain=weak_quality_gain,
            weak_utility_gain=weak_utility_gain,
            full_loss_increase=full_loss_increase,
            dominant_loss_increase=dominant_loss_increase,
            compatibility_drift_accuracy=(
                self.baseline.dominant_only.accuracy
                - metrics.dominant_only.accuracy
            ),
            reactivation_full_loss_gap=(
                metrics.full.loss - metrics.intervention.loss
            ),
            is_feasible=(recovery_feasible and compatibility_safe),
            is_safe=compatibility_safe,
            paired_estimates={
                name: estimate.state_dict()
                for name, estimate in estimates.items()
            },
            weak_accuracy_gain=weak_accuracy_gain,
        )

    @staticmethod
    def _finite(record):
        values = (
            record.metrics.full.loss,
            record.metrics.full.accuracy,
            record.metrics.dominant_only.loss,
            record.metrics.dominant_only.accuracy,
            record.metrics.weak_only.loss,
            record.metrics.weak_only.accuracy,
            record.metrics.intervention.loss,
            record.metrics.intervention.accuracy,
            record.weak_quality_gain,
            record.weak_accuracy_gain,
        )
        return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())

    def _hard_violation(self, record):
        if self.config.emergency_stop_mode == "disabled":
            return False
        if self.config.emergency_stop_mode == "numerical_only":
            return not self._finite(record)
        estimates = record.paired_estimates
        return (
            estimates["full_loss_increase"]["lower"]
            > self.config.hard_max_full_loss_increase
            or estimates["dominant_loss_increase"]["lower"]
            > self.config.hard_max_dominant_loss_increase
        )

    def _compatibility_violation(self, record):
        estimates = record.paired_estimates
        return (
            estimates["full_accuracy_change"]["upper"]
            < -self.config.hard_max_full_accuracy_drop
            or estimates["dominant_accuracy_change"]["upper"]
            < -self.config.hard_max_dominant_accuracy_drop
        )

    def _remember_trigger(self, name, active, epoch):
        if active and name not in self.first_trigger_epochs:
            self.first_trigger_epochs[name] = int(epoch)

    def update(self, metrics, checkpoint_path):
        frozen = self.final_decision
        if frozen is not None and not self.config.shadow_continue_after_stop:
            return frozen

        self.evaluation_count += 1
        record = self._record(metrics, checkpoint_path)
        self.history.append(metrics)

        if record.is_safe and (
            self.best_safe is None
            or _weak_recovery_better(
                record, self.best_safe, self.config.min_delta
            )
        ):
            self.best_safe = record

        best_improved = False
        if record.is_feasible and (
            self.best_feasible is None
            or _weak_recovery_better(
                record, self.best_feasible, self.config.min_delta
            )
        ):
            self.best_feasible = record
            self.bad_checks = 0
            best_improved = True
        elif self.best_feasible is not None:
            self.bad_checks += 1

        hard_violation = self._hard_violation(record)
        compatibility_violation = self._compatibility_violation(record)
        self.safety_bad_checks = (
            self.safety_bad_checks + 1 if compatibility_violation else 0
        )
        enough_exposure = (
            metrics.phase_epoch >= self.config.min_epochs
            and self.evaluation_count
            >= self.config.minimum_exposure_evaluations
        )

        reversal = False
        if enough_exposure and self.best_feasible is not None:
            if best_improved:
                self.reversal_bad_checks = 0
            else:
                comparison = self._comparison_to_best(metrics)
                reversal = (
                    comparison["weak_accuracy_change_from_best"].upper < 0.0
                )
                if (
                    self.config.recovery_primary_metric
                    == "accuracy_and_loss"
                ):
                    reversal = (
                        reversal
                        and comparison[
                            "weak_quality_change_from_best"
                        ].upper <= 0.0
                    )
                self.reversal_bad_checks = (
                    self.reversal_bad_checks + 1 if reversal else 0
                )
        else:
            self.reversal_bad_checks = 0

        self.last_trend_estimates = (
            self._trend_estimates() if enough_exposure else None
        )
        plateau = False
        if self.last_trend_estimates is not None:
            plateau = (
                self.last_trend_estimates[
                    "weak_accuracy_slope"
                ].upper <= self.config.max_weak_accuracy_slope
            )
            if (
                self.config.recovery_primary_metric
                == "accuracy_and_loss"
            ):
                plateau = (
                    plateau
                    and self.last_trend_estimates[
                        "weak_quality_slope"
                    ].upper <= self.config.max_weak_quality_slope
                )
        self.futility_bad_checks = (
            self.futility_bad_checks + 1 if plateau else 0
        )
        self.last_optimistic_bounds = None

        hard_active = hard_violation
        compatibility_active = (
            self.safety_bad_checks >= self.config.safety_patience
        )
        reversal_active = (
            self.reversal_bad_checks >= self.config.reversal_patience
        )
        plateau_active = (
            self.futility_bad_checks >= self.config.plateau_patience
        )
        max_active = metrics.phase_epoch >= self.config.max_epochs
        self._remember_trigger("emergency_stop", hard_active, metrics.phase_epoch)
        self._remember_trigger(
            "compatibility_breach",
            compatibility_active,
            metrics.phase_epoch,
        )
        self._remember_trigger("trend_reversal", reversal_active, metrics.phase_epoch)
        self._remember_trigger("recovery_plateau", plateau_active, metrics.phase_epoch)
        self._remember_trigger("max_epochs", max_active, metrics.phase_epoch)

        stop_reason = None
        if hard_active:
            stop_reason = "emergency_stop"
        elif compatibility_active:
            stop_reason = "compatibility_breach"
        elif reversal_active:
            stop_reason = "trend_reversal"
        elif plateau_active:
            stop_reason = "recovery_plateau"
        elif max_active:
            stop_reason = "max_epochs"

        selected, selection_status = self.selection()
        decision = Phase3StopDecision(
            should_stop=stop_reason is not None,
            stop_reason=stop_reason,
            selection_status=selection_status if stop_reason else None,
            selected=selected if stop_reason else None,
            current=record,
            bad_checks=self.bad_checks,
            safety_bad_checks=self.safety_bad_checks,
        )
        if frozen is not None:
            return Phase3StopDecision(
                should_stop=True,
                stop_reason=frozen.stop_reason,
                selection_status=frozen.selection_status,
                selected=frozen.selected,
                current=record,
                bad_checks=self.bad_checks,
                safety_bad_checks=self.safety_bad_checks,
            )
        if decision.should_stop:
            self.final_decision = decision
        return decision

    def state_dict(self):
        state = super().state_dict()
        state["first_trigger_epochs"] = dict(self.first_trigger_epochs)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.first_trigger_epochs = {
            str(name): int(epoch)
            for name, epoch in state.get("first_trigger_epochs", {}).items()
        }


class Phase3LocalAccuracyStopper(Phase3RecoveryStopper):
    """Locally adaptive accuracy stopper without pre-Phase-3 drop limits.

    Decisions use paired correctness on validation proper.  Gradient
    diagnostics may confirm harm, but are never sufficient without weak
    validation-accuracy futility.
    """

    def __init__(self, config, baseline):
        if config.decision_rule != "local_accuracy":
            raise ValueError(
                "Phase3LocalAccuracyStopper requires local_accuracy"
            )
        super().__init__(config, baseline)
        self.target_bad_checks = 0
        self.pareto_bad_checks = 0
        self.futility_harm_bad_checks = 0
        self.pareto_frontier = []
        self.last_candidate = None
        self.local_selected = None
        self.futility_anchor = None
        self.last_diagnostics = None
        self.last_local_flags = {}

    def _record(self, metrics, checkpoint_path):
        record = super()._record(metrics, checkpoint_path)
        weak_recovered = (
            record.paired_estimates["weak_accuracy_gain"]["lower"]
            >= self.config.min_weak_accuracy_gain
        )
        finite = self._finite(record)
        return replace(
            record,
            is_feasible=bool(finite and weak_recovered),
            is_safe=bool(finite),
        )

    @staticmethod
    def _accuracies(record):
        return (
            record.metrics.weak_only.accuracy,
            record.metrics.full.accuracy,
            record.metrics.dominant_only.accuracy,
        )

    def _point_dominates(self, candidate, current):
        candidate_values = self._accuracies(candidate)
        current_values = self._accuracies(current)
        return (
            all(
                left >= right - self.config.min_delta
                for left, right in zip(candidate_values, current_values)
            )
            and any(
                left > right + self.config.min_delta
                for left, right in zip(candidate_values, current_values)
            )
        )

    def _dominance_estimates(self, candidate, current):
        candidate_correct = _correctness_arrays(candidate.metrics)
        current_correct = _correctness_arrays(current.metrics)
        return {
            name: self._estimate(
                candidate_correct[name] - current_correct[name]
            )
            for name in ("weak_only", "full", "dominant_only")
        }

    def _meaningfully_dominates(self, candidate, current):
        if not self._point_dominates(candidate, current):
            return False
        estimates = self._dominance_estimates(candidate, current)
        return any(
            estimate.lower > self.config.min_delta
            for estimate in estimates.values()
        )

    def _find_dominator(self, record):
        candidates = [
            candidate
            for candidate in self.pareto_frontier
            if self._meaningfully_dominates(candidate, record)
        ]
        if not candidates:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if _weak_recovery_better(
                candidate, best, self.config.min_delta
            ):
                best = candidate
        return best

    def _update_frontier(self, record):
        if any(
            self._point_dominates(candidate, record)
            for candidate in self.pareto_frontier
        ):
            return
        self.pareto_frontier = [
            candidate
            for candidate in self.pareto_frontier
            if not self._point_dominates(record, candidate)
        ]
        self.pareto_frontier.append(record)
        retention_limit = (
            self.config.trend_window + self.config.pareto_patience
        )
        self.pareto_frontier = sorted(
            self.pareto_frontier,
            key=lambda candidate: candidate.metrics.phase_epoch,
        )[-retention_limit:]

    @property
    def retained_checkpoint_paths(self):
        records = [
            *self.pareto_frontier,
            self.last_candidate,
            self.local_selected,
            self.futility_anchor,
        ]
        return {
            record.checkpoint_path
            for record in records
            if record is not None
        }

    def selection(self):
        if self.local_selected is not None:
            return self.local_selected, "local_selected"
        if self.last_candidate is not None:
            return self.last_candidate, "local_pareto_candidate"
        return None, "rollback_pre_phase3"

    def update(self, metrics, checkpoint_path, diagnostics=None):
        frozen = self.final_decision
        if frozen is not None and not self.config.shadow_continue_after_stop:
            return frozen

        self.evaluation_count += 1
        record = self._record(metrics, checkpoint_path)
        self.history.append(metrics)
        if diagnostics is not None:
            self.last_diagnostics = dict(diagnostics)
        enough_exposure = (
            metrics.phase_epoch >= self.config.min_epochs
            and self.evaluation_count
            >= self.config.minimum_exposure_evaluations
        )
        self.last_trend_estimates = (
            self._trend_estimates() if enough_exposure else None
        )

        dominator = self._find_dominator(record)
        trends = self.last_trend_estimates
        weak_futile = bool(
            trends is not None
            and trends["weak_accuracy_slope"].upper
            <= self.config.max_weak_accuracy_slope
        )
        accuracy_harm = bool(
            trends is not None
            and (
                trends["full_accuracy_slope"].upper < 0.0
                or trends["dominant_accuracy_slope"].upper < 0.0
            )
        )
        gradient_harm = False
        effective_diagnostics = self.last_diagnostics
        if effective_diagnostics is not None:
            cosines = [
                effective_diagnostics.get("shared_cosine_weak_dominant"),
                effective_diagnostics.get("shared_cosine_weak_full"),
            ]
            gradient_harm = any(
                value is not None
                and np.isfinite(float(value))
                and float(value) < self.config.gradient_conflict_threshold
                for value in cosines
            )

        estimates = record.paired_estimates
        target = bool(
            enough_exposure
            and trends is not None
            and dominator is None
            and estimates["weak_accuracy_gain"]["lower"]
            >= self.config.min_weak_accuracy_gain
            and estimates["weak_minus_dominant_accuracy"]["lower"] >= 0.0
            and trends["full_accuracy_slope"].upper >= 0.0
            and trends["dominant_accuracy_slope"].upper >= 0.0
        )
        pareto_reversal = bool(enough_exposure and dominator is not None)
        futility_with_harm = bool(
            enough_exposure
            and weak_futile
            and (accuracy_harm or gradient_harm)
        )

        self.target_bad_checks = (
            self.target_bad_checks + 1 if target else 0
        )
        self.pareto_bad_checks = (
            self.pareto_bad_checks + 1 if pareto_reversal else 0
        )
        if futility_with_harm and self.futility_harm_bad_checks == 0:
            self.futility_anchor = self.last_candidate
        self.futility_harm_bad_checks = (
            self.futility_harm_bad_checks + 1
            if futility_with_harm
            else 0
        )

        if dominator is None and not accuracy_harm:
            self.last_candidate = record
            self.best_feasible = record
            self.best_safe = record
        self._update_frontier(record)

        hard_active = self._hard_violation(record)
        target_active = (
            self.target_bad_checks >= self.config.target_patience
        )
        pareto_active = (
            self.pareto_bad_checks >= self.config.pareto_patience
        )
        futility_active = (
            self.futility_harm_bad_checks
            >= self.config.futility_harm_patience
        )
        max_active = metrics.phase_epoch >= self.config.max_epochs
        self._remember_trigger("emergency_stop", hard_active, metrics.phase_epoch)
        self._remember_trigger("target_reached", target_active, metrics.phase_epoch)
        self._remember_trigger("pareto_reversal", pareto_active, metrics.phase_epoch)
        self._remember_trigger(
            "futility_with_harm", futility_active, metrics.phase_epoch
        )
        self._remember_trigger("max_epochs", max_active, metrics.phase_epoch)

        stop_reason = None
        selected = None
        if hard_active:
            stop_reason = "emergency_stop"
        elif target_active:
            stop_reason = "target_reached"
            selected = record
        elif pareto_active:
            stop_reason = "pareto_reversal"
            selected = dominator
        elif futility_active:
            stop_reason = "futility_with_harm"
            selected = self.futility_anchor or self.last_candidate
        elif max_active:
            stop_reason = "max_epochs"
            selected = self.last_candidate

        if stop_reason is not None and frozen is None:
            self.local_selected = selected
        selected, selection_status = self.selection()
        self.last_local_flags = {
            "target": target,
            "pareto_reversal": pareto_reversal,
            "weak_futile": weak_futile,
            "accuracy_harm": accuracy_harm,
            "gradient_harm": gradient_harm,
            "futility_with_harm": futility_with_harm,
        }
        self.reversal_bad_checks = self.pareto_bad_checks
        self.futility_bad_checks = self.futility_harm_bad_checks
        self.safety_bad_checks = 0
        self.last_optimistic_bounds = None
        decision = Phase3StopDecision(
            should_stop=stop_reason is not None,
            stop_reason=stop_reason,
            selection_status=selection_status if stop_reason else None,
            selected=selected if stop_reason else None,
            current=record,
            bad_checks=self.futility_harm_bad_checks,
            safety_bad_checks=0,
        )
        if frozen is not None:
            return Phase3StopDecision(
                should_stop=True,
                stop_reason=frozen.stop_reason,
                selection_status=frozen.selection_status,
                selected=frozen.selected,
                current=record,
                bad_checks=self.futility_harm_bad_checks,
                safety_bad_checks=0,
            )
        if decision.should_stop:
            self.final_decision = decision
        return decision

    def state_dict(self):
        state = super().state_dict()
        state["local_accuracy"] = {
            "target_bad_checks": self.target_bad_checks,
            "pareto_bad_checks": self.pareto_bad_checks,
            "futility_harm_bad_checks": self.futility_harm_bad_checks,
            "pareto_frontier": [
                record.state_dict() for record in self.pareto_frontier
            ],
            "last_candidate": (
                self.last_candidate.state_dict()
                if self.last_candidate is not None
                else None
            ),
            "local_selected": (
                self.local_selected.state_dict()
                if self.local_selected is not None
                else None
            ),
            "futility_anchor": (
                self.futility_anchor.state_dict()
                if self.futility_anchor is not None
                else None
            ),
            "last_diagnostics": self.last_diagnostics,
            "last_local_flags": dict(self.last_local_flags),
        }
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        local = state.get("local_accuracy", {})
        self.target_bad_checks = int(local.get("target_bad_checks", 0))
        self.pareto_bad_checks = int(local.get("pareto_bad_checks", 0))
        self.futility_harm_bad_checks = int(
            local.get("futility_harm_bad_checks", 0)
        )
        self.pareto_frontier = [
            InterventionCheckpointRecord.from_state_dict(item)
            for item in local.get("pareto_frontier", [])
        ]
        for name in ("last_candidate", "local_selected", "futility_anchor"):
            value = local.get(name)
            setattr(
                self,
                name,
                (
                    InterventionCheckpointRecord.from_state_dict(value)
                    if value is not None
                    else None
                ),
            )
        self.last_diagnostics = local.get("last_diagnostics")
        self.last_local_flags = dict(local.get("last_local_flags", {}))


@dataclass(frozen=True)
class Phase4CheckpointRecord:
    metrics: ModalityEvaluationResult
    checkpoint_path: str

    def state_dict(self):
        return {
            "metrics": self.metrics.state_dict(),
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            metrics=ModalityEvaluationResult.from_state_dict(state["metrics"]),
            checkpoint_path=str(state["checkpoint_path"]),
        )


class Phase4CheckpointSelector:
    def __init__(self, max_epochs, intervention_epochs):
        self.max_epochs = int(max_epochs)
        self.intervention_epochs = int(intervention_epochs)
        if (
            self.max_epochs < 0
            or not 0 <= self.intervention_epochs <= self.max_epochs
        ):
            raise ValueError("invalid phase 4/intervention epoch budget")
        self.best_full = None
        self.best_budget_matched = None
        self.best_full_accuracy = None
        self.best_budget_matched_accuracy = None

    @staticmethod
    def _better_loss(candidate, incumbent):
        if incumbent is None:
            return True
        return (
            candidate.metrics.full.loss,
            -candidate.metrics.full.accuracy,
            candidate.metrics.phase_epoch,
        ) < (
            incumbent.metrics.full.loss,
            -incumbent.metrics.full.accuracy,
            incumbent.metrics.phase_epoch,
        )

    @staticmethod
    def _better_accuracy(candidate, incumbent):
        if incumbent is None:
            return True
        return (
            -candidate.metrics.full.accuracy,
            candidate.metrics.full.loss,
            candidate.metrics.phase_epoch,
        ) < (
            -incumbent.metrics.full.accuracy,
            incumbent.metrics.full.loss,
            incumbent.metrics.phase_epoch,
        )

    def add(self, record: Phase4CheckpointRecord):
        if record.metrics.phase_epoch > self.max_epochs:
            raise ValueError("phase 4 checkpoint exceeds the configured budget")
        if self._better_loss(record, self.best_full):
            self.best_full = record
        if self._better_accuracy(record, self.best_full_accuracy):
            self.best_full_accuracy = record
        budget = self.max_epochs - self.intervention_epochs
        if record.metrics.phase_epoch <= budget:
            if self._better_loss(record, self.best_budget_matched):
                self.best_budget_matched = record
            if self._better_accuracy(
                record, self.best_budget_matched_accuracy
            ):
                self.best_budget_matched_accuracy = record

    def best_full_for(self, metric):
        if metric == "loss":
            return self.best_full
        if metric == "accuracy":
            return self.best_full_accuracy
        raise ValueError("unknown phase 4 primary metric")

    def best_budget_for(self, metric):
        if metric == "loss":
            return self.best_budget_matched
        if metric == "accuracy":
            return self.best_budget_matched_accuracy
        raise ValueError("unknown phase 4 primary metric")

    def state_dict(self):
        return {
            "max_epochs": self.max_epochs,
            "intervention_epochs": self.intervention_epochs,
            "best_full": (
                self.best_full.state_dict()
                if self.best_full is not None
                else None
            ),
            "best_budget_matched": (
                self.best_budget_matched.state_dict()
                if self.best_budget_matched is not None
                else None
            ),
            "best_full_accuracy": (
                self.best_full_accuracy.state_dict()
                if self.best_full_accuracy is not None
                else None
            ),
            "best_budget_matched_accuracy": (
                self.best_budget_matched_accuracy.state_dict()
                if self.best_budget_matched_accuracy is not None
                else None
            ),
        }

    def load_state_dict(self, state):
        if (
            int(state["max_epochs"]) != self.max_epochs
            or int(state["intervention_epochs"])
            != self.intervention_epochs
        ):
            raise ValueError("phase 4 selection budget changed across resume")
        self.best_full = (
            Phase4CheckpointRecord.from_state_dict(state["best_full"])
            if state.get("best_full") is not None
            else None
        )
        self.best_budget_matched = (
            Phase4CheckpointRecord.from_state_dict(
                state["best_budget_matched"]
            )
            if state.get("best_budget_matched") is not None
            else None
        )
        self.best_full_accuracy = (
            Phase4CheckpointRecord.from_state_dict(
                state["best_full_accuracy"]
            )
            if state.get("best_full_accuracy") is not None
            else self.best_full
        )
        self.best_budget_matched_accuracy = (
            Phase4CheckpointRecord.from_state_dict(
                state["best_budget_matched_accuracy"]
            )
            if state.get("best_budget_matched_accuracy") is not None
            else self.best_budget_matched
        )
