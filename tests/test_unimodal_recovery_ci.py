from __future__ import annotations

import numpy as np

from scripts.python_new.replay_unimodal_recovery_ci import (
    paired_recovery_gap_estimate,
    select_recovery_checkpoint,
)


def _metrics(weak, dominant):
    return {
        "per_example_correctness": {
            "weak_only": list(weak),
            "dominant_only": list(dominant),
        }
    }


def test_paired_gap_mean_matches_normalized_recovery_formula():
    baseline = _metrics([0, 0, 1, 1], [1, 1, 1, 0])
    current = _metrics([1, 1, 1, 0], [0, 0, 0, 0])
    estimate = paired_recovery_gap_estimate(
        baseline_metrics=baseline,
        current_metrics=current,
        unimodal_left_accuracy=0.75,
        unimodal_right_accuracy=0.75,
        target_fraction=1.0,
        confidence_level=0.95,
        max_looks=2,
        family_size=2,
    )
    expected = np.mean(np.array([1, 1, 1, 0]) / 0.75) - np.mean(
        np.array([1, 1, 1, 0]) / 0.75
    )
    assert estimate.mean == expected
    assert estimate.lower <= estimate.mean <= estimate.upper
    assert estimate.effective_alpha == (1.0 - 0.95) / 4


def test_selector_returns_first_of_confirmed_crossing_streak():
    records = [
        {"phase_epoch": 4, "ci_lower": -0.1, "gap_mean": 0.0,
         "weak_only_accuracy": 0.4, "weak_only_loss": 1.0},
        {"phase_epoch": 8, "ci_lower": 0.01, "gap_mean": 0.02,
         "weak_only_accuracy": 0.5, "weak_only_loss": 0.9},
        {"phase_epoch": 12, "ci_lower": 0.02, "gap_mean": 0.03,
         "weak_only_accuracy": 0.6, "weak_only_loss": 0.8},
    ]
    selection = select_recovery_checkpoint(records, confirmations=2)
    assert selection["selection_status"] == "confirmed_ci_crossing"
    assert selection["selected"]["phase_epoch"] == 8
    assert selection["confirmation_epoch"] == 12


def test_selector_resets_streak_and_falls_back_to_best_lower_bound():
    records = [
        {"phase_epoch": 4, "ci_lower": 0.01, "gap_mean": 0.02,
         "weak_only_accuracy": 0.5, "weak_only_loss": 0.9},
        {"phase_epoch": 8, "ci_lower": -0.02, "gap_mean": 0.03,
         "weak_only_accuracy": 0.6, "weak_only_loss": 0.8},
        {"phase_epoch": 12, "ci_lower": -0.01, "gap_mean": 0.04,
         "weak_only_accuracy": 0.7, "weak_only_loss": 0.7},
    ]
    selection = select_recovery_checkpoint(records, confirmations=2)
    assert selection["selection_status"] == "fallback_best_lower_ci"
    assert selection["selected"]["phase_epoch"] == 4
