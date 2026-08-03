"""Replay unimodal recovery targets with paired simultaneous confidence intervals.

The replay uses aligned per-example validation correctness saved in a Phase-3
trajectory.  Unimodal reference accuracies are treated as fixed constants; the
resulting interval is therefore conditional on the selected reference models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.python_new.replay_unimodal_recovery_fraction import load_trajectory


@dataclass(frozen=True)
class PairedGapEstimate:
    mean: float
    lower: float
    upper: float
    standard_error: float
    n_examples: int
    effective_alpha: float


def paired_recovery_gap_estimate(
    *,
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    unimodal_left_accuracy: float,
    unimodal_right_accuracy: float,
    target_fraction: float,
    confidence_level: float,
    max_looks: int,
    family_size: int,
) -> PairedGapEstimate:
    """Estimate the paired gap above a requested recovered-deficit fraction.

    For target q, the per-example contribution is

      w_e / U_R - (1-q) w_0 / U_R - q d_0 / U_L,

    whose mean is zero exactly at the requested normalized recovery target.
    A Bonferroni correction covers every scheduled look and every jointly
    replayed target fraction.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if max_looks <= 0 or family_size <= 0:
        raise ValueError("max_looks and family_size must be positive")
    if unimodal_left_accuracy <= 0.0 or unimodal_right_accuracy <= 0.0:
        raise ValueError("unimodal reference accuracies must be positive")

    baseline_correct = baseline_metrics["per_example_correctness"]
    current_correct = current_metrics["per_example_correctness"]
    weak_current = np.asarray(current_correct["weak_only"], dtype=np.float64)
    weak_baseline = np.asarray(baseline_correct["weak_only"], dtype=np.float64)
    dominant_baseline = np.asarray(
        baseline_correct["dominant_only"], dtype=np.float64
    )
    if not (
        weak_current.shape == weak_baseline.shape == dominant_baseline.shape
    ):
        raise ValueError("paired correctness arrays must have identical shapes")
    if weak_current.ndim != 1 or weak_current.size < 2:
        raise ValueError("paired correctness arrays must be one-dimensional, n >= 2")

    contributions = (
        weak_current / unimodal_right_accuracy
        - (1.0 - target_fraction) * weak_baseline / unimodal_right_accuracy
        - target_fraction * dominant_baseline / unimodal_left_accuracy
    )
    if not np.isfinite(contributions).all():
        raise ValueError("paired recovery contributions must be finite")

    n_examples = int(contributions.size)
    mean = float(contributions.mean())
    standard_error = float(contributions.std(ddof=1) / math.sqrt(n_examples))
    effective_alpha = (1.0 - confidence_level) / (max_looks * family_size)
    quantile = NormalDist().inv_cdf(1.0 - effective_alpha / 2.0)
    half_width = quantile * standard_error
    return PairedGapEstimate(
        mean=mean,
        lower=mean - half_width,
        upper=mean + half_width,
        standard_error=standard_error,
        n_examples=n_examples,
        effective_alpha=effective_alpha,
    )


def select_recovery_checkpoint(
    estimates: Sequence[dict[str, Any]], *, confirmations: int
) -> dict[str, Any]:
    """Select the first of consecutive lower-CI crossings, else best lower CI."""

    if confirmations <= 0:
        raise ValueError("confirmations must be positive")
    if not estimates:
        raise ValueError("at least one estimate is required")

    streak: list[dict[str, Any]] = []
    for record in estimates:
        if float(record["ci_lower"]) >= 0.0:
            streak.append(record)
            if len(streak) >= confirmations:
                selected = streak[0]
                return {
                    "selection_status": "confirmed_ci_crossing",
                    "selected": selected,
                    "first_crossing_epoch": int(selected["phase_epoch"]),
                    "confirmation_epoch": int(record["phase_epoch"]),
                }
        else:
            streak = []

    best = max(
        estimates,
        key=lambda record: (
            float(record["ci_lower"]),
            float(record["gap_mean"]),
            float(record["weak_only_accuracy"]),
            -float(record["weak_only_loss"]),
            -int(record["phase_epoch"]),
        ),
    )
    return {
        "selection_status": "fallback_best_lower_ci",
        "selected": best,
        "first_crossing_epoch": None,
        "confirmation_epoch": None,
    }


def replay_trajectory(
    trajectory_path: Path,
    *,
    seed: int,
    target_fractions: Sequence[float],
    confidence_level: float,
    confirmations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory = load_trajectory(trajectory_path)
    if not trajectory:
        raise ValueError(f"empty trajectory: {trajectory_path}")
    baseline = trajectory[0]
    baseline_metrics = baseline["metrics"]
    references = baseline["unimodal_references"]
    unimodal_left_accuracy = float(references["left"]["validation_accuracy"])
    unimodal_right_accuracy = float(references["right"]["validation_accuracy"])
    max_looks = len(trajectory)
    family_size = len(target_fractions)

    all_estimates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for target_fraction in target_fractions:
        threshold_estimates: list[dict[str, Any]] = []
        for trajectory_record in trajectory:
            metrics = trajectory_record["metrics"]
            estimate = paired_recovery_gap_estimate(
                baseline_metrics=baseline_metrics,
                current_metrics=metrics,
                unimodal_left_accuracy=unimodal_left_accuracy,
                unimodal_right_accuracy=unimodal_right_accuracy,
                target_fraction=target_fraction,
                confidence_level=confidence_level,
                max_looks=max_looks,
                family_size=family_size,
            )
            checkpoint_path = trajectory_record.get("checkpoint_path")
            row = {
                "seed": seed,
                "target_fraction": target_fraction,
                "phase_epoch": int(trajectory_record["phase_epoch"]),
                "checkpoint_path": checkpoint_path,
                "weak_only_accuracy": float(metrics["weak_only"]["accuracy"]),
                "weak_only_loss": float(metrics["weak_only"]["loss"]),
                "gap_mean": estimate.mean,
                "ci_lower": estimate.lower,
                "ci_upper": estimate.upper,
                "standard_error": estimate.standard_error,
                "effective_alpha": estimate.effective_alpha,
                "n_examples": estimate.n_examples,
                "ci_crossing": estimate.lower >= 0.0,
            }
            threshold_estimates.append(row)
            all_estimates.append(row)

        selection = select_recovery_checkpoint(
            threshold_estimates, confirmations=confirmations
        )
        selected = selection.pop("selected")
        if not selected.get("checkpoint_path"):
            raise ValueError(
                "selected trajectory record has no checkpoint path: "
                f"seed={seed}, target={target_fraction}"
            )
        selections.append(
            {
                "seed": seed,
                "trajectory_path": str(trajectory_path),
                "target_fraction": target_fraction,
                "confidence_level": confidence_level,
                "max_looks": max_looks,
                "family_size": family_size,
                "confirmations": confirmations,
                "unimodal_left_accuracy": unimodal_left_accuracy,
                "unimodal_right_accuracy": unimodal_right_accuracy,
                **selection,
                "selected_epoch": int(selected["phase_epoch"]),
                "selected_checkpoint_path": selected["checkpoint_path"],
                "selected_gap_mean": float(selected["gap_mean"]),
                "selected_ci_lower": float(selected["ci_lower"]),
                "selected_ci_upper": float(selected["ci_upper"]),
                "reference_uncertainty_included": False,
            }
        )
    return all_estimates, selections


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="Phase-3 trajectory; repeat once per seed",
    )
    parser.add_argument(
        "--target-fraction", action="append", type=float, default=None
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = args.target_fraction or [0.99, 1.0]
    all_estimates: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    for specification in args.trajectory:
        seed_text, path_text = specification.split("=", 1)
        estimates, selections = replay_trajectory(
            Path(path_text),
            seed=int(seed_text),
            target_fractions=targets,
            confidence_level=args.confidence_level,
            confirmations=args.confirmations,
        )
        all_estimates.extend(estimates)
        all_selections.extend(selections)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired_gap_trajectory.csv", all_estimates)
    _write_csv(args.output_dir / "selected_checkpoints.csv", all_selections)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "paired_normal_ci_bonferroni_repeated_looks",
                "confidence_level": args.confidence_level,
                "target_fractions": targets,
                "confirmations": args.confirmations,
                "reference_uncertainty_included": False,
                "selections": all_selections,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_selections, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
