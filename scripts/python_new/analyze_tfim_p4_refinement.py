#!/usr/bin/env python3
"""Analyze P4=200 recovery refinement against clean accuracy gold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.python_new.tfim_analysis_common import (
    FEATURES,
    probe_features,
    spearman,
)


PREFIX_EPOCHS = (5, 8, 11, 14)
FULL_EPOCHS = (5, 8, 11, 14, 17)
FULL_GAP_EQUIVALENCE_TOLERANCE = 0.01


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="JOB_ID:SEED:E3",
    )
    parser.add_argument(
        "--slurm-log-dir", action="append", type=Path, required=True
    )
    parser.add_argument("--accuracy-gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_run(value):
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"invalid --run value: {value!r}")
    job_id, seed, e3 = fields
    return job_id, int(seed), int(e3)


def _find_log(log_dir, job_id):
    matches = sorted(log_dir.glob(f"*-{job_id}.out"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one Slurm log for {job_id}, found {matches}"
        )
    return matches[0]


def _find_log_in_directories(log_dirs, job_id):
    matches = sorted(
        path
        for log_dir in log_dirs
        for path in log_dir.glob(f"*-{job_id}.out")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one Slurm log for {job_id}, found {matches}"
        )
    return matches[0]


def _phase_summary(log_path, phase):
    matches = []
    marker = "Phase summary: "
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            payload = json.loads(line.split(marker, 1)[1])
            if payload.get("phase") == phase:
                matches.append(payload)
    if len(matches) != 1:
        raise ValueError(
            f"expected one Phase-{phase} summary in {log_path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _phase4_summary(log_path):
    return _phase_summary(log_path, 4)


def _tfim_trace(summary, *, phase=4, epochs=FULL_EPOCHS):
    checkpoint_value = summary.get("primary_checkpoint") or summary.get(
        "selected_checkpoint"
    )
    if checkpoint_value is None:
        raise ValueError(
            "phase summary has neither primary_checkpoint nor "
            "selected_checkpoint"
        )
    checkpoint = Path(checkpoint_value)
    trace_path = checkpoint.parent.parent / "trace_fim_train.jsonl"
    by_epoch = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record.get("phase", -1)) != int(phase):
                continue
            epoch = int(record["phase_epoch"])
            if epoch in by_epoch:
                raise ValueError(f"duplicate TFIM epoch {epoch} in {trace_path}")
            metrics = record["metrics"]
            by_epoch[epoch] = {
                "trace_left": float(
                    metrics["trace_fim_overall_train/proper_trace1_weight"]
                ),
                "trace_right": float(
                    metrics["trace_fim_overall_train/proper_trace2_weight"]
                ),
                "ratio": float(
                    metrics[
                        "trace_fim_overall_train/proper_ratio_left_to_right_weight"
                    ]
                ),
            }
    expected_epochs = tuple(epochs)
    if tuple(sorted(by_epoch)) != expected_epochs:
        raise ValueError(
            f"expected Phase-{phase} TFIM epochs {expected_epochs} "
            f"in {trace_path}, "
            f"found {tuple(sorted(by_epoch))}"
        )
    return trace_path, by_epoch


def _load_accuracy_gold(path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["seed"] == "mean":
                continue
            seed = int(row["seed"])
            if seed in result:
                raise ValueError(f"duplicate accuracy gold for seed {seed}")
            result[seed] = {
                "full": float(row["full_accuracy"]),
                "dominant_only": float(row["dominant_only_accuracy"]),
                "weak_only": float(row["weak_only_accuracy"]),
            }
    if not result:
        raise ValueError("accuracy gold CSV contains no per-seed rows")
    return result


def _add_accuracy_gold_distances(row, target):
    row["gold_full_accuracy"] = target["full"]
    row["gold_dominant_accuracy"] = target["dominant_only"]
    row["gold_weak_accuracy"] = target["weak_only"]
    row["full_accuracy_gap_abs"] = abs(
        row["validation_full_accuracy"] - target["full"]
    )
    row["dominant_accuracy_gap_abs"] = abs(
        row["validation_dominant_accuracy"] - target["dominant_only"]
    )
    row["weak_accuracy_gap_abs"] = abs(
        row["validation_weak_accuracy"] - target["weak_only"]
    )
    row["branch_accuracy_gap_mean_abs"] = 0.5 * (
        row["dominant_accuracy_gap_abs"] + row["weak_accuracy_gap_abs"]
    )


def _secondary_recovery_rank_key(row):
    return (
        row["branch_accuracy_gap_mean_abs"],
        row["dominant_accuracy_gap_abs"],
        row["weak_accuracy_gap_abs"],
        row["full_accuracy_gap_abs"],
        row["e3"],
    )


def _select_best_recovery(
    rows, *, full_gap_tolerance=FULL_GAP_EQUIVALENCE_TOLERANCE
):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot select recovery from an empty candidate set")
    best_full_gap = min(row["full_accuracy_gap_abs"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["full_accuracy_gap_abs"]
        <= best_full_gap + full_gap_tolerance + 1e-12
    ]
    return min(eligible, key=_secondary_recovery_rank_key)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(log_dirs, specification):
    job_id, seed, e3 = _parse_run(specification)
    log_path = _find_log_in_directories(log_dirs, job_id)
    summary = _phase4_summary(log_path)
    if int(summary["executed_epochs"]) != 200:
        raise ValueError(f"job {job_id} is not a complete P4=200 run")
    trace_path, trace = _tfim_trace(summary)
    ratios = [trace[epoch]["ratio"] for epoch in FULL_EPOCHS]
    prefix_features = probe_features(PREFIX_EPOCHS, ratios[:4])
    full_features = probe_features(FULL_EPOCHS, ratios)
    metrics = summary["best_full_accuracy_metrics"]
    row = {
        "job_id": job_id,
        "seed": seed,
        "e3": e3,
        "selected_p4_epoch": int(metrics["phase_epoch"]),
        "validation_full_accuracy": float(metrics["full_accuracy"]),
        "validation_dominant_accuracy": float(
            metrics["dominant_only_accuracy"]
        ),
        "validation_weak_accuracy": float(metrics["weak_only_accuracy"]),
        "slurm_log": log_path.name,
        "tfim_trace_artifact": trace_path.name,
    }
    for epoch in FULL_EPOCHS:
        row[f"trace_left_e{epoch}"] = trace[epoch]["trace_left"]
        row[f"trace_right_e{epoch}"] = trace[epoch]["trace_right"]
        row[f"ratio_e{epoch}"] = trace[epoch]["ratio"]
    for feature in FEATURES:
        row[f"prefix4_{feature}"] = prefix_features[feature]
        row[f"full5_{feature}"] = full_features[feature]
    row["slope_sign_agrees"] = (
        math.copysign(1.0, prefix_features["slope_log_ratio"])
        == math.copysign(1.0, full_features["slope_log_ratio"])
    )
    return row


def _stability(rows):
    records = []
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        for feature in FEATURES:
            prefix = [row[f"prefix4_{feature}"] for row in selected]
            full = [row[f"full5_{feature}"] for row in selected]
            records.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "candidate_count": len(selected),
                    "spearman_prefix4_vs_full5": (
                        spearman(prefix, full)
                        if len(selected) >= 2
                        else float("nan")
                    ),
                    "max_absolute_change": max(
                        abs(left - right)
                        for left, right in zip(prefix, full)
                    ),
                }
            )
    return records


def _plot(rows, output_dir):
    seeds = sorted({row["seed"] for row in rows})
    figure, axes = plt.subplots(
        len(seeds), 1, figsize=(9, 3.5 * len(seeds)), squeeze=False
    )
    for axis, seed in zip(axes[:, 0], seeds):
        for row in sorted(
            (item for item in rows if item["seed"] == seed),
            key=lambda item: item["e3"],
        ):
            axis.plot(
                FULL_EPOCHS,
                [row[f"ratio_e{epoch}"] for epoch in FULL_EPOCHS],
                marker="o",
                label=(
                    f"e3={row['e3']}, val={row['validation_full_accuracy']:.4f}"
                ),
            )
        axis.axvline(14, color="black", linestyle="--", alpha=0.35)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("lokalna epoka Phase 4")
        axis.set_ylabel("Tr(F_L) / Tr(F_R)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"tfim_p4_refinement_by_seed.{suffix}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(figure)


def main():
    args = _arguments()
    accuracy_gold = _load_accuracy_gold(args.accuracy_gold_csv)
    rows = sorted(
        (_load_run(args.slurm_log_dir, item) for item in args.run),
        key=lambda row: (row["seed"], row["e3"]),
    )
    keys = [(row["seed"], row["e3"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed/e3 run specification")
    observed_seeds = {row["seed"] for row in rows}
    if observed_seeds != set(accuracy_gold):
        raise ValueError(
            "run/gold seed mismatch: "
            f"runs={sorted(observed_seeds)}, gold={sorted(accuracy_gold)}"
        )
    for row in rows:
        _add_accuracy_gold_distances(row, accuracy_gold[row["seed"]])
    stability = _stability(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "tfim_p4_refinement_runs.csv", rows)
    _write_csv(args.output_dir / "tfim_prefix4_vs_full5_stability.csv", stability)
    with (args.output_dir / "tfim_p4_refinement_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "prefix_epochs": PREFIX_EPOCHS,
                "full_epochs": FULL_EPOCHS,
                "outcome": (
                    "hierarchical distance to per-seed clean P2=200 "
                    "validation_proper accuracy gold"
                ),
                "selection_order": [
                    "full-gap equivalence band: minimum absolute full "
                    "accuracy gap + 0.01",
                    "minimum mean absolute dominant/weak accuracy gap "
                    "within the band",
                    "absolute dominant accuracy gap",
                    "absolute weak accuracy gap",
                    "absolute full accuracy gap",
                    "earlier e3",
                ],
                "full_gap_equivalence_tolerance": (
                    FULL_GAP_EQUIVALENCE_TOLERANCE
                ),
                "best_recovery_by_seed": {
                    str(seed): _select_best_recovery(
                        row for row in rows if row["seed"] == seed
                    )["e3"]
                    for seed in sorted({row["seed"] for row in rows})
                },
                "all_slope_signs_agree": all(
                    row["slope_sign_agrees"] for row in rows
                ),
                "test_metrics_used": False,
                "refinement_goal": "one exact Phase-3 epoch per seed",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    _plot(rows, args.output_dir)


if __name__ == "__main__":
    main()
