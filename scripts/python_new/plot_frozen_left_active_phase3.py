#!/usr/bin/env python3
"""Plot validation trajectories for the frozen-left-active Phase-3 control."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = (
    ("full", "full", "#275dad", "o"),
    ("dominant_only", "dominant-only", "#d62828", "s"),
    ("weak_only", "weak-only", "#2a9d55", "^"),
)


def _series(value):
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("series must have the form SEED=PATH")
    return int(seed), Path(path)


def _load(seed, path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            metrics = record["metrics"]
            row = {"seed": seed, "phase_epoch": int(record["phase_epoch"])}
            for mode, _, _, _ in MODES:
                row[f"{mode}_accuracy"] = float(metrics[mode]["accuracy"])
                row[f"{mode}_loss"] = float(metrics[mode]["loss"])
            rows.append(row)
    by_epoch = {row["phase_epoch"]: row for row in rows}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _aggregate_plot(rows, metric, ylabel, output):
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    epochs = sorted({row["phase_epoch"] for row in rows})
    for mode, label, color, marker in MODES:
        means, deviations = [], []
        for epoch in epochs:
            values = np.asarray(
                [row[f"{mode}_{metric}"] for row in rows if row["phase_epoch"] == epoch]
            )
            means.append(values.mean())
            deviations.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        scale = 100.0 if metric == "accuracy" else 1.0
        axis.errorbar(
            epochs,
            np.asarray(means) * scale,
            yerr=np.asarray(deviations) * scale,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=3,
        )
    axis.set_xlabel("Phase-3 epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Frozen-left-active Phase 3: validation {metric}, mean ± SD")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _per_seed_accuracy(rows, output):
    seeds = sorted({row["seed"] for row in rows})
    figure, axes = plt.subplots(1, len(seeds), figsize=(15, 4.5), sharey=True)
    for axis, seed in zip(np.atleast_1d(axes), seeds):
        selected = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["phase_epoch"],
        )
        epochs = [row["phase_epoch"] for row in selected]
        for mode, label, color, marker in MODES:
            axis.plot(
                epochs,
                [100.0 * row[f"{mode}_accuracy"] for row in selected],
                label=label,
                color=color,
                marker=marker,
            )
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("Phase-3 epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation accuracy [%]")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Frozen-left-active Phase 3: per-seed trajectories")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", type=_series, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed, path in args.series:
        rows.extend(_load(seed, path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frozen_left_active_phase3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _aggregate_plot(
        rows,
        "accuracy",
        "Validation accuracy [%]",
        args.output_dir / "frozen_left_active_validation_accuracy",
    )
    _aggregate_plot(
        rows,
        "loss",
        "Validation loss",
        args.output_dir / "frozen_left_active_validation_loss",
    )
    _per_seed_accuracy(
        rows, args.output_dir / "frozen_left_active_validation_accuracy_per_seed"
    )


if __name__ == "__main__":
    main()
