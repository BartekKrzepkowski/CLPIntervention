#!/usr/bin/env python3
"""Plot local Phase-3 stopper selections against downstream Phase-4 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "analysis/results/phase4_milestone_p1_40.csv"
DEFAULT_REPLAY = (
    REPO_ROOT / "analysis/results/phase3_local_accuracy_replay_p1_40.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs/figures/phase3_local_stopper_phase4_validation"
)
EXACT_SELECTIONS = {83: 30, 184: 35, 285: 30}


def _read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["e3"] = int(row["e3"])
        for key in (
            "val_full_accuracy",
            "val_dominant_accuracy",
            "val_weak_accuracy",
        ):
            row[key] = float(row[key])
    return rows


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = _parse_args()
    rows = _read_rows(args.results)
    replay = json.loads(args.replay.read_text(encoding="utf-8"))["runs"]
    seeds = sorted({row["seed"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=False)
    seed_axes = axes.flat[:3]

    for axis, seed in zip(seed_axes, seeds):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["e3"],
        )
        e3 = np.asarray([row["e3"] for row in seed_rows])
        accuracy = np.asarray(
            [row["val_full_accuracy"] for row in seed_rows]
        ) * 100.0
        best_index = int(np.argmax(accuracy))
        replay_selected = int(
            replay[str(seed)]["selected_epoch_point_estimate"]
        )
        selected = EXACT_SELECTIONS.get(seed, replay_selected)
        selection_source = (
            "paired CI" if seed in EXACT_SELECTIONS else "point replay"
        )
        reason = replay[str(seed)]["stop_reason"]
        axis.plot(e3, accuracy, color="#275dad", marker="o", linewidth=2)
        axis.scatter(
            [e3[best_index]],
            [accuracy[best_index]],
            color="#2a9d55",
            marker="*",
            s=170,
            zorder=4,
            label="best tested milestone",
        )
        axis.axvline(
            selected,
            color="#d62828",
            linestyle="--",
            linewidth=2,
            label=f"{selection_source} selects e3={selected}",
        )
        axis.set_title(f"seed {seed}: {reason}")
        axis.set_ylabel("P4-selected validation full accuracy [%]")
        axis.set_xticks(e3)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    aggregate = axes.flat[3]
    e3_values = sorted({row["e3"] for row in rows})
    styles = (
        ("val_full_accuracy", "full", "#275dad", "o"),
        ("val_dominant_accuracy", "dominant-only", "#d62728", "s"),
        ("val_weak_accuracy", "weak-only", "#2a9d55", "^"),
    )
    for metric, label, color, marker in styles:
        means = []
        deviations = []
        for epoch in e3_values:
            values = np.asarray(
                [row[metric] for row in rows if row["e3"] == epoch]
            ) * 100.0
            means.append(float(values.mean()))
            deviations.append(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        aggregate.errorbar(
            e3_values,
            means,
            yerr=deviations,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=3,
        )
    aggregate.set_title("branch trade-off after Phase 4: mean ± SD")
    aggregate.set_ylabel("validation accuracy [%]")
    aggregate.set_xticks(e3_values)
    aggregate.grid(alpha=0.25)
    aggregate.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Phase-3 exposure e3 [epochs]")
    figure.suptitle(
        "P1=40/P2=200: local Phase-3 stopper versus downstream recovery",
        fontsize=13,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
