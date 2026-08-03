#!/usr/bin/env python3
"""Plot the validation-only Phase 3/4 sweep from scalar audit exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = (
    REPO_ROOT / "analysis/results/phase3_phase4_grid_2026-07-29.csv"
)
DEFAULT_GOLD = REPO_ROOT / "analysis/results/phase2_minimal_exposure_p1_1_2026-07-29.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/figures"


def read_numeric_csv(path: Path, text_columns: set[str]):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key, value in row.items():
            if key not in text_columns:
                row[key] = float(value)
    return rows


def grouped(rows, key):
    values = sorted({int(row[key]) for row in rows})
    return values, {
        value: [row for row in rows if int(row[key]) == value]
        for value in values
    }


def mean_sd(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def save_figure(figure, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_recovery_accuracy(grid_rows, gold_rows, output_dir):
    e3_values, by_e3 = grouped(grid_rows, "e3")
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    styles = (
        ("max_accuracy", "Pełny budżet Phase 4", "#275dad", "o"),
        (
            "budget_max_accuracy",
            "Budżet wyrównany: Phase 4 ≤ 200−e3",
            "#e07a1f",
            "s",
        ),
        (
            "accuracy_at_best_loss",
            "Checkpoint minimum validation loss",
            "#6a4c93",
            "^",
        ),
    )
    for metric, label, color, marker in styles:
        means, deviations = zip(
            *(mean_sd(by_e3[e3], metric) for e3 in e3_values)
        )
        axis.errorbar(
            e3_values,
            np.asarray(means) * 100,
            yerr=np.asarray(deviations) * 100,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=4,
        )
    gold = np.asarray(
        [row["max_accuracy"] for row in gold_rows], dtype=np.float64
    )
    gold_mean = gold.mean() * 100
    gold_sd = gold.std(ddof=1) * 100
    axis.axhline(
        gold_mean,
        color="#2a9d55",
        linestyle="--",
        linewidth=2,
        label=f"Kontrola minimalnej ekspozycji P1=1: {gold_mean:.2f}%",
    )
    axis.axhspan(
        gold_mean - gold_sd,
        gold_mean + gold_sd,
        color="#2a9d55",
        alpha=0.10,
    )
    axis.axvspan(60, 80, color="#d62828", alpha=0.07)
    axis.annotate(
        "początek załamania\nkompatybilności",
        xy=(80, 86.4),
        xytext=(105, 88.1),
        arrowprops={"arrowstyle": "->", "color": "#8d1b1b"},
        color="#8d1b1b",
    )
    axis.set(
        xlabel="Długość interwencji Phase 3 (epoki)",
        ylabel="Najlepsza validation accuracy [%]",
        title="Odtwarzanie wyniku po interwencji: średnia ± SD, 3 seedy",
        xticks=e3_values,
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", fontsize=9)
    save_figure(figure, output_dir, "phase3_phase4_recovery_accuracy")


def plot_branch_tradeoff(grid_rows, output_dir):
    e3_values, by_e3 = grouped(grid_rows, "e3")
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    styles = (
        ("final_full_accuracy", "full", "#1f77b4", "o"),
        ("final_dominant_accuracy", "dominant-only", "#d62728", "s"),
        ("final_weak_accuracy", "weak-only", "#2ca02c", "^"),
    )
    for metric, label, color, marker in styles:
        means, deviations = zip(
            *(mean_sd(by_e3[e3], metric) for e3 in e3_values)
        )
        axis.errorbar(
            e3_values,
            np.asarray(means) * 100,
            yerr=np.asarray(deviations) * 100,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=4,
        )
    axis.axvspan(60, 80, color="#d62828", alpha=0.07)
    axis.annotate(
        "weak-only nadal rośnie,\nale dominant-only się załamuje",
        xy=(80, 33.7),
        xytext=(92, 53),
        arrowprops={"arrowstyle": "->", "color": "#8d1b1b"},
        color="#8d1b1b",
    )
    axis.set(
        xlabel="Długość interwencji Phase 3 (epoki)",
        ylabel="Validation accuracy po 200 epokach Phase 4 [%]",
        title="Kompromis między odzyskiem słabej gałęzi a kompatybilnością",
        xticks=e3_values,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output_dir, "phase3_phase4_branch_tradeoff")


def plot_gold_selection(gold_rows, output_dir):
    figure, axis = plt.subplots(figsize=(7.4, 5.6))
    colors = ("#275dad", "#e07a1f", "#2a9d55")
    for row, color in zip(sorted(gold_rows, key=lambda item: item["seed"]), colors):
        start = (
            row["min_loss"],
            row["accuracy_at_min_loss"] * 100,
        )
        end = (
            row["loss_at_max_accuracy"],
            row["max_accuracy"] * 100,
        )
        axis.scatter(*start, color=color, marker="o", s=65)
        axis.scatter(*end, color=color, marker="*", s=140)
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8},
        )
        axis.text(
            end[0] + 0.006,
            end[1],
            f"seed {int(row['seed'])}",
            color=color,
            va="center",
        )
    axis.scatter(
        [],
        [],
        color="black",
        marker="o",
        label="minimum validation loss",
    )
    axis.scatter(
        [],
        [],
        color="black",
        marker="*",
        s=120,
        label="maksimum validation accuracy",
    )
    axis.set(
        xlabel="Full validation loss",
        ylabel="Full validation accuracy [%]",
        title="P1=1: loss i accuracy wybierają inne checkpointy Phase 2",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        output_dir,
        "phase2_minimal_exposure_loss_accuracy_selection",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    grid_rows = read_numeric_csv(args.grid, {"wandb_run_id"})
    gold_rows = read_numeric_csv(args.gold, {"wandb_run_id"})
    plot_recovery_accuracy(grid_rows, gold_rows, args.output_dir)
    plot_branch_tradeoff(grid_rows, args.output_dir)
    plot_gold_selection(gold_rows, args.output_dir)


if __name__ == "__main__":
    main()
