#!/usr/bin/env python3
"""Plot Phase-4 branch TFIM trajectories from local JSONL artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PREFIX = "trace_fim_overall_train"


def _parse_series(value):
    seed, separator, raw_path = value.partition("=")
    if not separator or not seed or not raw_path:
        raise argparse.ArgumentTypeError("series must have the form SEED=PATH")
    return int(seed), Path(raw_path)


def _load(seed, path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") != "proper":
                continue
            metrics = record["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "phase_epoch": int(record["phase_epoch"]),
                    "global_step": int(record["global_step"]),
                    "tfim_left": float(metrics[f"{PREFIX}/proper_trace1"]),
                    "tfim_right": float(metrics[f"{PREFIX}/proper_trace2"]),
                    "ratio_left_to_right": float(
                        metrics[f"{PREFIX}/proper_ratio_left_to_right"]
                    ),
                }
            )
    if not rows:
        raise ValueError(f"no proper-input TFIM records in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, type=_parse_series)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for seed, path in args.series:
        rows.extend(_load(seed, path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "phase4_tfim_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["seed"], row["phase_epoch"])))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed in sorted({row["seed"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["phase_epoch"],
        )
        epochs = [row["phase_epoch"] for row in selected]
        axes[0].plot(epochs, [row["tfim_left"] for row in selected], "o-", label=f"left, seed {seed}")
        axes[0].plot(epochs, [row["tfim_right"] for row in selected], "s--", label=f"right, seed {seed}")
        axes[1].plot(epochs, [row["ratio_left_to_right"] for row in selected], "o-", label=f"seed {seed}")

    axes[0].set_yscale("log")
    axes[0].set_title("Phase 4 sampled TFIM")
    axes[0].set_xlabel("Phase-4 epoch")
    axes[0].set_ylabel("Trace")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_title(r"Branch TFIM ratio $Tr(F_L)/Tr(F_R)$")
    axes[1].set_xlabel("Phase-4 epoch")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"phase4_tfim_trajectory.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
