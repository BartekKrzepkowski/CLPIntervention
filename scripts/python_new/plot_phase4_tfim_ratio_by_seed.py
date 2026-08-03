#!/usr/bin/env python3
"""Plot raw Phase-4 TFIM left/right ratios without averaging seeds."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY = "trace_fim_overall_train/proper_ratio_left_to_right"


def _series(value):
    identity, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("series must be SEED,E3=PATH")
    seed, comma, e3 = identity.partition(",")
    if not comma:
        raise argparse.ArgumentTypeError("series must be SEED,E3=PATH")
    return int(seed), int(e3), Path(raw_path)


def _load(seed, e3, path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] != "proper":
                continue
            rows.append(
                {
                    "seed": seed,
                    "e3": e3,
                    "phase4_epoch": int(record["phase_epoch"]),
                    "ratio_left_to_right": float(record["metrics"][KEY]),
                }
            )
    if not rows:
        raise ValueError(f"no proper TFIM records in {path}")
    return sorted(rows, key=lambda row: row["phase4_epoch"])


def _draw(axis, rows, *, title):
    for e3 in sorted({row["e3"] for row in rows}):
        selected = [row for row in rows if row["e3"] == e3]
        axis.plot(
            [row["phase4_epoch"] for row in selected],
            [row["ratio_left_to_right"] for row in selected],
            marker="o",
            linewidth=1.8,
            label=f"e3={e3}",
        )
    axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Liczba epok Phase 4")
    axis.set_ylabel(r"$Tr(F_L) / Tr(F_R)$")
    axis.set_xticks(sorted({row["phase4_epoch"] for row in rows}))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, type=_series)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="phase4_tfim_ratio")
    args = parser.parse_args()

    rows = []
    for seed, e3, path in args.series:
        rows.extend(_load(seed, e3, path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seeds = sorted({row["seed"] for row in rows})
    figure, axes = plt.subplots(1, len(seeds), figsize=(5.3 * len(seeds), 4.8))
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        _draw(
            axis,
            [row for row in rows if row["seed"] == seed],
            title=f"seed {seed}",
        )
    figure.suptitle("Phase 4: nieuśredniony stosunek TFIM gałęzi")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{args.stem}_all_seeds.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)

    for seed in seeds:
        figure, axis = plt.subplots(figsize=(8.6, 5.3))
        _draw(
            axis,
            [row for row in rows if row["seed"] == seed],
            title=f"Phase 4 TFIM ratio — seed {seed}",
        )
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(
                args.output_dir / f"{args.stem}_seed{seed}.{suffix}",
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(figure)


if __name__ == "__main__":
    main()
