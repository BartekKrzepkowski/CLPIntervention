#!/usr/bin/env python3
"""Plot clean-control training FIM traces from local JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = ("bimodal", "left_proper", "right_proper")
TRACE_LEFT_KEY = "trace_fim_overall_train/proper_trace1"
TRACE_RIGHT_KEY = "trace_fim_overall_train/proper_trace2"
RATIO_KEY = "trace_fim_overall_train/proper_ratio_left_to_right"
SEED_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-seed clean-control T-FIM plots and summaries from local "
            "trace_fim_train.jsonl files."
        )
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="SEED,CONDITION=PATH",
        help=(
            "Input series; repeat once for each seed and condition. CONDITION must "
            "be bimodal, left_proper, or right_proper."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which CSV, JSON, PNG, and PDF outputs are written.",
    )
    return parser.parse_args()


def seed_sort_key(seed: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(seed), seed)
    except ValueError:
        return (1, seed, seed)


def parse_series_specs(specs: Sequence[str]) -> dict[str, dict[str, Path]]:
    series_paths: dict[str, dict[str, Path]] = {}
    for spec in specs:
        try:
            seed_condition, path_text = spec.split("=", 1)
            seed, condition = seed_condition.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --series {spec!r}; expected SEED,CONDITION=PATH"
            ) from exc

        seed = seed.strip()
        condition = condition.strip()
        if not seed or not SEED_PATTERN.fullmatch(seed):
            raise ValueError(
                f"Invalid seed {seed!r}; use only letters, digits, '.', '_', or '-'"
            )
        if condition not in CONDITIONS:
            raise ValueError(
                f"Invalid condition {condition!r} for seed {seed!r}; "
                f"expected one of {', '.join(CONDITIONS)}"
            )
        if not path_text:
            raise ValueError(f"Empty path in --series {spec!r}")
        if condition in series_paths.setdefault(seed, {}):
            raise ValueError(
                f"Duplicate series for seed {seed!r}, condition {condition!r}"
            )
        series_paths[seed][condition] = Path(path_text)

    for seed, paths_by_condition in series_paths.items():
        actual = set(paths_by_condition)
        expected = set(CONDITIONS)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected: {', '.join(extra)}")
            raise ValueError(
                f"Seed {seed!r} must have exactly three conditions ({'; '.join(details)})"
            )
    return series_paths


def require_finite_number(record: dict[str, Any], key: str, context: str) -> int | float:
    if key not in record:
        raise ValueError(f"Missing {key!r} in {context}")
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be numeric in {context}; got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{key!r} must be finite in {context}; got {value!r}")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def read_series(path: Path, seed: str, condition: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(
            f"Input for seed {seed!r}, condition {condition!r} is not a file: {path}"
        )

    rows_by_epoch: dict[int | float, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {context}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {context}")
            if record.get("kind") != "proper":
                continue

            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"metrics must be an object in {context}")
            epoch = require_finite_number(record, "phase_epoch", context)
            trace_left = require_finite_number(metrics, TRACE_LEFT_KEY, context)
            trace_right = require_finite_number(metrics, TRACE_RIGHT_KEY, context)
            ratio = (
                require_finite_number(metrics, RATIO_KEY, context)
                if condition == "bimodal"
                else None
            )
            if epoch in rows_by_epoch:
                raise ValueError(
                    f"Duplicate proper record for epoch {epoch!r}, seed {seed!r}, "
                    f"condition {condition!r} in {path}"
                )
            if condition == "left_proper" and trace_right != 0:
                raise ValueError(
                    f"Expected {TRACE_RIGHT_KEY}=0 for left_proper in {context}; "
                    f"got {trace_right!r}"
                )
            if condition == "right_proper" and trace_left != 0:
                raise ValueError(
                    f"Expected {TRACE_LEFT_KEY}=0 for right_proper in {context}; "
                    f"got {trace_left!r}"
                )
            rows_by_epoch[epoch] = {
                "seed": seed,
                "condition": condition,
                "epoch": epoch,
                "tfim_left": trace_left,
                "tfim_right": trace_right,
                "ratio_left_to_right": ratio,
            }

    if not rows_by_epoch:
        raise ValueError(
            f"No records with kind='proper' for seed {seed!r}, "
            f"condition {condition!r} in {path}"
        )
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def load_and_validate(
    series_paths: dict[str, dict[str, Path]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for seed in sorted(series_paths, key=seed_sort_key):
        data[seed] = {}
        for condition in CONDITIONS:
            data[seed][condition] = read_series(
                series_paths[seed][condition], seed, condition
            )

        epoch_sets = {
            condition: {row["epoch"] for row in data[seed][condition]}
            for condition in CONDITIONS
        }
        reference_epochs = epoch_sets["bimodal"]
        for condition in CONDITIONS[1:]:
            if epoch_sets[condition] != reference_epochs:
                missing = sorted(reference_epochs - epoch_sets[condition])
                extra = sorted(epoch_sets[condition] - reference_epochs)
                raise ValueError(
                    f"Epoch mismatch for seed {seed!r}, condition {condition!r}; "
                    f"missing relative to bimodal: {missing}, extra: {extra}"
                )
    return data


def series_values(rows: Sequence[dict[str, Any]], key: str) -> list[int | float]:
    return [row[key] for row in rows]


def summarize_values(
    epochs: Sequence[int | float], values: Sequence[int | float]
) -> dict[str, int | float]:
    min_index = min(range(len(values)), key=lambda index: (values[index], index))
    max_index = max(range(len(values)), key=lambda index: (values[index], -index))
    return {
        "first": values[0],
        "last": values[-1],
        "min": values[min_index],
        "max": values[max_index],
        "epoch_of_min": epochs[min_index],
        "epoch_of_max": epochs[max_index],
    }


def context_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def ratio_crosses_one(values: Sequence[int | float]) -> bool:
    if any(value == 1 for value in values):
        return True
    return any(
        (left - 1) * (right - 1) < 0
        for left, right in zip(values, values[1:])
    )


def build_summary(
    data: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"seeds": {}}
    for seed, conditions in data.items():
        bimodal = conditions["bimodal"]
        left_proper = conditions["left_proper"]
        right_proper = conditions["right_proper"]
        epochs = series_values(bimodal, "epoch")
        bimodal_left = series_values(bimodal, "tfim_left")
        bimodal_right = series_values(bimodal, "tfim_right")
        bimodal_ratio = series_values(bimodal, "ratio_left_to_right")
        unimodal_left = series_values(left_proper, "tfim_left")
        unimodal_right = series_values(right_proper, "tfim_right")

        summary["seeds"][seed] = {
            "epochs": epochs,
            "bimodal": {
                "trace_left": summarize_values(epochs, bimodal_left),
                "trace_right": summarize_values(epochs, bimodal_right),
                "ratio": summarize_values(epochs, bimodal_ratio),
            },
            "unimodal": {
                "trace_left": summarize_values(epochs, unimodal_left),
                "trace_right": summarize_values(epochs, unimodal_right),
            },
            "final_context_ratios": {
                "bimodal_left_over_unimodal_left": context_ratio(
                    bimodal_left[-1], unimodal_left[-1]
                ),
                "bimodal_right_over_unimodal_right": context_ratio(
                    bimodal_right[-1], unimodal_right[-1]
                ),
            },
            "ratio_crosses_one": ratio_crosses_one(bimodal_ratio),
        }
    return summary


def write_raw_csv(
    output_path: Path, data: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    fieldnames = (
        "seed",
        "condition",
        "epoch",
        "tfim_left",
        "tfim_right",
        "ratio_left_to_right",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in data:
            for condition in CONDITIONS:
                for row in data[seed][condition]:
                    writer.writerow(
                        {
                            **row,
                            "ratio_left_to_right": (
                                ""
                                if row["ratio_left_to_right"] is None
                                else row["ratio_left_to_right"]
                            ),
                        }
                    )


def write_summary_json(output_path: Path, summary: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def style_axis(
    axis: plt.Axes,
    epochs: Sequence[int | float],
    *,
    xlabel: bool = True,
) -> None:
    axis.set_xticks(epochs)
    if xlabel:
        axis.set_xlabel("Training epoch")
    axis.grid(True, alpha=0.3)


def plot_bimodal_traces(axis: plt.Axes, rows: Sequence[dict[str, Any]]) -> None:
    epochs = series_values(rows, "epoch")
    axis.plot(epochs, series_values(rows, "tfim_left"), marker="o", label=r"Tr($F_L$)")
    axis.plot(epochs, series_values(rows, "tfim_right"), marker="s", label=r"Tr($F_R$)")
    axis.set_ylabel("T-FIM trace")
    axis.set_title("Bimodal branch traces")
    style_axis(axis, epochs)
    axis.legend()


def plot_bimodal_ratio(axis: plt.Axes, rows: Sequence[dict[str, Any]]) -> None:
    epochs = series_values(rows, "epoch")
    axis.plot(
        epochs,
        series_values(rows, "ratio_left_to_right"),
        color="tab:purple",
        marker="o",
        label=r"Tr($F_L$) / Tr($F_R$)",
    )
    axis.axhline(1, color="black", linestyle="--", linewidth=1, label="Parity = 1")
    axis.set_ylabel("Left / right ratio")
    axis.set_title("Bimodal trace ratio")
    style_axis(axis, epochs)
    axis.legend()


def plot_unimodal_traces(
    axis: plt.Axes,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
) -> None:
    epochs = series_values(left_rows, "epoch")
    axis.plot(
        epochs,
        series_values(left_rows, "tfim_left"),
        color="tab:blue",
        marker="o",
        label=r"left_proper Tr($F_L$)",
    )
    axis.plot(
        epochs,
        series_values(right_rows, "tfim_right"),
        color="tab:orange",
        marker="s",
        label=r"right_proper Tr($F_R$)",
    )
    axis.set_ylabel("T-FIM trace")
    axis.set_title("Unimodal controls")
    style_axis(axis, epochs)
    axis.legend()


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    try:
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{basename}.{extension}", bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_by_seed(
    output_dir: Path, data: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    seeds = list(data)
    fig, axes = plt.subplots(
        3,
        len(seeds),
        squeeze=False,
        figsize=(5.2 * len(seeds), 12),
    )
    for column, seed in enumerate(seeds):
        conditions = data[seed]
        plot_bimodal_traces(axes[0][column], conditions["bimodal"])
        plot_bimodal_ratio(axes[1][column], conditions["bimodal"])
        plot_unimodal_traces(
            axes[2][column],
            conditions["left_proper"],
            conditions["right_proper"],
        )
        axes[0][column].set_title(f"Seed {seed} — bimodal branch traces")
        axes[1][column].set_title(f"Seed {seed} — bimodal trace ratio")
        axes[2][column].set_title(f"Seed {seed} — unimodal controls")
    save_figure(fig, output_dir, "tfim_clean_controls_by_seed")


def plot_branch_context(
    output_dir: Path, data: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    seeds = list(data)
    fig, axes = plt.subplots(
        2,
        len(seeds),
        squeeze=False,
        figsize=(5.2 * len(seeds), 8),
    )
    for column, seed in enumerate(seeds):
        conditions = data[seed]
        bimodal = conditions["bimodal"]
        left_proper = conditions["left_proper"]
        right_proper = conditions["right_proper"]
        epochs = series_values(bimodal, "epoch")

        left_axis = axes[0][column]
        left_axis.plot(
            epochs,
            series_values(bimodal, "tfim_left"),
            marker="o",
            label=r"bimodal Tr($F_L$)",
        )
        left_axis.plot(
            epochs,
            series_values(left_proper, "tfim_left"),
            marker="s",
            label=r"left_proper Tr($F_L$)",
        )
        left_axis.set_title(f"Seed {seed} — left branch context")
        left_axis.set_ylabel("T-FIM trace")
        style_axis(left_axis, epochs)
        left_axis.legend()

        right_axis = axes[1][column]
        right_axis.plot(
            epochs,
            series_values(bimodal, "tfim_right"),
            marker="o",
            label=r"bimodal Tr($F_R$)",
        )
        right_axis.plot(
            epochs,
            series_values(right_proper, "tfim_right"),
            marker="s",
            label=r"right_proper Tr($F_R$)",
        )
        right_axis.set_title(f"Seed {seed} — right branch context")
        right_axis.set_ylabel("T-FIM trace")
        style_axis(right_axis, epochs)
        right_axis.legend()
    save_figure(fig, output_dir, "tfim_clean_controls_branch_context")


def plot_individual_seeds(
    output_dir: Path, data: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    for seed, conditions in data.items():
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), squeeze=False)
        plot_bimodal_traces(axes[0][0], conditions["bimodal"])
        plot_bimodal_ratio(axes[1][0], conditions["bimodal"])
        plot_unimodal_traces(
            axes[2][0],
            conditions["left_proper"],
            conditions["right_proper"],
        )
        fig.suptitle(f"Clean-control T-FIM — seed {seed}")
        save_figure(fig, output_dir, f"tfim_clean_controls_seed{seed}")


def main() -> None:
    args = parse_args()
    try:
        series_paths = parse_series_specs(args.series)
        data = load_and_validate(series_paths)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_raw_csv(args.output_dir / "tfim_clean_controls_raw.csv", data)
    write_summary_json(
        args.output_dir / "tfim_clean_controls_summary.json",
        build_summary(data),
    )
    plot_by_seed(args.output_dir, data)
    plot_branch_context(args.output_dir, data)
    plot_individual_seeds(args.output_dir, data)


if __name__ == "__main__":
    main()
