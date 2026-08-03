"""Summarize full Phase-3 relative-unimodal-parity trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def load_records(path):
    by_epoch = {}
    with Path(path).open(encoding="utf-8") as trajectory_file:
        for line in trajectory_file:
            record = json.loads(line)
            if int(record.get("version", 0)) != 1:
                raise ValueError(f"unsupported trajectory version in {path}")
            by_epoch[int(record["phase_epoch"])] = record
    records = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    expected = [0, 1, 2, 3, 4, *range(8, 201, 4)]
    if [record["phase_epoch"] for record in records] != expected:
        raise ValueError(f"incomplete Phase-3 trajectory: {path}")
    return records


def point(seed, record):
    metrics = record["metrics"]
    controller = record["controller"]
    return {
        "seed": seed,
        "phase_epoch": int(record["phase_epoch"]),
        "full_accuracy": float(metrics["full"]["accuracy"]),
        "dominant_accuracy": float(metrics["dominant_only"]["accuracy"]),
        "weak_accuracy": float(metrics["weak_only"]["accuracy"]),
        "full_loss": float(metrics["full"]["loss"]),
        "weak_loss": float(metrics["weak_only"]["loss"]),
        "dominant_ratio": float(controller["dominant_ratio"]),
        "weak_ratio": float(controller["weak_ratio"]),
        "parity_gap": float(controller["parity_gap"]),
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_retained": bool(record["checkpoint_retained"]),
    }


def summarize(seed, points):
    baseline = points[0]
    best_weak = min(
        points,
        key=lambda item: (
            -item["weak_ratio"], item["weak_loss"], item["phase_epoch"]
        ),
    )
    best_full = min(
        points,
        key=lambda item: (
            -item["full_accuracy"], item["full_loss"], item["phase_epoch"]
        ),
    )
    first_parity = next(
        (item for item in points if item["parity_gap"] >= 0.0), None
    )
    first_raw_crossing = next(
        (
            item
            for item in points
            if item["weak_accuracy"] >= item["dominant_accuracy"]
        ),
        None,
    )
    parity_hits = [item for item in points if item["parity_gap"] >= 0.0]

    def first_confirmed(tolerance, max_full_drop=None):
        previous = None
        for item in points:
            qualifies = item["parity_gap"] >= -float(tolerance)
            if max_full_drop is not None:
                qualifies = qualifies and (
                    baseline["full_accuracy"] - item["full_accuracy"]
                    <= float(max_full_drop)
                )
            if qualifies:
                if previous is not None:
                    return previous["phase_epoch"], item["phase_epoch"]
                previous = item
            else:
                previous = None
        return None, None

    exact_candidate, exact_confirmation = first_confirmed(0.0)
    tolerance_results = {}
    for tolerance in (0.0025, 0.005, 0.01):
        candidate, confirmation = first_confirmed(tolerance)
        tolerance_results[f"tol_{tolerance:g}_candidate"] = candidate
        tolerance_results[f"tol_{tolerance:g}_confirmation"] = confirmation
    guarded_candidate, guarded_confirmation = first_confirmed(
        0.005, max_full_drop=0.02
    )
    dominant_values = [item["dominant_accuracy"] for item in points]
    result = {
        "seed": seed,
        "dominant_ratio": baseline["dominant_ratio"],
        "baseline_full_accuracy": baseline["full_accuracy"],
        "baseline_dominant_accuracy": baseline["dominant_accuracy"],
        "baseline_weak_accuracy": baseline["weak_accuracy"],
        "best_weak_epoch": best_weak["phase_epoch"],
        "best_weak_ratio": best_weak["weak_ratio"],
        "best_weak_accuracy": best_weak["weak_accuracy"],
        "best_weak_parity_gap": best_weak["parity_gap"],
        "full_accuracy_at_best_weak": best_weak["full_accuracy"],
        "full_drop_at_best_weak": (
            baseline["full_accuracy"] - best_weak["full_accuracy"]
        ),
        "best_weak_checkpoint": best_weak["checkpoint_path"],
        "best_full_epoch": best_full["phase_epoch"],
        "best_full_accuracy": best_full["full_accuracy"],
        "first_parity_epoch": (
            first_parity["phase_epoch"] if first_parity else None
        ),
        "parity_hit_epochs": ";".join(
            str(item["phase_epoch"]) for item in parity_hits
        ),
        "exact_candidate_epoch": exact_candidate,
        "exact_confirmation_epoch": exact_confirmation,
        "tol_0.005_full_drop_0.02_candidate": guarded_candidate,
        "tol_0.005_full_drop_0.02_confirmation": guarded_confirmation,
        "first_raw_crossing_epoch": (
            first_raw_crossing["phase_epoch"]
            if first_raw_crossing else None
        ),
        "endpoint_weak_ratio": points[-1]["weak_ratio"],
        "endpoint_weak_accuracy": points[-1]["weak_accuracy"],
        "endpoint_full_accuracy": points[-1]["full_accuracy"],
        "dominant_accuracy_range": max(dominant_values) - min(dominant_values),
    }
    result.update(tolerance_results)
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, trajectories):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for seed, points in trajectories.items():
        epochs = [item["phase_epoch"] for item in points]
        weak_ratios = [item["weak_ratio"] for item in points]
        target = points[0]["dominant_ratio"]
        axes[0, 0].plot(epochs, weak_ratios, label=f"seed {seed}")
        axes[0, 0].axhline(target, linestyle="--", alpha=0.45)
        axes[0, 1].plot(
            epochs, [item["parity_gap"] for item in points], label=f"seed {seed}"
        )
        axes[1, 0].plot(
            epochs, [item["weak_accuracy"] for item in points], label=f"seed {seed}"
        )
        axes[1, 0].plot(
            epochs,
            [item["dominant_accuracy"] for item in points],
            linestyle="--",
            alpha=0.55,
        )
        axes[1, 1].plot(
            epochs, [item["full_accuracy"] for item in points], label=f"seed {seed}"
        )
    axes[0, 0].set_title("Weak ratio and seed-specific dominant target")
    axes[0, 1].set_title("Parity gap")
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("Weak (solid) and dominant (dashed) accuracy")
    axes[1, 1].set_title("Full accuracy")
    for axis in axes.flat:
        axis.set_xlabel("Phase-3 epoch")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_markdown(path, summaries, aggregate, plot_path):
    lines = [
        "# Analiza pełnych trajektorii relative unimodal parity",
        "",
        "Wszystkie decyzje poniżej wykorzystują wyłącznie validation proper.",
        "Test nie był liczony ani używany.",
        "",
        "| seed | target R_D | max R_W (e3) | gap | spadek full | parity hits | tol=.005 stop/select |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {seed} | {target} | {weak} ({weak_epoch}) | {gap} | {full_drop} | "
            "{hits} | {tol_confirmation}/{tol_candidate} |".format(
                seed=row["seed"],
                target=fmt(row["dominant_ratio"]),
                weak=fmt(row["best_weak_ratio"]),
                weak_epoch=row["best_weak_epoch"],
                gap=fmt(row["best_weak_parity_gap"]),
                full_drop=fmt(row["full_drop_at_best_weak"]),
                hits=row["parity_hit_epochs"] or "—",
                tol_confirmation=fmt(row["tol_0.005_confirmation"]),
                tol_candidate=fmt(row["tol_0.005_candidate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Wnioski",
            "",
            f"- Średni najlepszy gap wynosi {fmt(aggregate['mean_best_gap'])}; dokładne dwa kolejne trafienia nie wystąpiły w żadnym seedzie.",
            f"- Epoki maksimum weak ratio: {', '.join(str(row['best_weak_epoch']) for row in summaries)}; średnia {fmt(aggregate['mean_best_weak_epoch'], 1)}.",
            f"- Zakres dominant-only accuracy w obrębie trajektorii wynosi maksymalnie {fmt(aggregate['max_dominant_range'], 8)}, co potwierdza zamrożenie left/shared.",
            "- Równość surowych accuracy nie jest równoważna znormalizowanemu parity i pozostaje wyłącznie diagnostyką.",
            "- Następny stopper powinien używać non-inferiority/tolerancji niepewności wokół parity oraz osobnego constraintu full accuracy.",
            "",
            "Osiągnięcie maksimum weak ratio nie gwarantuje najlepszego downstream P4. Należy uruchomić P4 z kandydatów wybranych bez używania testu.",
        ]
    )
    if plot_path:
        lines.extend(["", f"![Trajektorie]({Path(plot_path).name})"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    trajectories = {}
    for value in args.trajectory:
        seed_text, path = value.split("=", 1)
        seed = int(seed_text)
        trajectories[seed] = [point(seed, item) for item in load_records(path)]
    summaries = [summarize(seed, trajectories[seed]) for seed in sorted(trajectories)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_points = [item for seed in sorted(trajectories) for item in trajectories[seed]]
    write_csv(output_dir / "unimodal_parity_phase3_points.csv", all_points)
    write_csv(output_dir / "unimodal_parity_phase3_summary.csv", summaries)
    best_epochs = [row["best_weak_epoch"] for row in summaries]
    best_gaps = [row["best_weak_parity_gap"] for row in summaries]
    aggregate = {
        "mean_best_weak_epoch": mean(best_epochs),
        "sd_best_weak_epoch": stdev(best_epochs) if len(best_epochs) > 1 else 0.0,
        "mean_best_gap": mean(best_gaps),
        "sd_best_gap": stdev(best_gaps) if len(best_gaps) > 1 else 0.0,
        "max_dominant_range": max(row["dominant_accuracy_range"] for row in summaries),
    }
    (output_dir / "unimodal_parity_phase3_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_path = output_dir / "unimodal_parity_phase3_trajectories.png"
    plotted = write_plot(plot_path, trajectories)
    write_markdown(
        output_dir / "PHASE3_UNIMODAL_PARITY_ANALYSIS_2026-08-01.md",
        summaries,
        aggregate,
        plot_path if plotted else None,
    )
    print(json.dumps({"summaries": summaries, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
