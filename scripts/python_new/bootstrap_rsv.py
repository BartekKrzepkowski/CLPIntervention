#!/usr/bin/env python3
"""Compare matched RSV artifacts with a hierarchical paired bootstrap."""

import argparse
import json
from pathlib import Path

from src.analysis.bootstrap import (
    PairedRSV,
    hierarchical_paired_bootstrap,
    load_rsv_result,
)


def _named_paths(values, option):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} entries must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"{option} entries must use NAME=PATH")
        if name in parsed:
            raise ValueError(f"duplicate {option} name: {name}")
        parsed[name] = path
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument(
        "--intervention", action="append", required=True, metavar="SEED=PATH"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--unit-statistic", choices=("mean", "median"), default="median")
    parser.add_argument("--image-statistic", choices=("mean", "median"), default="median")
    parser.add_argument("--model-statistic", choices=("mean", "median"), default="mean")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-class-stratification", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    controls = _named_paths(args.control, "--control")
    interventions = _named_paths(args.intervention, "--intervention")
    if set(controls) != set(interventions):
        missing_control = sorted(set(interventions) - set(controls))
        missing_intervention = sorted(set(controls) - set(interventions))
        raise ValueError(
            "control/intervention seeds must match; "
            f"missing control={missing_control}, "
            f"missing intervention={missing_intervention}"
        )
    pairs = [
        PairedRSV(
            name,
            load_rsv_result(controls[name]),
            load_rsv_result(interventions[name]),
        )
        for name in sorted(controls)
    ]
    result = hierarchical_paired_bootstrap(
        pairs,
        replicates=args.replicates,
        seed=args.seed,
        unit_statistic=args.unit_statistic,
        image_statistic=args.image_statistic,
        model_statistic=args.model_statistic,
        confidence=args.confidence,
        stratified=not args.no_class_stratification,
    )
    result["pairs"] = {
        name: {
            "control": str(Path(controls[name]).resolve()),
            "intervention": str(Path(interventions[name]).resolve()),
        }
        for name in sorted(controls)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
