"""Hierarchical paired bootstrap for post-hoc RSV comparisons."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.analysis.rsv import RSV_FORMAT


@dataclass(frozen=True)
class PairedRSV:
    """One seed/model represented by matched control and intervention artifacts."""

    name: str
    control: dict
    intervention: dict


def load_rsv_result(path):
    result = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or result.get("format") != RSV_FORMAT:
        raise ValueError(f"{path} is not a supported RSV artifact")
    return result


def _reduce(values, statistic, axis=None):
    if statistic == "mean":
        return np.mean(values, axis=axis)
    if statistic == "median":
        return np.median(values, axis=axis)
    raise ValueError("statistic must be 'mean' or 'median'")


def _validate_pair(pair):
    control = pair.control
    intervention = pair.intervention
    for key in ("selected_indices", "selected_labels"):
        if not torch.equal(control[key], intervention[key]):
            raise ValueError(f"{pair.name}: mismatched {key}")
    if control["rsv"].shape != intervention["rsv"].shape:
        raise ValueError(f"{pair.name}: mismatched RSV shapes")
    control_meta = control["metadata"]
    intervention_meta = intervention["metadata"]
    protocol_keys = (
        "measurement",
        "layer",
        "spatial_average_pool",
        "sign_convention",
    )
    for key in protocol_keys:
        if control_meta.get(key) != intervention_meta.get(key):
            raise ValueError(f"{pair.name}: mismatched RSV protocol field {key}")


def paired_image_differences(pair, unit_statistic="median"):
    """Return intervention-minus-control RSV for each matched image."""
    _validate_pair(pair)
    control = pair.control["rsv"].detach().cpu().numpy()
    intervention = pair.intervention["rsv"].detach().cpu().numpy()
    return _reduce(intervention, unit_statistic, axis=1) - _reduce(
        control, unit_statistic, axis=1
    )


def _resample_images(values, labels, generator, stratified):
    if not stratified:
        indices = generator.integers(0, values.size, size=values.size)
        return values[indices]
    sampled = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        sampled.append(
            values[generator.choice(candidates, size=candidates.size, replace=True)]
        )
    return np.concatenate(sampled)


def hierarchical_paired_bootstrap(
    pairs,
    *,
    replicates=10_000,
    seed=83,
    unit_statistic="median",
    image_statistic="median",
    model_statistic="mean",
    confidence=0.95,
    stratified=True,
):
    """Bootstrap matched model/image differences without treating units as IID."""
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError("at least one paired model is required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    protocol_keys = (
        "measurement",
        "layer",
        "spatial_average_pool",
        "sign_convention",
    )
    reference_metadata = pairs[0].control["metadata"]
    for pair in pairs:
        _validate_pair(pair)
        for key in protocol_keys:
            if pair.control["metadata"].get(key) != reference_metadata.get(key):
                raise ValueError(f"{pair.name}: protocol differs between models: {key}")

    differences = []
    labels = []
    for pair in pairs:
        differences.append(paired_image_differences(pair, unit_statistic))
        labels.append(pair.control["selected_labels"].detach().cpu().numpy())

    per_model_observed = np.asarray(
        [_reduce(values, image_statistic) for values in differences],
        dtype=np.float64,
    )
    observed = float(_reduce(per_model_observed, model_statistic))
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    model_count = len(pairs)
    for replicate in range(replicates):
        selected_models = generator.integers(0, model_count, size=model_count)
        model_values = []
        for model_index in selected_models:
            sampled_images = _resample_images(
                differences[model_index],
                labels[model_index],
                generator,
                stratified,
            )
            model_values.append(_reduce(sampled_images, image_statistic))
        samples[replicate] = _reduce(np.asarray(model_values), model_statistic)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    first_metadata = pairs[0].control["metadata"]
    return {
        "format": "clpintervention.rsv.paired_bootstrap",
        "version": 1,
        "difference": "intervention-control",
        "measurement": first_metadata.get("measurement"),
        "layer": first_metadata.get("layer"),
        "spatial_average_pool": first_metadata.get("spatial_average_pool", False),
        "paired_models": model_count,
        "selected_examples_per_model": [int(item.size) for item in differences],
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "stratified_by_class": bool(stratified),
        "unit_statistic": unit_statistic,
        "image_statistic": image_statistic,
        "model_statistic": model_statistic,
        "observed": observed,
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_probability_above_zero": float(np.mean(samples > 0.0)),
        "per_model_observed": {
            pair.name: float(value)
            for pair, value in zip(pairs, per_model_observed, strict=True)
        },
        "warning": (
            None
            if model_count >= 5
            else "Fewer than five paired models; uncertainty is weakly identified."
        ),
    }
