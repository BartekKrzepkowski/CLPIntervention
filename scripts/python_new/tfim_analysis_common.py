"""Neutral numerical helpers shared by TFIM analysis entrypoints."""

from __future__ import annotations

import numpy as np


FEATURES = (
    "endpoint_log_ratio",
    "ema_log_ratio_alpha_0_5",
    "slope_log_ratio",
    "weighted_slope_log_ratio",
    "forecast_log_ratio_e18",
    "mean_log_ratio",
)


def probe_features(epochs, ratios):
    epochs = np.asarray(epochs, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if np.any(ratios <= 0.0):
        raise ValueError("TFIM ratios must be positive before log transform")
    log_ratios = np.log(ratios)

    ema = float(log_ratios[0])
    for value in log_ratios[1:]:
        ema = 0.5 * float(value) + 0.5 * ema

    _, slope = _weighted_line(
        epochs, log_ratios, np.ones_like(log_ratios)
    )
    exponential_weights = np.exp((epochs - epochs[-1]) / 6.0)
    intercept, weighted_slope = _weighted_line(
        epochs, log_ratios, exponential_weights
    )
    return {
        "endpoint_log_ratio": float(log_ratios[-1]),
        "ema_log_ratio_alpha_0_5": ema,
        "slope_log_ratio": slope,
        "weighted_slope_log_ratio": weighted_slope,
        "forecast_log_ratio_e18": intercept + weighted_slope * 18.0,
        "mean_log_ratio": float(np.mean(log_ratios)),
    }


def spearman(x_values, y_values):
    x_ranks = _rank(np.asarray(x_values, dtype=float))
    y_ranks = _rank(np.asarray(y_values, dtype=float))
    if np.std(x_ranks) == 0.0 or np.std(y_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _weighted_line(epochs, values, weights):
    design = np.column_stack((np.ones_like(epochs), epochs))
    root_weights = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weights[:, None],
        values * root_weights,
        rcond=None,
    )
    return float(coefficients[0]), float(coefficients[1])


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks
