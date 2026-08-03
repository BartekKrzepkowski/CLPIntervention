"""Shared validation for versioned per-field normalization statistics."""


def normalization_for(stats_by_overlap, overlap):
    if not stats_by_overlap:
        raise ValueError(
            "Normalization statistics are unavailable or unverified for this "
            "dataset. Recompute them with scripts/python_new/get_mean_and_std.py "
            "before training."
        )
    try:
        return stats_by_overlap[overlap]
    except KeyError as error:
        supported = ", ".join(str(value) for value in sorted(stats_by_overlap))
        raise ValueError(
            f"Unsupported overlap={overlap}; configured values: {supported}"
        ) from error


def normalization_from_transform(transform):
    """Return JSON-safe mean/std from a torchvision Compose, if present."""
    for operation in getattr(transform, "transforms", ()):
        if operation.__class__.__name__ == "Normalize":
            return {
                "mean": [float(value) for value in operation.mean],
                "std": [float(value) for value in operation.std],
            }
    return None
