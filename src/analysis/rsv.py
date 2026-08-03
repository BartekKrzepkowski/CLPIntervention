"""Reproducible Relative Source Variance measurements.

The probe follows the protocol described in the CLPIntervention thesis:
select a fixed number of training images per class, keep the original plus
random affine variants of one visual field while the other field is fixed,
and measure activations at a declared shared layer.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as torch_functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from src.modules.callbacks import relative_source_variance, source_variances
from src.data.normalization import normalization_from_transform
from src.utils.utils_data import get_targets


RSV_FORMAT = "clpintervention.rsv"
RSV_VERSION = 3


@dataclass(frozen=True)
class RSVProbeConfig:
    samples_per_class: int = 5
    variants_per_source: int = 100
    translate_pixels: int = 4
    rotation_degrees: float = 10.0
    seed: int = 83
    overlap: float = 0.0

    def validate(self):
        if self.samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        if self.variants_per_source < 2:
            raise ValueError("variants_per_source must be at least two")
        if self.translate_pixels < 0:
            raise ValueError("translate_pixels must be non-negative")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("overlap must be in [0, 1]")


@dataclass(frozen=True)
class RSVLayerSpec:
    """A named RSV measurement point and its optional analysis-only pooling."""

    name: str
    layer_name: str
    spatial_average_pool: bool = False


DEFAULT_RSV_LAYER_SPECS = (
    RSVLayerSpec("stage3_avgpool", "main_branch.0", spatial_average_pool=True),
    RSVLayerSpec("stage4_avgpool", "avgpool"),
)


def balanced_fixed_count_indices(targets, samples_per_class=5, seed=0):
    """Select exactly ``samples_per_class`` examples from every class."""
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    generator = np.random.default_rng(seed)
    selected = []
    for label in np.unique(targets):
        candidates = np.flatnonzero(targets == label)
        if candidates.size < samples_per_class:
            raise ValueError(f"class {label} has fewer than {samples_per_class} samples")
        selected.extend(
            generator.choice(candidates, size=samples_per_class, replace=False).tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def split_visual_fields(image, overlap=0.0):
    """Split a PIL image using the same geometry as SplitAndAugmentDataset."""
    width, height = image.size
    field_width = math.ceil(width * (0.5 + overlap / 2.0))
    left = image.crop((0, 0, field_width, height))
    right = image.crop((width - field_width, 0, width, height))
    return left, right


def affine_variants(field, config, generator):
    """Return the unmodified field followed by deterministic random variants."""
    variants = [field.copy()]
    for _ in range(config.variants_per_source - 1):
        angle = float(
            torch.empty(()).uniform_(
                -config.rotation_degrees,
                config.rotation_degrees,
                generator=generator,
            )
        )
        translation = [
            int(
                torch.randint(
                    -config.translate_pixels,
                    config.translate_pixels + 1,
                    (),
                    generator=generator,
                )
            )
            for _ in range(2)
        ]
        variants.append(
            transform_functional.affine(
                field,
                angle=angle,
                translate=translation,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )
    return variants


def resolve_shared_layer(model, layer_name=None):
    """Resolve an explicit layer or the first stage in ``main_branch``."""
    analysis_model = model.main_model if hasattr(model, "main_model") else model
    modules = dict(analysis_model.named_modules())
    if layer_name is not None:
        try:
            return layer_name, modules[layer_name]
        except KeyError as error:
            raise ValueError(f"Unknown RSV layer: {layer_name}") from error

    shared = getattr(analysis_model, "main_branch", None)
    if shared is None:
        raise ValueError("model does not expose main_branch; pass an explicit layer")
    first_stage = next(iter(shared.named_children()), None)
    if first_stage is None:
        raise ValueError("main_branch contains no child stage")
    relative_name, module = first_stage
    return f"main_branch.{relative_name}", module


def _captured_outputs(model, layer_specs, left, right, batch_size, device):
    outputs = {spec.name: [] for spec, _layer in layer_specs}
    current = {spec.name: [] for spec, _layer in layer_specs}
    handles = []

    def capture_for(spec):
        def capture(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("RSV layer must return a tensor")
            if spec.spatial_average_pool:
                if output.ndim != 4:
                    raise ValueError(
                        f"{spec.name} requested spatial pooling for a "
                        f"{output.ndim}-dimensional activation"
                    )
                output = torch_functional.adaptive_avg_pool2d(output, (1, 1))
            current[spec.name].append(output.detach().cpu().clone())

        return capture

    for spec, layer in layer_specs:
        handles.append(layer.register_forward_hook(capture_for(spec)))
    try:
        for start in range(0, left.size(0), batch_size):
            for captured in current.values():
                captured.clear()
            end = start + batch_size
            model(left[start:end].to(device), right[start:end].to(device))
            for spec, _layer in layer_specs:
                captured = current[spec.name]
                if len(captured) != 1:
                    raise RuntimeError(
                        f"RSV layer {spec.name!r} was called {len(captured)} "
                        "times for one forward pass"
                    )
                outputs[spec.name].append(captured[0])
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(chunks) for name, chunks in outputs.items()}


@torch.no_grad()
def measure_rsv_layers(
    model,
    raw_dataset,
    left_transform,
    right_transform,
    *,
    config=RSVProbeConfig(),
    layer_specs=DEFAULT_RSV_LAYER_SPECS,
    batch_size=32,
    device=None,
    extra_metadata=None,
):
    """Measure multiple RSV points in the same model forward passes."""
    config.validate()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    layer_specs = tuple(layer_specs)
    if not layer_specs:
        raise ValueError("at least one RSV layer spec is required")
    names = [spec.name for spec in layer_specs]
    paths = [spec.layer_name for spec in layer_specs]
    if len(set(names)) != len(names):
        raise ValueError("RSV layer spec names must be unique")
    if len(set(paths)) != len(paths):
        raise ValueError("RSV layer module paths must be unique")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    targets = get_targets(raw_dataset)
    indices = balanced_fixed_count_indices(
        targets, config.samples_per_class, config.seed
    )
    resolved_layers = []
    for spec in layer_specs:
        resolved_name, layer = resolve_shared_layer(model, spec.layer_name)
        resolved_layers.append(
            (
                RSVLayerSpec(
                    spec.name,
                    resolved_name,
                    spatial_average_pool=spec.spatial_average_pool,
                ),
                layer,
            )
        )

    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(config.seed)
    rsv_rows = {spec.name: [] for spec, _layer in resolved_layers}
    left_variance_rows = {spec.name: [] for spec, _layer in resolved_layers}
    right_variance_rows = {spec.name: [] for spec, _layer in resolved_layers}
    activation_shapes = {spec.name: None for spec, _layer in resolved_layers}
    try:
        for index in indices:
            image, _label = raw_dataset[int(index)]
            left_field, right_field = split_visual_fields(image, config.overlap)
            left_fields = affine_variants(left_field, config, generator)
            right_fields = affine_variants(right_field, config, generator)

            left_varying = torch.stack([left_transform(item) for item in left_fields])
            right_varying = torch.stack([right_transform(item) for item in right_fields])
            left_fixed = left_transform(left_field).unsqueeze(0).expand_as(left_varying)
            right_fixed = right_transform(right_field).unsqueeze(0).expand_as(right_varying)

            responses_left = _captured_outputs(
                model, resolved_layers, left_varying, right_fixed, batch_size, device
            )
            responses_right = _captured_outputs(
                model, resolved_layers, left_fixed, right_varying, batch_size, device
            )
            for spec, _layer in resolved_layers:
                response_left = responses_left[spec.name]
                response_right = responses_right[spec.name]
                current_shape = tuple(response_left.shape[1:])
                if activation_shapes[spec.name] is None:
                    activation_shapes[spec.name] = current_shape
                elif activation_shapes[spec.name] != current_shape:
                    raise RuntimeError(
                        f"RSV activation shape changed for {spec.name}"
                    )

                left_variance, right_variance = source_variances(
                    response_left, response_right
                )
                rsv = relative_source_variance(response_left, response_right)
                rsv_rows[spec.name].append(rsv.flatten())
                left_variance_rows[spec.name].append(left_variance.flatten())
                right_variance_rows[spec.name].append(right_variance.flatten())
    finally:
        model.train(was_training)

    results = {}
    for spec, _layer in resolved_layers:
        rsv_by_example = torch.stack(rsv_rows[spec.name])
        results[spec.name] = {
            "format": RSV_FORMAT,
            "version": RSV_VERSION,
            "rsv": rsv_by_example,
            "source_variance_left": torch.stack(left_variance_rows[spec.name]),
            "source_variance_right": torch.stack(right_variance_rows[spec.name]),
            "selected_indices": torch.from_numpy(indices.copy()),
            "selected_labels": torch.as_tensor(targets[indices], dtype=torch.long),
            "metadata": {
                "probe": asdict(config),
                "sign_convention": "+1=left,-1=right (Kleinman et al.)",
                "measurement": spec.name,
                "layer": spec.layer_name,
                "spatial_average_pool": spec.spatial_average_pool,
                "activation_shape": activation_shapes[spec.name],
                "inference_batch_size": int(batch_size),
                "normalization": {
                    "left": normalization_from_transform(left_transform),
                    "right": normalization_from_transform(right_transform),
                },
                "selected_examples": int(indices.size),
                "units_per_example": int(rsv_by_example.size(1)),
                **dict(extra_metadata or {}),
            },
        }
    return results


@torch.no_grad()
def measure_rsv(
    model,
    raw_dataset,
    left_transform,
    right_transform,
    *,
    config=RSVProbeConfig(),
    layer_name=None,
    spatial_average_pool=False,
    batch_size=32,
    device=None,
    extra_metadata=None,
):
    """Measure one RSV point; retained as the single-layer public API."""
    resolved_layer_name, _layer = resolve_shared_layer(model, layer_name)
    result = measure_rsv_layers(
        model,
        raw_dataset,
        left_transform,
        right_transform,
        config=config,
        layer_specs=(
            RSVLayerSpec(
                "custom",
                resolved_layer_name,
                spatial_average_pool=spatial_average_pool,
            ),
        ),
        batch_size=batch_size,
        device=device,
        extra_metadata=extra_metadata,
    )["custom"]
    if layer_name is None:
        result["metadata"]["measurement"] = "legacy_stage3_unpooled"
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rsv_result(result, output_path):
    """Save mandatory raw tensors plus a compact, human-readable manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    rsv = result["rsv"]
    manifest = {
        "format": result["format"],
        "version": result["version"],
        "raw_file": output_path.name,
        "raw_sha256": _sha256(output_path),
        "shape": list(rsv.shape),
        "value_count": int(rsv.numel()),
        "mean": float(rsv.mean()),
        "median": float(rsv.median()),
        "minimum": float(rsv.min()),
        "maximum": float(rsv.max()),
        "metadata": result["metadata"],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path
