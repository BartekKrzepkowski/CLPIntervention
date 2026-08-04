from math import ceil

from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine

from src.data.cifar10_protocol import (
    CIFAR10_LEGACY_NORMALIZATION_PROFILE,
    CIFAR10_PROTOCOL_PROFILE,
    cifar10_protocol_normalization,
)
from src.data.normalization import normalization_for as _normalization

# mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)

OVERLAP_TO_NORMALIZATION_MAP_PROPER_L = {
    0.0: ((0.4915608321950317, 0.4824227237087908, 0.4468044428398145), (0.2470178401439214, 0.24345336919802013, 0.26166858447567926)),
    0.125: ((0.4919278650465282, 0.4817010015715583, 0.4461662961518023), (0.24664751492283543, 0.243049730414396, 0.26093230506660736)),
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
    0.0: ((0.49123854756134744, 0.4818941155435833, 0.4462574056167809), (0.2470465242578638, 0.2435166015426914, 0.26150679336264493)),
    0.125: ((0.49164333684916606, 0.48123654295264184, 0.4457057886634943), (0.24666987280713285, 0.24309154138602998, 0.26078715614853976)),
}

OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
    0.0: ((0.49135468529694687, 0.48185410232077586, 0.4461651198096588), (0.20231621757381077, 0.19990853153973656, 0.22172341125578746)),
    0.125: ((0.49158368517123047, 0.4810221906115423, 0.4454515447791533), (0.2024501112316537, 0.19999062794872283, 0.2214495767624039)),
}

SIDE_MAP_PROPER = {
    "left": OVERLAP_TO_NORMALIZATION_MAP_PROPER_L,
    "right": OVERLAP_TO_NORMALIZATION_MAP_PROPER_R,
}


def _field_normalization(field, overlap, normalization_profile=None):
    profile = normalization_profile or CIFAR10_LEGACY_NORMALIZATION_PROFILE
    if profile == CIFAR10_LEGACY_NORMALIZATION_PROFILE:
        if field == "blurred_right":
            return _normalization(OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R, overlap)
        side = field.removeprefix("proper_")
        return _normalization(SIDE_MAP_PROPER[side], overlap)
    if profile == CIFAR10_PROTOCOL_PROFILE:
        return cifar10_protocol_normalization(field, overlap)
    raise ValueError(f"Unsupported CIFAR-10 normalization profile: {profile}")


def transform_train_blurred(
    h, w, resize_factor, overlap, normalization_profile=None
):
    field_width = ceil((overlap / 2 + 0.5) * w)
    return Compose([
        Resize(
            (ceil(resize_factor * h), ceil(resize_factor * field_width)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        ),
        Resize(
            (h, field_width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        ),
        RandomAffine(degrees=0.0, translate=(1 / 8, 1 / 8)),
        ToTensor(),
        Normalize(
            *_field_normalization(
                "blurred_right", overlap, normalization_profile
            )
        ),
    ])


def transform_train_proper(overlap, side, normalization_profile=None):
    return Compose([
        RandomAffine(degrees=0.0, translate=(1 / 8, 1 / 8)),
        ToTensor(),
        Normalize(
            *_field_normalization(
                f"proper_{side}", overlap, normalization_profile
            )
        ),
    ])


def transform_eval_blurred(
    h, w, resize_factor, overlap, normalization_profile=None
):
    field_width = ceil((overlap / 2 + 0.5) * w)
    return Compose([
        Resize(
            (ceil(resize_factor * h), ceil(resize_factor * field_width)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        ),
        Resize(
            (h, field_width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        ),
        ToTensor(),
        Normalize(
            *_field_normalization(
                "blurred_right", overlap, normalization_profile
            )
        ),
    ])


def transform_eval_proper(overlap, side, normalization_profile=None):
    return Compose([
        ToTensor(),
        Normalize(
            *_field_normalization(
                f"proper_{side}", overlap, normalization_profile
            )
        ),
    ])


TRANSFORMS_NAME_MAP = {
    "transform_train_blurred": transform_train_blurred,
    "transform_train_proper": transform_train_proper,
    "transform_eval_blurred": transform_eval_blurred,
    "transform_eval_proper": transform_eval_proper,
}
