from math import ceil

from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine

from src.data.normalization import normalization_for as _normalization
    

transform_train_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    RandomAffine(degrees=10.0, translate=(1/8, 1/8)),
    ToTensor(),
    Normalize(*_normalization(OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R, overlap))
])

transform_train_proper = lambda overlap, side: Compose([
    RandomAffine(degrees=10.0, translate=(1/8, 1/8)),
    ToTensor(),
    Normalize(*_normalization(SIDE_MAP_PROPER[side], overlap))
])

transform_eval_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    ToTensor(),
    Normalize(*_normalization(OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R, overlap))
])

transform_eval_proper = lambda overlap, side: Compose([
    ToTensor(),
    Normalize(*_normalization(SIDE_MAP_PROPER[side], overlap))
])


OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
    0.0: ((0.48079526, 0.44845405, 0.3977925), (0.2450607, 0.23671584, 0.25229153)),
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
    0.0: ((0.48073354, 0.4484018, 0.39780444), (0.27681842, 0.2689017, 0.2819355)),
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_L = {
    0.0: ((0.47975746, 0.44774577, 0.39728615), (0.27715358, 0.26922715, 0.28222984)),
}


SIDE_MAP_PROPER = {
    'left': OVERLAP_TO_NORMALIZATION_MAP_PROPER_L,
    'right': OVERLAP_TO_NORMALIZATION_MAP_PROPER_R,
}

TRANSFORMS_NAME_MAP = {
    'transform_train_blurred': transform_train_blurred,
    'transform_train_proper': transform_train_proper,
    'transform_eval_blurred': transform_eval_blurred,
    'transform_eval_proper': transform_eval_proper,
}
