from math import ceil

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from src.data.normalization import normalization_for as _normalization
    

transform_train_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    transforms.RandomAffine(degrees=0.0, translate=(1/8, 1/8)),
    ToTensor(),
    Normalize(*_normalization(OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R, overlap))
])

transform_train_proper = lambda overlap, side: Compose([
    transforms.RandomAffine(degrees=0.0, translate=(1/8, 1/8)),
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


OVERLAP_TO_NORMALIZATION_MAP_PROPER_L = {
    0.0: ((0.25959462,), (0.34508348,)),
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
    0.0: ((0.31248608,), (0.35884658,)),
}

OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
    0.0: ((0.32540634,), (0.31021202,)),
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
