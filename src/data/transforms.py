from math import ceil

from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, RandomHorizontalFlip
    
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)

transform_train_blurred = lambda h, w, resize_factor, overlap: Compose([
    ToTensor(),
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    RandomAffine(degrees=0, translate=(1/8, 1/8)),
    # RandomHorizontalFlip(),
    Normalize(*OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[overlap])
])

transform_train_proper = lambda overlap, side: Compose([
    ToTensor(),
    RandomAffine(degrees=0, translate=(1/8, 1/8)),
    # RandomHorizontalFlip(),
    Normalize(*SIDE_MAP_PROPER[side][overlap])
])

transform_eval_blurred = lambda h, w, resize_factor, overlap: Compose([
    ToTensor(),
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Normalize(*OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[overlap])
])


transform_eval_proper = lambda overlap, side: Compose([
    ToTensor(),
    Normalize(*SIDE_MAP_PROPER[side][overlap])
])


OVERLAP_TO_NORMALIZATION_MAP_PROPER_L = {
    0.0: ((0.49156436, 0.48242152, 0.4468064), (0.24701783, 0.24345341, 0.26166818)),
    0.125: ((0.49192753, 0.48170146, 0.44616485), (0.24664736, 0.24305077, 0.2609319)),
    1.0: ((0.49156436, 0.48242152, 0.4468064), (0.24701783, 0.24345341, 0.26166818)), #FAKE
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
    0.0: ((0.49123746, 0.48189786, 0.44625866), (0.24704705, 0.24351668, 0.2615068)),
    0.125: ((0.4916447, 0.4812342, 0.4457098), (0.24667019, 0.2430911 , 0.26078665)),
    1.0: ((0.49123746, 0.48189786, 0.44625866), (0.24704705, 0.24351668, 0.2615068)), #FAKE
}

OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
    0.0: ((0.4908226 , 0.4814503 , 0.44576296), (0.21958971, 0.21675968, 0.23706897)),
    0.125: ((0.49126044, 0.48079324, 0.44522214), (0.21784401, 0.21501008, 0.23512621)),
    1.0: ((0.4908226 , 0.4814503 , 0.44576296), (0.21958971, 0.21675968, 0.23706897)), #FAKE
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