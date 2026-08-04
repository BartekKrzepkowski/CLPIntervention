from math import ceil

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from src.data.normalization import normalization_for as _normalization
    
# mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)

transform_train_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(*_normalization(OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R, overlap))
])

transform_train_proper = lambda overlap, side: Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
}

OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
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
