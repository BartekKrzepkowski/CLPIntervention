from math import ceil

from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, RandomHorizontalFlip
    

transform_train_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    RandomAffine(degrees=0.0, translate=(1/8, 1/8)),
    ToTensor(),
    Normalize(*OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[overlap])
])

transform_train_proper = lambda overlap, side: Compose([
    RandomAffine(degrees=0.0, translate=(1/8, 1/8)),
    ToTensor(),
    Normalize(*SIDE_MAP_PROPER[side][overlap])
])

transform_eval_blurred = lambda h, w, resize_factor, overlap: Compose([
    Resize((ceil(resize_factor * h), ceil(resize_factor * ceil((overlap / 2 + 0.5) * w))), interpolation=InterpolationMode.BILINEAR, antialias=None),
    Resize((h, ceil((overlap / 2 + 0.5) * w)), interpolation=InterpolationMode.BILINEAR, antialias=None),
    ToTensor(),
    Normalize(*OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[overlap])
])

transform_eval_proper = lambda overlap, side: Compose([
    ToTensor(),
    Normalize(*SIDE_MAP_PROPER[side][overlap])
])


OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R = {
    0.0: ((0.48079526, 0.44845405, 0.3977925), (0.2450607, 0.23671584, 0.25229153)),
    0.125: ((0.49126044, 0.48079324, 0.44522214), (0.21784401, 0.21501008, 0.23512621)),
    1.0: ((0.4908226 , 0.4814503 , 0.44576296), (0.21958971, 0.21675968, 0.23706897)), #FAKE
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_R = {
    0.0: ((0.48073354, 0.4484018, 0.39780444), (0.27681842, 0.2689017, 0.2819355)),
    0.125: ((0.4916447, 0.4812342, 0.4457098), (0.24667019, 0.2430911 , 0.26078665)),
    1.0: ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

OVERLAP_TO_NORMALIZATION_MAP_PROPER_L = {
    0.0: ((0.47975746, 0.44774577, 0.39728615), (0.27715358, 0.26922715, 0.28222984)),
    0.125: ((0.49192753, 0.48170146, 0.44616485), (0.24664736, 0.24305077, 0.2609319)),
    1.0: ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
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