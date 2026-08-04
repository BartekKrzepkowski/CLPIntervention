from src.data import transforms_cifar10, transforms_fmnist, transforms_kmnist, transforms_mnist, transforms_svhn, transforms_tinyimagenet

TRANSFORMS_BLURRED_RIGHT_NAME_MAP = {
    "mm_cifar10": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_cifar10.TRANSFORMS_NAME_MAP["transform_train_blurred"](
        32, 32, resize_factor, overlap, normalization_profile
    ),
    "mm_fmnist": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_fmnist.TRANSFORMS_NAME_MAP["transform_train_blurred"](28, 28, resize_factor, overlap),
    "mm_kmnist": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_kmnist.TRANSFORMS_NAME_MAP["transform_train_blurred"](28, 28, resize_factor, overlap),
    "mm_mnist": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_mnist.TRANSFORMS_NAME_MAP["transform_train_blurred"](28, 28, resize_factor, overlap),
    "mm_svhn": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_svhn.TRANSFORMS_NAME_MAP["transform_train_blurred"](32, 32, resize_factor, overlap),
    "mm_tinyimagenet": lambda overlap, resize_factor=1 / 4, normalization_profile=None: transforms_tinyimagenet.TRANSFORMS_NAME_MAP["transform_train_blurred"](64, 64, resize_factor, overlap),
}

TRANSFORMS_PROPER_RIGHT_NAME_MAP = {
    "mm_cifar10": lambda overlap, normalization_profile=None: transforms_cifar10.TRANSFORMS_NAME_MAP["transform_train_proper"](
        overlap, "right", normalization_profile
    ),
    "mm_fmnist": lambda overlap, normalization_profile=None: transforms_fmnist.TRANSFORMS_NAME_MAP["transform_train_proper"](overlap, "right"),
    "mm_kmnist": lambda overlap, normalization_profile=None: transforms_kmnist.TRANSFORMS_NAME_MAP["transform_train_proper"](overlap, "right"),
    "mm_mnist": lambda overlap, normalization_profile=None: transforms_mnist.TRANSFORMS_NAME_MAP["transform_train_proper"](overlap, "right"),
    "mm_svhn": lambda overlap, normalization_profile=None: transforms_svhn.TRANSFORMS_NAME_MAP["transform_train_proper"](overlap, "right"),
    "mm_tinyimagenet": lambda overlap, normalization_profile=None: transforms_tinyimagenet.TRANSFORMS_NAME_MAP["transform_train_proper"](overlap, "right"),
}
