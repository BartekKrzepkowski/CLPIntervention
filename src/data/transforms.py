from src.data import transforms_cifar10, transforms_fmnist, transforms_kmnist, transforms_mnist, transforms_svhn, transforms_tinyimagenet

TRANSFORMS_BLURRED_RIGHT_NAME_MAP = {
    'mm_cifar10': lambda overlap: transforms_cifar10.TRANSFORMS_NAME_MAP['transform_train_blurred'](32, 32, 1/4, overlap),
    'mm_fmnist': lambda overlap: transforms_fmnist.TRANSFORMS_NAME_MAP['transform_train_blurred'](28, 28, 1/4, overlap),
    'mm_kmnist': lambda overlap: transforms_kmnist.TRANSFORMS_NAME_MAP['transform_train_blurred'](28, 28, 1/4, overlap),
    'mm_mnist': lambda overlap: transforms_mnist.TRANSFORMS_NAME_MAP['transform_train_blurred'](28, 28, 1/4, overlap),
    'mm_svhn': lambda overlap: transforms_svhn.TRANSFORMS_NAME_MAP['transform_train_blurred'](32, 32, 1/4, overlap),
    'mm_tinyimagenet': lambda overlap: transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_train_blurred'](64, 64, 1/4, overlap)
}

TRANSFORMS_PROPER_RIGHT_NAME_MAP = {
    'mm_cifar10': lambda overlap: transforms_cifar10.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'),
    'mm_fmnist': lambda overlap: transforms_fmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'),
    'mm_kmnist': lambda overlap: transforms_kmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'),
    'mm_mnist': lambda overlap: transforms_mnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'),
    'mm_svhn': lambda overlap: transforms_svhn.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'),
    'mm_tinyimagenet': lambda overlap: transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right')
}