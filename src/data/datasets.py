import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from src.data.datasets_class import SplitAndAugmentDataset
from src.data.transforms import TRANSFORMS_NAME_MAP

DOWNLOAD = False


def get_mnist(dataset_path):
    dataset_path = dataset_path if dataset_path is not None else os.environ['MNIST_PATH']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(dataset_path=None, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_blurred = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), 
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None),
        transforms.Normalize(mean, std)
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    transform_train_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transform_train_2 if whether_aug else transform_eval
    train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
    test_proper_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_eval)
    test_blurred_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_blurred)
    return train_dataset, test_proper_dataset, test_blurred_dataset


def get_dual_cifar10(dataset_path=None, whether_aug=True, proper_normalization=True, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True)
    
    test_proper_dataset = datasets.CIFAR10(dataset_path, train=False, download=True)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False)
    
    test_blurred_dataset = datasets.CIFAR10(dataset_path, train=False, download=True)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), TRANSFORMS_NAME_MAP['transform_eval_blurred'](32, 32, resize_factor, overlap), overlap=overlap, is_train=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset


def get_cifar100(dataset_path=None, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR100_PATH']
    if proper_normalization:
        mean, std = (0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_blurred = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), 
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None),
        transforms.Normalize(mean, std)
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    transform_train_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transform_train_2 if whether_aug else transform_eval
    train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True)
    test_proper_dataset = datasets.CIFAR100(dataset_path, train=False, download=True)
    test_blurred_dataset = datasets.CIFAR100(dataset_path, train=False, download=True)
    return train_dataset, test_proper_dataset, test_blurred_dataset
    


def get_tinyimagenet(proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['TINYIMAGENET_PATH']
    if proper_normalization:
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/val/images'
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(train_path, transform=transform_eval)
    test_data = datasets.ImageFolder(test_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_imagenet(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['IMAGENET_PATH']
    transform_eval = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageNet(dataset_path, transform=transform_train)
    train_eval_data = datasets.ImageNet(dataset_path, transform=transform_eval)
    test_data = datasets.ImageNet(dataset_path, split='val', transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_cubbirds(proper_normalization=False):
    if proper_normalization:
        raise NotImplementedError()
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CUBBIRDS_PATH']
    # TODO include the script that generates the symlinks somewhere
    trainset_path = f'{dataset_path}/images_train_test/train'
    eval_path = f'{dataset_path}/images_train_test/val'
    transform_eval = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageFolder(trainset_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(trainset_path, transform=transform_eval)
    test_data = datasets.ImageFolder(eval_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_food101(dataset_path, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['FOOD101_PATH']
    if proper_normalization:
        mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_eval = transforms.Compose([
        transforms.Resize(150, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, interpolation=InterpolationMode.BILINEAR),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    transform_train_2 = transforms.Compose([
        transforms.Resize(140, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(1/64, 1/64)),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transform_train_2 if whether_aug else transform_eval
    train_data = datasets.Food101(dataset_path, split='train', transform=transform_train)
    train_eval_data = datasets.Food101(dataset_path, split='train', transform=transform_eval)
    test_data = datasets.Food101(dataset_path, split='test', transform=transform_eval)
    return train_data, train_eval_data, test_data


DATASETS_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
    'imagenet': get_imagenet,
    'cubbirds': get_cubbirds,
    'food101': get_food101,
    'dual_cifar10': get_dual_cifar10,
}



