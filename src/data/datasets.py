import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from src.data.datasets_class import SplitAndAugmentDataset
from src.data import transforms_cifar10, transforms_mnist, transforms_fmnist, transforms_svhn, transforms_kmnist, transforms_tinyimagenet


DOWNLOAD = False


def get_mm_mnist(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['MNIST_PATH']
    print(dataset_path)
    
    train_dataset = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_mnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_mnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True, reverse=False)
    
    test_proper_dataset = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_mnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_mnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False, reverse=False)
    
    test_blurred_dataset = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_mnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_mnist.TRANSFORMS_NAME_MAP['transform_eval_blurred'](28, 28, resize_factor, overlap), overlap=overlap, is_train=False, reverse=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset


def get_mm_kmnist(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['KMNIST_PATH']
    print(dataset_path)
    
    train_dataset = datasets.KMNIST(dataset_path, train=True, download=DOWNLOAD)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_kmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_kmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True, reverse=False)
    
    test_proper_dataset = datasets.KMNIST(dataset_path, train=False, download=DOWNLOAD)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_kmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_kmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False, reverse=False)
    
    test_blurred_dataset = datasets.KMNIST(dataset_path, train=False, download=DOWNLOAD)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_kmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_kmnist.TRANSFORMS_NAME_MAP['transform_eval_blurred'](28, 28, resize_factor, overlap), overlap=overlap, is_train=False, reverse=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset


def get_mm_fmnist(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['FMNIST_PATH']
    print(dataset_path)
    
    train_dataset = datasets.FashionMNIST(dataset_path, train=True, download=DOWNLOAD)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_fmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_fmnist.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True)
    
    test_proper_dataset = datasets.FashionMNIST(dataset_path, train=False, download=DOWNLOAD)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_fmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_fmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False)
    
    test_blurred_dataset = datasets.FashionMNIST(dataset_path, train=False, download=DOWNLOAD)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_fmnist.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_fmnist.TRANSFORMS_NAME_MAP['transform_eval_blurred'](28, 28, resize_factor, overlap), overlap=overlap, is_train=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset


def get_mm_svhn(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['SVHN_PATH']
    
    train_dataset = datasets.SVHN(dataset_path, split='train', download=DOWNLOAD)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_svhn.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_svhn.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True, reverse=False)
    
    test_proper_dataset = datasets.SVHN(dataset_path, split='test', download=DOWNLOAD)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_svhn.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_svhn.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False, reverse=False)
    
    test_blurred_dataset = datasets.SVHN(dataset_path, split='test', download=DOWNLOAD)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_svhn.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_svhn.TRANSFORMS_NAME_MAP['transform_eval_blurred'](32, 32, resize_factor, overlap), overlap=overlap, is_train=False, reverse=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset


def get_mm_cifar10(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_cifar10.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_cifar10.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True)
    
    test_proper_dataset = datasets.CIFAR10(dataset_path, train=False, download=True)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_cifar10.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_cifar10.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False)
    
    test_blurred_dataset = datasets.CIFAR10(dataset_path, train=False, download=True)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_cifar10.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_cifar10.TRANSFORMS_NAME_MAP['transform_eval_blurred'](32, 32, resize_factor, overlap), overlap=overlap, is_train=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset
    

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
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(train_path, transform=transform_eval)
    test_data = datasets.ImageFolder(test_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_mm_tinyimagenet(dataset_path=None, overlap=0.0, resize_factor=1/4):
    dataset_path = dataset_path if dataset_path is not None else os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/val/images'
    
    train_dataset = datasets.ImageFolder(train_path)
    train_dual_augment_dataset = SplitAndAugmentDataset(train_dataset, transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'left'), transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_train_proper'](overlap, 'right'), overlap=overlap, is_train=True)
    
    test_proper_dataset = datasets.ImageFolder(test_path)
    test_proper_dual_augment_dataset = SplitAndAugmentDataset(test_proper_dataset, transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'right'), overlap=overlap, is_train=False)
    
    test_blurred_dataset = datasets.ImageFolder(test_path)
    test_blurred_dual_augment_dataset = SplitAndAugmentDataset(test_blurred_dataset, transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_eval_proper'](overlap, 'left'), transforms_tinyimagenet.TRANSFORMS_NAME_MAP['transform_eval_blurred'](64, 64, resize_factor, overlap), overlap=overlap, is_train=False)
    
    return train_dual_augment_dataset, test_proper_dual_augment_dataset, test_blurred_dual_augment_dataset



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

