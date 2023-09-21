import torch

from src.data.datasets import get_mnist, get_cifar10, get_cifar100, get_tinyimagenet, get_imagenet, get_cubbirds, get_food101, get_dual_cifar10
from src.modules.losses import ClassificationLoss, FisherPenaltyLoss, MSESoftmaxLoss, BADGELoss
from src.modules.architectures.models import MLP, MLPwithNorm, SimpleCNN, SimpleCNNwithNorm,\
    SimpleCNNwithDropout, SimpleCNNwithNormandDropout, DualSimpleCNN
from src.modules.architectures.resnets import ResNet18, ResNet34
from src.modules.architectures.mm_resnets import build_mm_resnet
from src.utils.utils_optim import MultiStepwithDoubleLinearWarmup
from src.visualization.clearml_logger import ClearMLLogger
from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger

ACT_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'identity': torch.nn.Identity
}

DATASET_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
    'imagenet': get_imagenet,
    'cubbirds': get_cubbirds,
    'food101': get_food101,
    'dual_cifar10': get_dual_cifar10,
}

LOGGERS_NAME_MAP = {
    'clearml': ClearMLLogger,
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

LOSS_NAME_MAP = {
    'ce': torch.nn.CrossEntropyLoss,
    'cls': ClassificationLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss,
    'mse_softmax': MSESoftmaxLoss,
    'fp': FisherPenaltyLoss,
    'badge': BADGELoss,
}

MODEL_NAME_MAP = {
    'mlp': MLP,
    'mlp_with_norm': MLPwithNorm,
    'simple_cnn': SimpleCNN,
    'simple_cnn_with_norm': SimpleCNNwithNorm,
    'simple_cnn_with_dropout': SimpleCNNwithDropout,
    'simple_cnn_with_norm_and_dropout': SimpleCNNwithNormandDropout,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'dual_simple_cnn': DualSimpleCNN,
    'mm_resnet': build_mm_resnet
}

NORM_LAYER_NAME_MAP = {
    'bn1d': torch.nn.BatchNorm1d,
    'bn2d': torch.nn.BatchNorm2d,
    'layer_norm': torch.nn.LayerNorm,
    'group_norm': torch.nn.GroupNorm,
    'instance_norm_1d': torch.nn.InstanceNorm1d,
    'instance_norm_2d': torch.nn.InstanceNorm2d,
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multistep_with_double_linear_warmup': MultiStepwithDoubleLinearWarmup,
}
