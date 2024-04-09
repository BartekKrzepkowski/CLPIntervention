import torch

from src.data.datasets import get_cubbirds, get_food101, get_tinyimagenet,\
    get_mm_cifar10, get_mm_fmnist, get_mm_kmnist, get_mm_mnist, get_mm_svhn, get_mm_tinyimagenet
from src.modules.architectures.mm_effnetv2 import MMEffNetV2S, ResNet18PyTorch, MMConvNext
from src.modules.architectures.mm_mlp import MMMLPwithNorm
from src.modules.architectures.mm_resnets import build_mm_resnet
from src.modules.architectures.models import MMSimpleCNN
from src.modules.losses import ClassificationLoss, FisherPenaltyLoss, MSESoftmaxLoss, BalancePenaltyLoss
from src.visualization.clearml_logger import ClearMLLogger
from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger


ACT_NAME_MAP = {
    'gelu': torch.nn.GELU,
    'identity': torch.nn.Identity,
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
}

DATASET_NAME_MAP = {
    'cubbirds': get_cubbirds,
    'food101': get_food101,
    'tinyimagenet': get_tinyimagenet,
    'mm_cifar10': get_mm_cifar10,
    'mm_fmnist': get_mm_fmnist,
    'mm_kmnist': get_mm_kmnist,
    'mm_mnist': get_mm_mnist,
    'mm_svhn': get_mm_svhn,
    'mm_tinyimagenet': get_mm_tinyimagenet,
}

LOGGERS_NAME_MAP = {
    'clearml': ClearMLLogger,
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

LOSS_NAME_MAP = {
    'balance_loss': BalancePenaltyLoss,
    'ce': torch.nn.CrossEntropyLoss,
    'cls': ClassificationLoss,
    'fp': FisherPenaltyLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss,
    'mse_softmax': MSESoftmaxLoss,
}

MODEL_NAME_MAP = {
    'mm_mlp_bn': MMMLPwithNorm,
    'mm_simple_cnn': MMSimpleCNN,
    'mm_resnet': build_mm_resnet,
    'mm_effnetv2s': MMEffNetV2S,
    'mm_resnet18': ResNet18PyTorch,
    'mm_convnext': MMConvNext,
}

NORM_LAYER_NAME_MAP = {
    'bn1d': torch.nn.BatchNorm1d,
    'bn2d': torch.nn.BatchNorm2d,
    'group_norm': torch.nn.GroupNorm,
    'instance_norm_1d': torch.nn.InstanceNorm1d,
    'instance_norm_2d': torch.nn.InstanceNorm2d,
    'layer_norm': torch.nn.LayerNorm,
}

OPTIMIZER_NAME_MAP = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}

SCHEDULER_NAME_MAP = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}
