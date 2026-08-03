import torch

from src.modules.architectures.mm_mlp import MMMLPwithNorm
from src.modules.architectures.mm_resnets import build_mm_resnet
from src.modules.architectures.wrappers import BiModalModelwithPretrainedBranches
from src.utils.common import MODEL_NAME_MAP
from src.utils.utils_model import infer_dims_from_blocks, load_model_specific_params


def test_dimension_inference_does_not_mutate_batchnorm_state_or_mode():
    blocks = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
    )
    blocks.train()
    batchnorm = blocks[1]
    mean_before = batchnorm.running_mean.clone()
    batches_before = batchnorm.num_batches_tracked.clone()

    assert infer_dims_from_blocks(blocks, torch.randn(2, 1, 8, 8), 1) == (4, 8, 8, 4)
    assert blocks.training
    torch.testing.assert_close(batchnorm.running_mean, mean_before)
    torch.testing.assert_close(batchnorm.num_batches_tracked, batches_before)


def test_mlp_concatenation_and_feature_contract():
    model = MMMLPwithNorm(
        num_classes=3,
        input_channels=1,
        img_height=4,
        img_width=4,
        overlap=0.0,
        hidden_layers_dim=[16, 12, 8, 6],
        activation_name="relu",
        scaling_factor=2,
    ).eval()
    x = torch.randn(2, 1, 4, 4)
    logits, left_features, right_features = model(x, x, return_features=True)

    assert logits.shape == (2, 3)
    assert left_features.shape == right_features.shape == (2, 12)


def test_resnet_supports_grayscale_and_disabled_branch():
    model = build_mm_resnet(
        num_classes=3,
        input_channels=1,
        img_height=16,
        img_width=8,
        overlap=0.0,
        backbone_type="resnet18",
        batchnorm_layers=True,
        modify_resnet=True,
        only_features=False,
        skips=True,
        wheter_concate=False,
        width_scale=0.125,
    ).eval()
    x = torch.randn(2, 1, 16, 8)
    logits = model(x, x, left_branch_intervention="deactivation", enable_left_branch=False)
    assert logits.shape == (2, 3)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left_branch = torch.nn.Linear(4, 3)
        self.right_branch = torch.nn.Linear(4, 3)
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x1, x2, return_features=False, **kwargs):
        left = self.left_branch(x1)
        right = self.right_branch(x2)
        logits = self.fc(left + right)
        return (logits, left, right) if return_features else logits


def test_umt_teachers_stay_frozen_and_in_eval_mode():
    student = Student()
    left_teacher = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout())
    right_teacher = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout())
    model = BiModalModelwithPretrainedBranches(student, left_teacher, right_teacher)
    model.train()

    assert not left_teacher.training and not right_teacher.training
    assert all(not parameter.requires_grad for parameter in left_teacher.parameters())
    teacher_features = model.teacher_features(torch.randn(2, 4), torch.randn(2, 4))
    assert all(not feature.requires_grad for feature in teacher_features)
    assert model(torch.randn(2, 4), torch.randn(2, 4))[0].shape == (2, 2)


def test_every_registered_model_has_a_configuration_file():
    for model_name in MODEL_NAME_MAP:
        assert isinstance(load_model_specific_params(model_name), dict)
