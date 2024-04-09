import torch
from torch import Tensor
from typing import Callable, List, Optional, Type, Union
import torch.nn as nn
from functools import partial

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from src.utils.utils_model import infer_dims_from_blocks


class BasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        self.skips = kwargs.pop("skips")
        super().__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.skips:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None and self.skips:
            identity = self.downsample(x)

        if self.skips:
            out += identity
        out = self.relu2(out)

        return out


class Bottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        self.skips = kwargs.pop("skips", True)
        super().__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.skips:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None and self.skips:
            identity = self.downsample(x)

        if self.skips:
            out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        width_scale: float = 1.0,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skips: bool = True,
        wheter_concate: bool = False,
        eps: float = 1e-5,
        overlap: float = 0.0,
        img_height: int = 32,
        img_width: int = 32,
        modify_resnet: bool = False,
    ) -> None:
        super().__init__()
        from math import ceil
        # _log_api_usage_once(self)
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scale_width = width_scale
        self.skips = skips


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.inplanes = int(64 * width_scale)
        self.conv11 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False) if modify_resnet else \
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = torch.nn.Identity() if modify_resnet else nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv21 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=2, bias=False) if modify_resnet else \
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool2 = torch.nn.Identity() if modify_resnet else nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.left_branch = nn.Sequential(self.conv11,
                                  norm_layer(self.inplanes),
                                  nn.ReLU(inplace=True),
                                  self.maxpool1,
                                  self._make_layer(block, 64, layers[0]),
                                  self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]))
        self.inplanes = int(64 * width_scale)
        self.right_branch = nn.Sequential(self.conv21,
                                  norm_layer(self.inplanes),
                                  nn.ReLU(inplace=True),
                                  self.maxpool2,
                                  self._make_layer(block, 64, layers[0]),
                                  self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]))

        z = torch.randn(1, 3, img_height, img_width)
        # z = self.left_branch(z)
        # _, self.channels_out, self.height, self.width = z.shape
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        self.main_branch = nn.Sequential(self._make_layer(block, 256 * self.scaling_factor, layers[2], stride=2, dilate=replace_stride_with_dilation[1]),
                                  self._make_layer(block, 512 * self.scaling_factor, layers[3], stride=2, dilate=replace_stride_with_dilation[2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_scale * block.expansion), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:

        planes = int(planes * self.scale_width)

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                skips=self.skips,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    skips=self.skips,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"

        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)

            x1 = self.left_branch(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn((x1.size(0), self.channels_out, self.height, self.width), device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros((x1.size(0), self.channels_out, self.height, self.width), device=x1.device)
            else:
                raise ValueError("Invalid left branch intervention")
    
        if enable_right_branch:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn_like(x2, device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros_like(x2, device=x2.device)

            x2 = self.right_branch(x2)
        else:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn((x2.size(0), self.channels_out, self.height, self.width), device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros((x2.size(0), self.channels_out, self.height, self.width), device=x2.device)
            else:
                raise ValueError("Invalid right branch intervention")

        y = torch.cat((x1, x2), dim=-1) if self.scaling_factor == 2 else x1 + x2
        y = self.main_branch(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


def build_mm_resnet(num_classes, input_channels, img_height, img_width, overlap, backbone_type, batchnorm_layers, modify_resnet, only_features, skips, wheter_concate, width_scale):

    resnet = partial(
        ResNet, num_classes=num_classes, width_scale=width_scale, skips=skips, overlap=overlap, modify_resnet=modify_resnet, wheter_concate=wheter_concate, img_height=img_height, img_width=img_width
    )
    if not batchnorm_layers:
        resnet = partial(resnet, norm_layer=nn.Identity)
    match backbone_type:
        case "resnet18":
            model = resnet(BasicBlock, [2, 2, 2, 2])
        case "resnet34":
            model = resnet(BasicBlock, [3, 4, 6, 3])
        case "resnet50":
            model = resnet(Bottleneck, [3, 4, 6, 3])
        case "resnet101":
            model = resnet(Bottleneck, [3, 4, 23, 3])
        case "resnet152":
            model = resnet(Bottleneck, [3, 8, 36, 3])
        case _:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    if only_features:
        model.fc = torch.nn.Identity()

    if not batchnorm_layers:
        # turn off batch norm tracking stats and learning parameters
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
                m.affine = False
                m.running_mean = None
                m.running_var = None

    renset_penultimate_layer_size = {
        "resnet18": int(512 * width_scale),
        "resnet34": int(512 * width_scale),
        "resnet50": int(2048 * width_scale),
        "resnet101": int(2048 * width_scale),
        "resnet152": int(2048 * width_scale),
    }
    model.penultimate_layer_size = renset_penultimate_layer_size[backbone_type]

    return model


if __name__ == "__main__":
    pass