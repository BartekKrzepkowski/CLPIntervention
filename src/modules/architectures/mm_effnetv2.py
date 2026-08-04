from copy import deepcopy

import torch
import torchvision

from src.utils.utils_model import infer_dims_from_blocks


def _replace_input_conv(conv, input_channels):
    if conv.in_channels == input_channels:
        return conv
    return torch.nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )


class MMEffNetV2S(torch.nn.Module):
    def __init__(self, num_classes=200, dropout=0.2, stochastic_depth_prob=0.2, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(MMEffNetV2S, self).__init__()
        if wheter_concate:
            raise ValueError("Concatenation is not supported for this architecture")
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.efficientnet_v2_s(num_classes=num_classes, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
        model.features[0][0] = _replace_input_conv(model.features[0][0], input_channels)
        
        self.left_branch = model.features[:len(model.features) // 2]
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = model.features[len(model.features) // 2:]
        
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        z = torch.zeros(1, input_channels, img_height, img_width)
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True, return_features=False):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        # Processing noise or zeros by the active branch is for demonstration purposes only, it makes no sense in practice
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            x1 = self.left_branch(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = x1.new_empty((x1.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = x1.new_zeros((x1.size(0), self.channels_out, self.height, self.width))
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
                x2 = x2.new_empty((x2.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = x2.new_zeros((x2.size(0), self.channels_out, self.height, self.width))
            else:
                raise ValueError("Invalid right branch intervention")
            
        features_left, features_right = x1, x2
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        y = self.main_branch(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return (y, features_left, features_right) if return_features else y
    


class ResNet18PyTorch(torch.nn.Module):
    def __init__(self, num_classes=200, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(ResNet18PyTorch, self).__init__()
        if wheter_concate:
            raise ValueError("Concatenation is not supported for this architecture")
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = _replace_input_conv(model.conv1, input_channels)
        
        # self.model.number_of_classes = num_classes
        # self.model.input_size = img_height
        # self.model.input_channels = input_channels
        
        self.left_branch = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2)
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = torch.nn.Sequential(model.layer3, model.layer4)
        
        self.avgpool = model.avgpool
        self.classifier = model.fc
        
        z = torch.zeros(1, input_channels, img_height, img_width)
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True, return_features=False):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        # Processing noise or zeros by the active branch is for demonstration purposes only, it makes no sense in practice
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            x1 = self.left_branch(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = x1.new_empty((x1.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = x1.new_zeros((x1.size(0), self.channels_out, self.height, self.width))
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
                x2 = x2.new_empty((x2.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = x2.new_zeros((x2.size(0), self.channels_out, self.height, self.width))
            else:
                raise ValueError("Invalid right branch intervention")
            
        features_left, features_right = x1, x2
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        y = self.main_branch(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return (y, features_left, features_right) if return_features else y
    
    
class MMConvNext(torch.nn.Module):
    def __init__(self, num_classes=200, stochastic_depth_prob=0.2, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(MMConvNext, self).__init__()
        if wheter_concate:
            raise ValueError("Concatenation is not supported for this architecture")
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.convnext_small(num_classes=num_classes, stochastic_depth_prob=stochastic_depth_prob)
        model.features[0][0] = _replace_input_conv(model.features[0][0], input_channels)
        
        self.left_branch = model.features[:len(model.features) // 2]
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = model.features[len(model.features) // 2:]
        
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        z = torch.zeros(1, input_channels, img_height, img_width)
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True, return_features=False):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        # Processing noise or zeros by the active branch is for demonstration purposes only, it makes no sense in practice
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            x1 = self.left_branch(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = x1.new_empty((x1.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = x1.new_zeros((x1.size(0), self.channels_out, self.height, self.width))
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
                x2 = x2.new_empty((x2.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = x2.new_zeros((x2.size(0), self.channels_out, self.height, self.width))
            else:
                raise ValueError("Invalid right branch intervention")
            
        features_left, features_right = x1, x2
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        y = self.main_branch(y)
        y = self.avgpool(y)
        y = self.classifier(y)
        return (y, features_left, features_right) if return_features else y