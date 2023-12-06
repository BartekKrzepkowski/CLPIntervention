from copy import deepcopy

import torch
import torchvision

from src.utils.utils_model import infer_dims_from_blocks


class MMEffNetV2S(torch.nn.Module):
    def __init__(self, num_classes=200, dropout=0.2, stochastic_depth_prob=0.2, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(MMEffNetV2S, self).__init__()
        from math import ceil
        assert wheter_concate == False, "Concatenation is not supported yet"
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.efficientnet_v2_s(num_classes=num_classes, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
        
        self.left_branch = model.features[:len(model.features) // 2]
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = model.features[len(model.features) // 2:]
        
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        z = torch.randn(1, input_channels, img_height, ceil(img_width * (overlap / 2 + 0.5)))
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
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
        y = self.classifier(y)
        return y
    


class ResNet18PyTorch(torch.nn.Module):
    def __init__(self, num_classes=200, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(ResNet18PyTorch, self).__init__()
        from math import ceil
        assert wheter_concate == False, "Concatenation is not supported yet"
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.resnet18(num_classes=num_classes)
        
        # self.model.number_of_classes = num_classes
        # self.model.input_size = img_height
        # self.model.input_channels = input_channels
        
        self.left_branch = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2)
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = torch.nn.Sequential(model.layer3, model.layer4)
        
        self.avgpool = model.avgpool
        self.classifier = model.fc
        
        z = torch.randn(1, input_channels, img_height, ceil(img_width * (overlap / 2 + 0.5)))
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
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
        y = self.classifier(y)
        return y
    
    
class MMConvNext(torch.nn.Module):
    def __init__(self, num_classes=200, stochastic_depth_prob=0.2, img_height=64, img_width=64, input_channels=3, overlap=0.0, eps=1e-5, wheter_concate=False):
        super(MMConvNext, self).__init__()
        from math import ceil
        assert wheter_concate == False, "Concatenation is not supported yet"
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        model = torchvision.models.convnext_small(num_classes=num_classes, stochastic_depth_prob=stochastic_depth_prob)
        
        self.left_branch = model.features[:len(model.features) // 2]
        self.right_branch = deepcopy(self.left_branch)
        self.main_branch = model.features[len(model.features) // 2:]
        
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        z = torch.randn(1, input_channels, img_height, ceil(img_width * (overlap / 2 + 0.5)))
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
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
        y = self.classifier(y)
        return y