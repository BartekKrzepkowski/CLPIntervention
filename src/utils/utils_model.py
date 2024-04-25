import json
from math import sqrt

from torch import nn


def default_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
# def default_init(model):
#     if hasattr(m, 'reset_parameters'):
#         m.reset_parameters()
            

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return count


def infer_dims_from_blocks(blocks, x, scaling_factor):
    for block in blocks:
        x = block(x)
    _, channels_out, height, width = x.shape
    pre_mlp_channels = channels_out * scaling_factor
    return channels_out, height, width, pre_mlp_channels


def load_model_specific_params(model_name):
    model_params = json.load(open(f'src/configs/{model_name}.json', 'r'))
    return model_params

def change_activation(model, old_activation, new_activation):
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation())
        else:
            change_activation(module, old_activation, new_activation)
            
            
import torch.nn as nn

class BiModalModelwithPretrainedBranches(nn.Module):
    def __init__(self, main_model, left_branch_pretrained, right_branch_pretrained):
        super(BiModalModelwithPretrainedBranches, self).__init__()
        self.main_model = main_model
        self.left_branch_pretrained = left_branch_pretrained
        self.right_branch_pretrained = right_branch_pretrained

    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None,
                enable_left_branch=True, enable_right_branch=True):
  
        out = self.main_model(x1, x2,
                              left_branch_intervention=left_branch_intervention,
                              right_branch_intervention=right_branch_intervention,
                              enable_left_branch=enable_left_branch,
                              enable_right_branch=enable_right_branch)
        return out
