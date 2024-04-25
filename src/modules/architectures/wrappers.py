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