import torch
import torch.nn as nn


class BiModalModelwithPretrainedBranches(nn.Module):
    """Attach frozen unimodal teachers to a bimodal student model."""

    def __init__(self, main_model, left_branch_pretrained, right_branch_pretrained):
        super().__init__()
        self.main_model = main_model
        self.left_branch_pretrained = left_branch_pretrained
        self.right_branch_pretrained = right_branch_pretrained
        for teacher in (self.left_branch_pretrained, self.right_branch_pretrained):
            teacher.requires_grad_(False)
            teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        # Teacher BatchNorm/dropout state must remain fixed.
        self.left_branch_pretrained.eval()
        self.right_branch_pretrained.eval()
        return self

    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None,
                enable_left_branch=True, enable_right_branch=True):
        return self.main_model(
            x1,
            x2,
            left_branch_intervention=left_branch_intervention,
            right_branch_intervention=right_branch_intervention,
            enable_left_branch=enable_left_branch,
            enable_right_branch=enable_right_branch,
            return_features=True,
        )

    @torch.no_grad()
    def teacher_features(
        self,
        x1,
        x2,
        *,
        enable_left_branch=True,
        enable_right_branch=True,
    ):
        left = (
            self.left_branch_pretrained(x1)
            if enable_left_branch
            else None
        )
        right = (
            self.right_branch_pretrained(x2)
            if enable_right_branch
            else None
        )
        return left, right
