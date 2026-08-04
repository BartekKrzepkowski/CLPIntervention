"""UMT trainer: base CLP loop with representation distillation."""

import torch

from src.trainer.trainer_classification_mm_clp import (
    TrainerClassification as BaseTrainerClassification,
)


class TrainerClassification(BaseTrainerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss()

    def compute_loss(self, x_left, x_right, targets, config):
        predictions, left_features, right_features = self.model(
            x_left,
            x_right,
            left_branch_intervention=config.extra["left_branch_intervention"],
            right_branch_intervention=config.extra["right_branch_intervention"],
            enable_left_branch=config.extra["enable_left_branch"],
            enable_right_branch=config.extra["enable_right_branch"],
        )
        teacher_left, teacher_right = self.model.teacher_features(
            x_left,
            x_right,
            enable_left_branch=config.extra["enable_left_branch"],
            enable_right_branch=config.extra["enable_right_branch"],
        )
        loss, evaluators = self.criterion(predictions, targets)

        distillation_loss = predictions.new_zeros(())
        if config.extra["enable_left_branch"]:
            mse_left = self.mse_loss(
                left_features.flatten(start_dim=1), teacher_left.flatten(start_dim=1)
            )
            evaluators["distillation/mse_left"] = mse_left.item()
            distillation_loss = distillation_loss + mse_left
        if config.extra["enable_right_branch"]:
            mse_right = self.mse_loss(
                right_features.flatten(start_dim=1), teacher_right.flatten(start_dim=1)
            )
            evaluators["distillation/mse_right"] = mse_right.item()
            distillation_loss = distillation_loss + mse_right

        classification_loss = loss
        weighted_distillation_loss = config.distill * distillation_loss
        total_loss = classification_loss + weighted_distillation_loss
        evaluators["classification_loss"] = classification_loss.item()
        evaluators["distillation/loss"] = distillation_loss.item()
        evaluators["distillation/weighted_loss"] = weighted_distillation_loss.item()
        evaluators["loss"] = total_loss.item()
        return total_loss, evaluators


def validation_controlled_umt_trainer_class():
    """Build the cooperative UMT/validation trainer without an import cycle."""
    from src.trainer.trainer_validation_clp import ValidationControlledTrainer

    class ValidationControlledUMTTrainer(
        TrainerClassification, ValidationControlledTrainer
    ):
        """Validation phase control with the UMT representation loss."""

    return ValidationControlledUMTTrainer
