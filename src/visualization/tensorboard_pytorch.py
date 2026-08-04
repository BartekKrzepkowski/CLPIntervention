import os

from torch.utils.tensorboard import SummaryWriter


class TensorboardPyTorch:
    def __init__(self, config):
        self.whether_use_wandb = bool(config.logger_config.get("whether_use_wandb", False))
        os.makedirs(config.logger_config["log_dir"], exist_ok=True)
        self.wandb = None
        if self.whether_use_wandb:
            import wandb

            self.wandb = wandb
            wandb.init(
                entity=config.logger_config.get("entity") or os.environ.get("WANDB_ENTITY"),
                project=config.logger_config["project_name"],
                name=config.exp_name,
                config=dict(config),
                dir=config.logger_config["log_dir"],
                mode=config.logger_config["mode"],
            )
            if wandb.patched["tensorboard"]:
                wandb.tensorboard.unpatch()
            wandb.tensorboard.patch(
                root_logdir=config.logger_config["log_dir"], pytorch=True, save=False
            )
        self.writer = SummaryWriter(log_dir=config.logger_config["log_dir"])

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def log_model(self, model, criterion, log=None, log_freq=1000, log_graph=True):
        # A bimodal graph requires two representative input tensors; the generic
        # trainer does not own such samples. Scalar and histogram logging remains active.
        return None

    def log_plots(self, images, step=None):
        for tag, image in images.items():
            self.writer.add_figure(tag, image, global_step=step)

    def log_histogram(self, values, step):
        for tag, value in values.items():
            self.writer.add_histogram(tag, value, global_step=step)

    def log_scalars(self, values, step):
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, global_step=step)
