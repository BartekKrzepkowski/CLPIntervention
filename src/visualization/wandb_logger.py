import os

import wandb


class WandbLogger:
    def __init__(self, config):
        self.project = config.logger_config["project_name"]
        self.writer = wandb
        api_key = os.environ.get("WANDB_API_KEY")
        mode = config.logger_config["mode"]
        if api_key and mode != "disabled":
            self.writer.login(key=api_key)
        os.makedirs(config.logger_config["log_dir"], exist_ok=True)
        self.writer.init(
            entity=config.logger_config["entity"],
            project=config.logger_config["project_name"],
            name=config.exp_name,
            config=dict(config),
            dir=config.logger_config["log_dir"],
            mode=mode,
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log=None, log_freq=1000, log_graph=True):
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)

    def log_histogram(self, values, step):
        histograms = {}
        for name, value in values.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            histograms[name] = self.writer.Histogram(value)
        self.writer.log(histograms, step=step)

    def log_scalars(self, values, step):
        self.writer.log(values, step=step)

    def log_plots(self, images, step=None):
        self.writer.log(images, step=step)
