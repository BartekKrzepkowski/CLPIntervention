import os

from clearml import Task


class ClearMLLogger:
    def __init__(self, config):
        self.project_name = config.logger_config["project_name"]
        self.task_name = config.exp_name
        access_key = os.environ.get("CLEARML_ACCESS_KEY")
        secret_key = os.environ.get("CLEARML_SECRET_KEY")
        if access_key and secret_key:
            Task.set_credentials(
                web_host="https://app.clear.ml",
                api_host="https://api.clear.ml",
                files_host="https://files.clear.ml",
                key=access_key,
                secret=secret_key,
            )
        os.makedirs(config.logger_config["log_dir"], exist_ok=True)
        self.task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            output_uri=config.logger_config["log_dir"],
        )
        self.task.connect(config.logger_config["hyperparameters"])
        self.writer = self.task.get_logger()

    def close(self):
        self.task.close()

    def log_model(self, model, criterion, log=None, log_freq=1000, log_graph=True):
        # ClearML captures the framework automatically after Task.init().
        return None

    def log_histogram(self, values, step):
        for tag, value in values.items():
            title, _, series = tag.partition("/")
            if hasattr(value, "detach"):
                value = value.detach().cpu().reshape(-1, 1).numpy()
            self.writer.report_histogram(
                title=title,
                series=series or "values",
                values=value,
                iteration=step,
            )

    def log_scalars(self, values, step):
        for tag, value in values.items():
            title, _, series = tag.partition("/")
            self.writer.report_scalar(
                title=title,
                series=series or "value",
                value=value,
                iteration=step,
            )

    def log_plots(self, images, step=None):
        iteration = 0 if step is None else step
        for tag, figure in images.items():
            title, _, series = tag.partition("/")
            self.writer.report_matplotlib_figure(
                title=title,
                series=series or "figure",
                figure=figure,
                iteration=iteration,
            )
