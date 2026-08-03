import inspect
import pytest
import torch

from src.visualization.wandb_logger import WandbLogger


LOGGER_METHODS = ("log_model", "log_scalars", "log_histogram", "log_plots", "close")


def test_logger_backends_share_the_runtime_protocol():
    logger_types = [WandbLogger]
    try:
        from src.visualization.clearml_logger import ClearMLLogger
        from src.visualization.tensorboard_pytorch import TensorboardPyTorch
    except ImportError:
        pass
    else:
        logger_types.extend((TensorboardPyTorch, ClearMLLogger))
    for logger_type in logger_types:
        assert all(callable(getattr(logger_type, method, None)) for method in LOGGER_METHODS)
        assert list(inspect.signature(logger_type.log_model).parameters) == [
            "self",
            "model",
            "criterion",
            "log",
            "log_freq",
            "log_graph",
        ]


def test_wandb_disabled_does_not_login(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.visualization.wandb_logger.wandb.login",
        lambda **kwargs: calls.append(("login", kwargs)),
    )
    monkeypatch.setattr(
        "src.visualization.wandb_logger.wandb.init",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    class Config(dict):
        __getattr__ = dict.__getitem__

    config = Config(
        exp_name="disabled-test",
        logger_config={
            "project_name": "test",
            "entity": None,
            "log_dir": str(tmp_path),
            "mode": "disabled",
        },
    )

    WandbLogger(config)

    assert [name for name, _ in calls] == ["init"]
    assert calls[0][1]["mode"] == "disabled"


def test_wandb_logger_propagates_step_and_histograms():
    class Writer:
        def __init__(self):
            self.calls = []

        def Histogram(self, value):
            return ("histogram", value)

        def log(self, values, step=None):
            self.calls.append((values, step))

    logger = object.__new__(WandbLogger)
    logger.writer = Writer()
    logger.log_scalars({"loss": 1.0}, 7)
    logger.log_histogram({"weights": torch.tensor([1.0])}, 8)

    assert logger.writer.calls[0] == ({"loss": 1.0}, 7)
    assert logger.writer.calls[1][0]["weights"][0] == "histogram"
    assert logger.writer.calls[1][1] == 8


def test_tensorboard_and_clearml_accept_shared_histogram_shape():
    pytest.importorskip("clearml")
    pytest.importorskip("tensorboard")
    from src.visualization.clearml_logger import ClearMLLogger
    from src.visualization.tensorboard_pytorch import TensorboardPyTorch

    class TensorboardWriter:
        def __init__(self):
            self.histograms = []

        def add_histogram(self, tag, value, global_step):
            self.histograms.append((tag, value, global_step))

    tensorboard = object.__new__(TensorboardPyTorch)
    tensorboard.writer = TensorboardWriter()
    tensorboard.log_histogram({"branch/weights": torch.ones(2)}, 4)
    assert tensorboard.writer.histograms[0][0::2] == ("branch/weights", 4)

    class ClearMLWriter:
        def __init__(self):
            self.histograms = []

        def report_histogram(self, **kwargs):
            self.histograms.append(kwargs)

    clearml = object.__new__(ClearMLLogger)
    clearml.writer = ClearMLWriter()
    clearml.log_histogram({"branch/weights": torch.ones(2)}, 5)
    assert clearml.writer.histograms[0]["title"] == "branch"
    assert clearml.writer.histograms[0]["series"] == "weights"
    assert clearml.writer.histograms[0]["iteration"] == 5
