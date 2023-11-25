import os

import wandb


class WandbLogger:
    def __init__(self, config):
        self.project = config.logger_config['project_name']
        self.writer = wandb
        self.writer.login(key=os.environ['WANDB_API_KEY'])
        if not os.path.isdir(config.logger_config['log_dir']):
            os.makedirs(config.logger_config['log_dir'])
        self.writer.init(
            entity=config.logger_config['entity'],
            project=config.logger_config['project_name'],
            name=config.exp_name,
            config=dict(config),
            # config=OmegaConf.to_container(config, resolve=True),  # czy nie wystarczy dict(config)?
            dir=config.logger_config['log_dir'],
            mode=config.logger_config['mode']
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log, log_freq: int=1000, log_graph: bool=True):
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)
    
    def log_histograms(self, hists):
        self.writer.log(hists)

    def log_scalars(self, evaluators, step):
        self.writer.log(evaluators)


