from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm, trange

from src.data.loaders import Loaders
from src.modules.metrics import RunStatsBiModal
from src.utils.common import LOGGERS_NAME_MAP
from src.modules.metrics import Stiffness
from src.modules.hooks import Hooks
from src.utils.utils_trainer import adjust_evaluators, adjust_evaluators_pre_log, create_paths, save_model
from src.utils.utils_optim import clip_grad_norm


class TrainerClassification:
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra):
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = None
        self.global_step = None

        self.run_stats = RunStatsBiModal(model, optim)
        # self.stiffness = Stiffness(model, **extra)

    def run_exp1(self, config):
        batch_size = config.logger_config['hyperparameters']['loaders']['batch_size']
        num_workers = config.logger_config['hyperparameters']['loaders']['num_workers']
        dataset_name = config.logger_config['hyperparameters']['type_names']['dataset']
        self.train_loader = Loaders(dataset_name=dataset_name)
        
        self.manual_seed(config)
        self.at_exp_start(config)

        if config.extra['window'] != 0:
            print('Entering deficit phase!!!')
            self.loaders['train'] = self.train_loader.get_blurred_loader(batch_size, is_train=True, num_workers=num_workers)
            self.run_loop(-config.extra['window'], 0, config)
            print('Leaving deficit phase!!!')

        self.criterion.fpw = 0.0
        self.loaders['train'] = self.train_loader.get_proper_loader(batch_size, is_train=True, num_workers=num_workers)
        self.run_loop(0, config.epoch_end_at, config)

        self.logger.close()
        save_model(self.model, self.save_path(self.global_step))
        
    def run_exp1_reverse(self, config):
        batch_size = config.logger_config['hyperparameters']['loaders']['batch_size']
        num_workers = config.logger_config['hyperparameters']['loaders']['num_workers']
        dataset_name = config.logger_config['hyperparameters']['type_names']['dataset']
        self.train_loader = Loaders(dataset_name=dataset_name)
        
        self.manual_seed(config)
        self.at_exp_start(config)

        if config.extra['window'] != 0:
            print('Entering deficit phase!!!')
            self.loaders['train'] = self.train_loader.get_proper_loader(batch_size, is_train=True, num_workers=num_workers)
            self.run_loop(-config.extra['window'], 0, config)
            print('Leaving deficit phase!!!')

        self.criterion.fpw = 0.0
        self.loaders['train'] = self.train_loader.get_blurred_loader(batch_size, is_train=True, num_workers=num_workers)
        self.run_loop(0, config.epoch_end_at, config)

        self.logger.close()
        save_model(self.model, self.save_path(self.global_step))


    def run_exp2(self, config):
        batch_size = config.logger_config['hyperparameters']['loaders']['batch_size']
        num_workers = config.logger_config['hyperparameters']['loaders']['num_workers']
        dataset_name = config.logger_config['hyperparameters']['type_names']['dataset']
        self.train_loader = Loaders(dataset_name=dataset_name)

        self.manual_seed(config)
        self.at_exp_start(config)

        if config.extra['window'] != 0:
            fpw = self.criterion.fpw
            self.criterion.fpw = 0.0
            self.loaders['train'] = self.train_loader.get_proper_loader(batch_size, is_train=True, num_workers=num_workers)
            self.run_loop(0, config.extra['window'], config)
            self.criterion.fpw = fpw

        self.loaders['train'] = self.train_loader.get_blurred_loader(batch_size, is_train=True, num_workers=num_workers)
        self.run_loop(config.extra['window'], config.extra['window'] + 40, config)
        self.criterion.fpw = 0.0
        self.loaders['train'] = self.train_loader.get_proper_loader(batch_size, is_train=True, num_workers=num_workers)
        self.run_loop(config.extra['window'] + 40, config.extra['window'] + 40 + config.epoch_end_at, config)

        self.logger.close()
        save_model(self.model, self.save_path(self.global_step))
    
    def run_loop(self, epoch_start_at, epoch_end_at, config):
        """
        Main method of trainer.
        Set seed, run train-val in the loop.
        Args:
            config (dict): Consists of:
                epoch_start (int): A number representing the beginning of run
                epoch_end (int): A number representing the end of run
                grad_accum_steps (int):
                step_multi (int):
                base_path (str): Base path
                exp_name (str): Base name of experiment
                logger_name (str): Logger type
                random_seed (int): Seed generator
        """
        for epoch in trange(epoch_start_at, epoch_end_at, desc='run_exp',
                            leave=True, position=0, colour='green', disable=config.whether_disable_tqdm):
            self.epoch = epoch
            self.model.train()
            self.run_epoch(phase='train', config=config)
            self.model.eval()
            # with torch.no_grad():
                # self.hooks.disable()
            self.run_epoch(phase='test_proper', config=config)
            self.run_epoch(phase='test_blurred', config=config)
                # self.hooks.enable()

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
        config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_config["logger_name"]}'
        self.logger = LOGGERS_NAME_MAP[config.logger_config['logger_name']](config)
        # self.stiffness.logger = self.logger
        # self.hooks = Hooks(self.model, self.logger, 'dead_relu')
        # self.hooks.register_hooks([torch.nn.ReLU])

    def run_epoch(self, phase, config):
        """
        Run single epoch
        Args:
            phase (str): phase of the trening
            config (dict):
        """
        running_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
            'traces': defaultdict(float)
        }
        epoch_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
            'traces': defaultdict(float)
        }
        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}',
                            leave=False, position=1, total=loader_size, colour='red', disable=config.whether_disable_tqdm)
        self.global_step = self.epoch * loader_size
        for i, data in enumerate(progress_bar):
            self.global_step += 1
            x_true, y_true = data
            x_true, y_true = x_true.to(config.device), y_true.to(config.device)
            # y_pred = self.model(x_true)
            loss, evaluators, traces = self.criterion(x_true, y_true)
            step_assets = {
                'evaluators': evaluators,
                'denom': y_true.size(0),
                'traces': traces
            }
            if 'train' == phase:
                loss /= config.grad_accum_steps
                loss.backward()
                if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == loader_size:
                    if config.clip_value > 0:
                        norm = clip_grad_norm(torch.nn.utils.clip_grad_norm_, self.model, config.clip_value)
                        step_assets['evaluators']['run_stats/model_gradient_norm_squared_from_pytorch'] = norm.item() ** 2
                    self.optim.step()
                    if self.run_stats is not None:
                        step_assets['evaluators'] = self.run_stats(step_assets['evaluators'], 'l2')
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optim.zero_grad()
                else: # problem z tym że nie przy nie aktualizacji gradientu nie ma sensu liczyć długości trajektorii
                    if self.run_stats is not None:
                        norm = self.run_stats.model_gradient_norm()
                        step_assets['evaluators']['run_stats/model_gradient_norm_squared_overall'] = norm ** 2
                loss *= config.grad_accum_steps
                if config.stiff_multi and self.global_step % (config.grad_accum_steps * config.stiff_multi) == 0:
                    self.hooks.disable()
                    self.stiffness.log_stiffness(self.global_step)
                    self.hooks.enable()
                # if self.hooks is not None:
                #     self.hooks.write_to_tensorboard(self.global_step)
            
            ### LOGGING ###
            running_assets = self.update_assets(running_assets, step_assets, step_assets['denom'], 'running', phase)

            whether_save_model = config.save_multi and (i + 1) % (config.grad_accum_steps * config.save_multi) == 0
            whether_log = (i + 1) % (config.grad_accum_steps * config.log_multi) == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_save_model and 'train' in phase:
                save_model(self.model, self.save_path(self.global_step))

            if whether_log or whether_epoch_end:
                epoch_assets = self.update_assets(epoch_assets, running_assets, 1.0, 'epoch', phase)

            if whether_log:
                self.log(running_assets, phase, 'running', progress_bar, self.global_step)
                running_assets['evaluators'] = defaultdict(float)
                running_assets['denom'] = 0.0
                running_assets['traces'] = defaultdict(float)

            if whether_epoch_end:
                self.log(epoch_assets, phase, 'epoch', progress_bar, self.epoch)

    def log(self, assets: Dict, phase: str, scope: str, progress_bar: tqdm, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        evaluators_log = adjust_evaluators_pre_log(assets['evaluators'], assets['denom'], round_at=4)
        evaluators_log[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(evaluators_log, step)
        progress_bar.set_postfix(evaluators_log)

        traces_log = adjust_evaluators_pre_log(assets['traces'], assets['denom'], round_at=4)
        self.logger.log_scalars(traces_log, global_step=step)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)

    def update_assets(self, assets_target: Dict, assets_source: Dict, multiplier, scope, phase: str):
        '''
        Update epoch assets
        Args:
            assets_target (Dict): Assets to which assets should be transferred
            assets_source (Dict): Assets from which assets should be transferred
            multiplier (int): Number to get rid of the average
            scope (str): Either running or epoch
            phase (str): Phase of the traning
        '''
        assets_target['evaluators'] = adjust_evaluators(assets_target['evaluators'], assets_source['evaluators'],
                                                        multiplier, scope, phase)
        assets_target['denom'] += assets_source['denom']
        scope_traces = 'running_trace_per_param/param' if scope == 'running' else scope
        assets_target['traces'] = adjust_evaluators(assets_target['traces'], assets_source['traces'],
                                                    multiplier, scope_traces, phase)
        return assets_target

    def manual_seed(self, config: defaultdict):
        """
        Set the environment for reproducibility purposes.
        Args:
            config (defaultdict): set of parameters
                usage of:
                    random seed (int):
                    device (torch.device):
        """
        import random
        import numpy as np
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if 'cuda' in config.device.type:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(config.random_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
