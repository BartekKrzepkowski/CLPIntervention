#!/usr/bin/env python3
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import omegaconf
import torch
from math import ceil

# from rich.traceback import install
# install(show_locals=True)

from src.modules.aux_modules import TraceFIM
from src.modules.metrics import RunStatsBiModal
from src.trainer.trainer_classification_mm_clp import TrainerClassification
from src.utils.prepare import prepare_criterion, prepare_loaders_clp, prepare_model, prepare_optim_and_scheduler
from src.utils.utils_criterion import get_samples_weights
from src.utils.utils_data import count_classes
from src.utils.utils_trainer import manual_seed


def objective(exp_name, phase3, lr, wd, phase1, phase2):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 125
    CLIP_VALUE = 0.0
    LOGS_PER_EPOCH = 0  # 0 means every batch
    LR_LAMBDA = 1.0
    NUM_WORKERS = 10
    OVERLAP = 0.0
    RANDOM_SEED = 83
    
    type_names = {
        'model': 'mm_simple_cnn',
        'criterion': 'cls',
        'dataset': 'mm_fmnist',
        'optim': 'sgd',
        'scheduler': 'multiplicative'
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'overlap': OVERLAP}
    loader_params = {'batch_size': BATCH_SIZE, 'pin_memory': True, 'num_workers': NUM_WORKERS}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    logging.info('Loaders prepared.')
    
    num_classes = count_classes(loaders['train'].dataset)

    
    # ════════════════════════ prepare model ════════════════════════ #

    
    N = 3
    NUM_FEATURES = 1
    DIMS = [NUM_FEATURES, 32] + [256] * N + [128, NUM_CLASSES]
    
    model_params = {'layers_dim': DIMS, 'activation_name': 'relu', 'overlap': OVERLAP, 'num_features': NUM_FEATURES, 'pre_mlp_depth': N}
    
    model = prepare_model(type_names['model'], model_params=model_params, model_path=model_checkpoint).to(device)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    

    samples_weights = get_samples_weights(loaders, num_classes).to(device)  # to handle class imbalance
    criterion_params = {'criterion_name': 'ce', 'weight': samples_weights}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)
    logging.info('Criterion prepared.')
    
    criterion_params['weight'] = samples_weights.tolist()  # problem with omegacong with primitive type
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    

    batches_per_epoch = len(loaders["train"])
    T_max = batches_per_epoch * phase3
    
    optim_params = {'lr': lr, 'weight_decay': wd}
    scheduler_params = {'lr_lambda': lambda epoch: LR_LAMBDA}
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    logging.info('Optimizer and scheduler prepared.')
    
    scheduler_params['lr_lambda'] = LR_LAMBDA  # problem with omegacong with primitive type
    
    
    # ════════════════════════ prepare wandb params ════════════════════════ #


    quick_name = f'intervention deactivation, trained with phase1={phase1} and phase2={phase2}'
    GROUP_NAME = f'{type_names["model"]}, {type_names["dataset"]}, {type_names["optim"]}, overlap={OVERLAP}_epochs={phase3}_lr={lr}_wd={wd}_lambda={LR_LAMBDA}'
    EXP_NAME = f'{exp_name}, {quick_name}, {GROUP_NAME}'

    h_params_overall = {
        'model': model_params,
        'criterion': criterion_params,
        'dataset': dataset_params,
        'loaders': loader_params,
        'optim': optim_params,
        'scheduler': scheduler_params,
        'type_names': type_names
    }   
 
 
    # ════════════════════════ prepare held out data ════════════════════════ #
    
    
    held_out = {}
    # held_out['proper_x_left'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    # held_out['proper_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    # held_out['blurred_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    extra_modules['run_stats'] = RunStatsBiModal(model, optim)
    # extra_modules['trace_fim'] = TraceFIM(held_out, model, num_classes=num_classes)
    
    
    # ════════════════════════ prepare trainer ════════════════════════ #
    
    
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
    }
    
    trainer = TrainerClassification(**params_trainer)
    logging.info('Trainer prepared.')


    # ════════════════════════ prepare run ════════════════════════ #


    logger_config = {'logger_name': 'wandb',
                     'entity': os.environ['WANDB_ENTITY'],
                     'project_name': os.environ['WANDB_PROJECT'],
                     'hyperparameters': h_params_overall,
                     'mode': 'online',
    }

    config = omegaconf.OmegaConf.create()
    
    config.exp_starts_at_epoch = 0
    config.exp_ends_at_epoch = phase3
    
    config.log_multi = batches_per_epoch // (LOGS_PER_EPOCH if LOGS_PER_EPOCH != 0 else batches_per_epoch)
    config.run_stats_multi = batches_per_epoch // 2
    config.fim_trace_multi = batches_per_epoch // 2
    
    config.clip_value = CLIP_VALUE
    config.overlap = OVERLAP
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = os.environ['REPORTS_DIR']
    config.exp_name = EXP_NAME
    config.logger_config = logger_config
    
    logging.info('Run prepared.')
    
    
    # ════════════════════════ run ════════════════════════ #
    
    
    logging.info(f'The built model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters and {sum(p.numel() for p in model.parameters() if not p.requires_grad)} non trainable parameters.')
    logging.info(f'The dataset has {len(loaders["train"].dataset)} train samples and {len(loaders["test_proper"].dataset)} test samples and {num_classes} classes and every epoch has {batches_per_epoch} batches.')
    
    if exp_name == 'phase1':
        trainer.run_phase1(config)
    elif exp_name == 'phase2':
        trainer.run_phase2(config)
    elif exp_name == 'phase3':
        trainer.run_phase3(config)
    elif exp_name == 'phase4':
        trainer.run_phase4(config)
    elif exp_name == 'all_at_once':
        trainer.run_all_at_once(config)
    else:
        raise ValueError('Exp is not recognized.')


if __name__ == "__main__":
    model_checkpoint = str(sys.argv[1])
    phase3 = int(sys.argv[2])
    lr = float(sys.argv[3])
    wd = float(sys.argv[4])
    phase1 = int(sys.argv[5])
    phase2 = int(sys.argv[6])
    
    logging.basicConfig(
            format=(
                '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
            ),
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
            force=True,
        )
    logging.info(f'Script started with phase3={phase3}, lr={lr}, wd={wd}, phase1={phase1}, phase2={phase2}')
    logging.info(f'Model checkpoint path: {model_checkpoint}')
    
    objective('phase3', model_checkpoint, phase3, lr, wd, phase1, phase2)
