#!/usr/bin/env python3
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf

# from rich.traceback import install
# install(show_locals=True)

from src.data.transforms import TRANSFORMS_BLURRED_RIGHT_NAME_MAP
from src.modules.aux_modules import TraceFIM, RepresentationsSpectra, DeadReLU
from src.modules.aux_modules_collapse import GradientsSpectralStiffness
from src.modules.metrics import RunStatsBiModal
from src.trainer.trainer_classification_mm_clp import TrainerClassification
from src.utils.prepare import prepare_criterion, prepare_loaders_clp, prepare_model, prepare_optim_and_scheduler
from src.utils.utils_criterion import get_samples_weights, load_criterion_specific_params
from src.utils.utils_data import count_classes, create_dataloader
from src.utils.utils_model import load_model_specific_params, change_activation
from src.utils.utils_trainer import manual_seed


def objective(exp_name, model_name, dataset_name, lr, wd, epochs):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 125
    CLIP_VALUE = 0.0
    LOGS_PER_EPOCH = 0  # 0 means each batch
    LR_LAMBDA = 1.0
    NUM_WORKERS = 12
    OVERLAP = 0.0
    RANDOM_SEED = 83
    
    type_names = {
        'model': model_name,
        'criterion': 'cls',
        'dataset': dataset_name,
        'optim': 'sgd',
        'scheduler': 'multiplicative'
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    frac = '0'  # tyle % przykładów nie zostanie rozmazanych, 0 - wszystkie będą rozmazane, 100 - wszystkie będą nierozmazane
    subset = np.load(f'data/{type_names["dataset"]}_subset_{frac}.npy') if frac != '0' else None
    dataset_params = {'overlap': OVERLAP, 'subset': subset}
    loader_params = {'batch_size': BATCH_SIZE, 'pin_memory': True, 'num_workers': NUM_WORKERS}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    logging.info('Loaders prepared.')
    del dataset_params['subset']
    
    num_classes = count_classes(loaders['train'].dataset)

    
    # ════════════════════════ prepare model ════════════════════════ #


    input_channels, img_height, img_width = loaders['train'].dataset[0][0][0].shape
    model_params = load_model_specific_params(type_names["model"])
    model_params = {
        'num_classes': num_classes,
        'input_channels': input_channels,
        'img_height': img_height,
        'img_width': img_width,
        'overlap': OVERLAP,
        **model_params
    }
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)
    # change_activation(model, torch.nn.ReLU, torch.nn.GELU)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    

    samples_weights = get_samples_weights(loaders, num_classes).to(device)  # to handle class imbalance
    criterion_params = load_criterion_specific_params(type_names["criterion"])
    criterion_params['weight'] = samples_weights
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)
    logging.info('Criterion prepared.')
    
    criterion_params['weight'] = samples_weights.tolist()  # problem with omegacong with primitive type
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    

    batches_per_epoch = len(loaders["train"])
    T_max = batches_per_epoch * epochs
    
    optim_params = {'lr': lr, 'weight_decay': wd}
    scheduler_params = {'lr_lambda': lambda epoch: LR_LAMBDA}
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    logging.info('Optimizer and scheduler prepared.')
    
    scheduler_params['lr_lambda'] = LR_LAMBDA  # problem with omegacong with primitive type
    
    
    # ════════════════════════ prepare wandb params ════════════════════════ #


    quick_name = f'left modality pretraining'
    GROUP_NAME = f'{type_names["dataset"]}, {type_names["model"]}, {type_names["optim"]}, epochs={epochs}_overlap={OVERLAP}_lr={lr}_wd={wd}_lambda={LR_LAMBDA}'
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
    
    
    held_out_train = {}
    held_out_train['proper_x_left'] = torch.load(f'data/train_{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    held_out_train['proper_x_right'] = torch.load(f'data/train_{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    held_out_train['blurred_x_right'] = torch.load(f'data/train_{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    held_out_train['y'] = torch.load(f'data/train_{type_names["dataset"]}_held_out_y.pt').to(device)
    
    held_out_val = {}
    held_out_val['proper_x_left'] = torch.load(f'data/val_{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    held_out_val['proper_x_right'] = torch.load(f'data/val_{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    held_out_val['blurred_x_right'] = torch.load(f'data/val_{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    held_out_val['y'] = torch.load(f'data/val_{type_names["dataset"]}_held_out_y.pt').to(device)
    
    loaders_rank = deepcopy(loaders)
    indices = torch.load(f'data/train_{type_names["dataset"]}_indices.pt').tolist()
    loaders_rank['train_proper'] = create_dataloader(loaders_rank['train'].dataset, indices, loader_params)
    loaders_rank['train_blurred'] = create_dataloader(loaders_rank['train'].dataset, indices, loader_params)
    loaders_rank['train_blurred'].transform2 = TRANSFORMS_BLURRED_RIGHT_NAME_MAP[type_names['dataset']](OVERLAP)
    del loaders_rank['train']
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    extra_modules['run_stats'] = RunStatsBiModal(model, optim)
    extra_modules['trace_fim_train'] = TraceFIM(held_out_train, model, num_classes=num_classes, postfix='train', m_sampling=5)
    extra_modules['trace_fim_test'] = TraceFIM(held_out_val, model, num_classes=num_classes, postfix='val', m_sampling=5)
    
    
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

    config = OmegaConf.create()
    
    config.exp_starts_at_epoch = 0
    config.exp_ends_at_epoch = epochs
    
    config.log_multi = batches_per_epoch // (LOGS_PER_EPOCH if LOGS_PER_EPOCH != 0 else batches_per_epoch)
    config.run_stats_multi = batches_per_epoch // 2
    config.fim_trace_multi = batches_per_epoch // 2
    config.stiffness_multi = batches_per_epoch * 20
    config.rank_multi = batches_per_epoch * 20
    
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
    logging.info(f'The dataset has {len(loaders["train"].dataset)} train samples, {len(loaders["test_proper"].dataset)} test samples, {num_classes} classes, each image has a dimension of {input_channels}x{img_height}x{img_width} and each epoch has {batches_per_epoch} batches.')
    
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
    elif exp_name == 'left_modality_pretraining_proper':
        trainer.run_left_modality_pretraining_proper(config)
    elif exp_name == 'right_modality_pretraining_proper':
        trainer.run_right_modality_pretraining_proper(config)
    elif exp_name == 'right_modality_pretraining_blurred':
        trainer.run_right_modality_pretraining_blurred(config)
    else:
        raise ValueError('Exp is not recognized.')


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    
    logging.basicConfig(
            format=(
                '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
            ),
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
            force=True,
        )
    logging.info(f'Script started model s-{conf.model_name} on dataset s-{conf.dataset_name} with lr={conf.lr}, wd={conf.wd}, phase1={conf.phase1}, phase2={conf.phase2}.')
    
    objective('left_modality_pretraining_proper', conf.model_name, conf.dataset_name, conf.lr, conf.wd, conf.epochs)
