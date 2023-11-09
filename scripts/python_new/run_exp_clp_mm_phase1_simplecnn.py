#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from math import ceil

import omegaconf

# from rich.traceback import install
# install(show_locals=True)

from src.data import transforms_fmnist
from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_dual_clp import TrainerClassification
from src.modules.aux_modules import TraceFIM
from src.modules.metrics import RunStatsBiModal

# TODO: 1) Napisz mm_simplecnn, 2) Napisz Dual_fminst


def objective(exp, epochs, lr, wd, N):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    GRAD_ACCUM_STEPS = 1
    NUM_CLASSES = 10
    RANDOM_SEED = 83
    OVERLAP = 0.0
    
    type_names = {
        'model': 'mm_simple_cnn',
        'criterion': 'cls',
        'dataset': 'dual_fmnist',
        'optim': 'sgd',
        'scheduler': 'multiplicative'
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare model ════════════════════════ #
    
    
    
    N = N
    NUM_FEATURES = 1
    DIMS = [NUM_FEATURES, 64] + [256] * N + [128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3, 3] * (N + 1), 'strides': [1, 1] * (N + 1), 'paddings': [1, 1] * (N + 1), 'whether_pooling': [False, True] * (N + 1)}
    model_params = {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS, 'overlap': OVERLAP, 'num_features': NUM_FEATURES, 'pre_mlp_depth': N}
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)
    print(model)
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': OVERLAP}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    loaders['train'].dataset.transform2 = transforms_fmnist.TRANSFORMS_NAME_MAP['transform_train_blurred'](32, 32, 1/4, OVERLAP)
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    
    class_counts = [0] * 10
    for _, label in loaders['train'].dataset:
        class_counts[label] += 1
    print(class_counts)
    samples_weights = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))]).to(device)
    FP = 0.0
    criterion_params = {'criterion_name': 'ce', 'weight': samples_weights}
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)
    criterion_params['weight'] = samples_weights.tolist()
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = lr
    MOMENTUM = 0.0
    WD = wd
    LR_LAMBDA = 1.0
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * epochs
    optim_params = {'lr': LR, 'weight_decay': WD}
    scheduler_params = {'lr_lambda': lambda epoch: LR_LAMBDA}
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    scheduler_params['lr_lambda'] = LR_LAMBDA # problem with omegacong with primitive type
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    ENTITY_NAME = 'gmum'
    PROJECT_NAME = 'Critical_Periods_Interventions'
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_fp_{FP}_lr_{LR}_wd_{WD}_lr_lambda_{LR_LAMBDA}_N_{N}'
    EXP_NAME = f'{GROUP_NAME} overlap={OVERLAP}, phase1'

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
    
    
    # DODAJ - POPRAWNE DANE
    print('liczba parametrów', sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    held_out = {}
    # held_out['proper_x_left'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    # held_out['proper_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    # held_out['blurred_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    # extra_modules['run_stats'] = RunStatsBiModal(model, optim)
    # extra_modules['trace_fim'] = TraceFIM(held_out, model, num_classes=NUM_CLASSES)
    
    
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


    # ════════════════════════ prepare run ════════════════════════ #


    CLIP_VALUE = 0.0
    W_ = ceil(32 * (OVERLAP / 2 + 0.5))
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    logger_config = {'logger_name': 'tensorboard',
                     'project_name': PROJECT_NAME,
                     'entity': ENTITY_NAME,
                     'hyperparameters': h_params_overall,
                     'whether_use_wandb': True,
                     'layout': ee_tensorboard_layout(params_names), # is it necessary?
                     'mode': 'online',
    }
    extra = {'window': 0,
             'overlap': OVERLAP,
             'left_branch_intervention': None,
             'right_branch_intervention': None,
             'enable_left_branch': True,
             'enable_right_branch': True
    }
    
    config = omegaconf.OmegaConf.create()
    
    config.epoch_start_at = 0
    config.epoch_end_at = epochs
    
    config.grad_accum_steps = GRAD_ACCUM_STEPS
    config.log_multi = 1#(T_max // epochs) // 10
    config.save_multi = 0#T_max // 10
    # config.stiff_multi = (T_max // (window + epochs)) // 2
    config.fim_trace_multi = (T_max // epochs) // 2
    config.run_stats_multi = (T_max // epochs) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = os.environ['REPORTS_DIR']
    config.exp_name = EXP_NAME
    config.extra = extra
    config.logger_config = logger_config
    config.checkpoint_path = None
    
    
    # ════════════════════════ run ════════════════════════ #
    
    
    if exp == 'deficit':
        trainer.run_exp1(config)
    elif exp == 'sensitivity':
        trainer.run_exp2(config)
    elif exp == 'deficit_reverse':
        trainer.run_exp1_reverse(config)
    elif exp == 'just_run':
        trainer.run_exp4(config)
    else:
        raise ValueError('exp should be either "deficit" or "sensitivity"')


if __name__ == "__main__":
    lr = float(sys.argv[1])
    wd = float(sys.argv[2])
    N = int(sys.argv[3])
    print(lr, wd)
    EPOCHS = 400
    objective('deficit_reverse', EPOCHS, lr, wd, N)
