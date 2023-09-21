#!/usr/bin/env python3
import os

import numpy as np
import torch
from math import ceil

import omegaconf

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed, find_paths
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_dual_clp import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(exp, window, epochs, model_path):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_ACCUM_STEPS = 1
    NUM_CLASSES = 10
    RANDOM_SEED = 83
    OVERLAP = 0.0
    
    type_names = {
        'model': 'dual_simple_cnn',
        'criterion': 'fp',
        'dataset': 'dual_cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare model ════════════════════════ #
    
    
    N = 1
    NUM_FEATURES = 3
    DIMS = [NUM_FEATURES, 32] + [64] * N + [128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3, 3] * (N + 1), 'strides': [1, 1] * (N + 1), 'paddings': [1, 1] * (N + 1), 'whether_pooling': [False, True] * (N + 1)}
    model_params = {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS, 'overlap': OVERLAP}
    
    model = prepare_model(type_names['model'], model_params=model_params, model_path=model_path).to(device)
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    
    FP = 0.0#1e-2
    criterion_params = {'model': model, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': False, 'fpw': FP}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)
    
    criterion_params['model'] = None
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': OVERLAP}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = 2e-1
    MOMENTUM = 0.0
    WD = 0.0
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * (window + epochs)
    # print(T_max//window, T_max-3*T_max//window, 3*T_max//window)
    # h_params_overall['scheduler'] = {'eta_max':LR, 'eta_medium':1e-2, 'eta_min':1e-6, 'warmup_iters2': 3*T_max//window, 'inter_warmups_iters': T_max-3*T_max//window, 'warmup_iters1': 3*T_max//window, 'milestones':[], 'gamma':1e-1}
    optim_params = {'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WD}
    scheduler_params = None
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    
    model_path = model_path.split('_')
    window2 = model_path[-3]
    window1 = int(model_path[-1].split('.')[0])
    
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_fp_{FP}_lr_{LR}_wd_{WD}'
    EXP_NAME = f'{GROUP_NAME}_window_{window} overlap={OVERLAP}, intervention deactivation, trained with phase1={window1} and phase2={window2} '
    PROJECT_NAME = 'Critical_Periods_Interventions'
    ENTITY_NAME = 'ideas_cv'

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
    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    # x_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x.pt').to(device)
    # y_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_y.pt').to(device)
    
    # x_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x.pt').to(device)
    # y_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_y.pt').to(device) 
    
    
    # ════════════════════════ prepare trainer ════════════════════════ #
    
    
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra': None,#{'x_true1': x_data_proper, 'y_true1': y_data_proper, 'x_true2': x_data_blurred, 'y_true2': y_data_blurred, 'num_classes': NUM_CLASSES},
    }
    
    trainer = TrainerClassification(**params_trainer)


    # ════════════════════════ prepare run ════════════════════════ #


    CLIP_VALUE = 100.0
    W_ = ceil(32 * (OVERLAP / 2 + 0.5))
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    logger_config = {'logger_name': 'tensorboard',
                     'project_name': PROJECT_NAME,
                     'entity': ENTITY_NAME,
                     'hyperparameters': h_params_overall,
                     'whether_use_wandb': True,
                     'layout': ee_tensorboard_layout(params_names), # is it necessary?
                     'mode': 'online',
                    #  'dummy_variable': (torch.randn(1, 3, 32, W_).to(device), torch.randn(1, 3, 32, W_).to(device))
    }
    extra = {'window': window,
             'overlap': OVERLAP,
             'left_branch_intervention': 'deactivation',
             'right_branch_intervention': None,
             'enable_left_branch': False,
             'enable_right_branch': True
    }
    
    config = omegaconf.OmegaConf.create()
    
    config.epoch_start_at = 0
    config.epoch_end_at = epochs
    
    config.grad_accum_steps = GRAD_ACCUM_STEPS
    config.log_multi = 1#(T_max // epochs) // 10
    config.save_multi = 0#T_max // 10
    config.stiff_multi = (T_max // (window + epochs)) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = 'reports'
    config.exp_name = EXP_NAME
    config.extra = extra
    config.logger_config = logger_config
    
    
    # ════════════════════════ run ════════════════════════ #
    
    
    if exp == 'deficit':
        trainer.run_exp1(config)
    elif exp == 'sensitivity':
        trainer.run_exp2(config)
    elif exp == 'deficit_reverse':
        trainer.run_exp1_reverse(config)
    elif exp == 'intervention':
        trainer.run_exp3(config)
    else:
        raise ValueError('exp should be either "deficit" or "sensitivity"')


if __name__ == "__main__":
    EPOCHS = 200
    main_dir = 'reports'
    for directory in sorted(os.listdir(main_dir)):
        if 'deficit' in directory:
            path = os.path.join(main_dir, directory)
            model_paths = find_paths(path)
            for model_path in model_paths:
                if 'step_16000' in model_path and 'window_120' in model_path:
                    for window in np.linspace(0, 200, 6):
                        objective('intervention', int(window), EPOCHS, model_path)
