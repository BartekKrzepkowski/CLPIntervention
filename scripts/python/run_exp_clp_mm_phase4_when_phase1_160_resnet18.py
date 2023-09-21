#!/usr/bin/env python3
import sys
from collections import defaultdict

import numpy as np
import torch
from math import ceil

import omegaconf

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_dual_clp import TrainerClassification
from src.modules.aux_modules import TraceFIM
from src.modules.metrics import RunStatsBiModal


def objective(exp, epochs, lr, wd, phase3):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_ACCUM_STEPS = 1
    NUM_CLASSES = 10
    RANDOM_SEED = 83
    OVERLAP = 0.0
    
    type_names = {
        'model': 'mm_resnet',
        'criterion': 'cls',
        'dataset': 'dual_cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare model ════════════════════════ #
    
    
    model_config = {'backbone_type': 'resnet18',
                    'only_features': False,
                    'batchnorm_layers': True,
                    'width_scale': 1.0,
                    'skips': True,
                    'modify_resnet': True,
                    'wheter_concate': False,
                    'overlap': OVERLAP,}
    model_params = {'model_config': model_config, 'num_classes': NUM_CLASSES, 'dataset_name': type_names['dataset']}
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    FP = 0.0
    criterion_params = {'criterion_name': 'ce'}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)
    
    
    criterion_params['model'] = None
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': OVERLAP}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = lr
    MOMENTUM = 0.0
    WD = wd
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * epochs
    # print(T_max//window, T_max-3*T_max//window, 3*T_max//window)
    # h_params_overall['scheduler'] = {'eta_max':LR, 'eta_medium':1e-2, 'eta_min':1e-6, 'warmup_iters2': 3*T_max//window, 'inter_warmups_iters': T_max-3*T_max//window, 'warmup_iters1': 3*T_max//window, 'milestones':[], 'gamma':1e-1}
    optim_params = {'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WD}
    scheduler_params = None
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    checkpoint_path = f'/raid/NFS_SHARE/home/b.krzepkowski/Github/CLPInterventions/reports/just_run, sgd, dual_cifar10, mm_resnet_fp_0.0_lr_0.55_wd_0.0 overlap=0.0, phase3, intervention deactivation, trained with phase1=160 and phase2=200 /2023-09-08_14-17-11/checkpoints/model_step_epoch_{phase3}.pth'
    phase1 = 160
    phase2 = 200
    quick_name = f'trained with phase1={phase1} and phase2={phase2} and phase3={phase3}'
    ENTITY_NAME = 'ideas_cv'
    PROJECT_NAME = 'Critical_Periods_Interventions'
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_fp_{FP}_lr_{LR}_wd_{WD}'
    EXP_NAME = f'{GROUP_NAME} overlap={OVERLAP}, phase4, {quick_name}'

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
    held_out['proper_x_left'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    held_out['proper_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    held_out['blurred_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    extra_modules['run_stats'] = RunStatsBiModal(model, optim)
    extra_modules['trace_fim'] = TraceFIM(held_out, model, num_classes=NUM_CLASSES)
    
    
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
    # config.stiff_multi = (T_max // epochs) // 2
    config.fim_trace_multi = (T_max // epochs) // 2
    config.run_stats_multi = (T_max // epochs) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = 'reports'
    config.exp_name = EXP_NAME
    config.extra = extra
    config.logger_config = logger_config
    config.checkpoint_path = checkpoint_path
    
    
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
    phase3 = int(sys.argv[3])
    print(lr, wd)
    EPOCHS = 200
    objective('just_run', EPOCHS, lr, wd, phase3)
