# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.
#
# Step 4 changes:
#   - CLI active_branches fused_dim patch: was sum(raw_branch_dims), now
#     projection_dim × num_active_branches (aligned with Step 3 detector).
#   - Added balance_classes startup log line.
#   - Added TODO for per-module learning rates (module_lr in YAML is defined
#     but choose_optimizer() uses flat model.parameters() — pre-existing gap).

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from dataset.nesy_defake_dataset import NeSyDeFakeDataset


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt',
                    action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat',
                    action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="",
                    help='specify the target of current training task')
parser.add_argument(
    '--active_branches',
    nargs='+',
    default=None,
    help='Override active_branches. E.g. --active_branches temporal spatial'
)

args = parser.parse_args()


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    # NeSyDeFake path — uses our custom dataset with full-segment loading,
    # balance_classes sampler, and DDP sampler all handled internally.
    if (config.get('dataset_type') == 'nesydefake'
            or config['model_name'] == 'nesydefake_hybrid'):
        return NeSyDeFakeDataset.prepare_data_loader(config, mode='train')

    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are supported for blending dataset')
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')

    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(
            num_groups=2 * 360,
            n_frame_per_vid=config['frame_num']['train'],
            batch_size=config['train_batchSize'],
            videos_per_group=5,
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            num_workers=int(config['workers']),
            sampler=custom_sampler,
            collate_fn=train_set.collate_fn,
        )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    return train_data_loader


def prepare_testing_data(config):
    if (config.get('dataset_type') == 'nesydefake'
            or config['model_name'] == 'nesydefake_hybrid'):
        test_data_loaders = {}
        for test_name in config['test_dataset']:
            test_config = config.copy()
            test_config['test_dataset'] = test_name
            test_set = NeSyDeFakeDataset(test_config, mode='test')

            if config['ddp']:
                sampler = DistributedSampler(
                    test_set, shuffle=False, drop_last=False)
                test_data_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config['test_batchSize'],
                    sampler=sampler,
                    num_workers=config['workers'],
                    collate_fn=test_set.collate_fn,
                )
            else:
                test_data_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config['test_batchSize'],
                    shuffle=False,
                    num_workers=config['workers'],
                    collate_fn=test_set.collate_fn,
                )
            test_data_loaders[test_name] = test_data_loader
        return test_data_loaders

    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
        else:
            test_set = LRLDataset(config=config, mode='test')

        if config['ddp']:
            sampler = DistributedSampler(
                test_set, shuffle=False, drop_last=False)
            test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                sampler=sampler,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        else:
            test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=(test_name == 'DeepFakeDetection'),
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(
            config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    """
    Build an Adam optimizer with per-module learning rate groups.

    Param group strategy (GenD-style):
      Group 1 — backbone_layernorms : very low LR (1e-5)
                 Only LayerNorm params inside frozen backbones.
                 These adapt normalization statistics without moving
                 the feature manifold. GenD's generalization mechanism.

      Group 2 — projection_heads    : standard LR (1e-4)
                 temporal_proj, spatial_proj, frequency_proj.
                 These are new modules — they need to learn from scratch.

      Group 3 — fusion              : standard LR (2e-4)
                 MultiModalFusion parameters.

      Group 4 — classifier          : standard LR (2e-4)
                 MultiTaskHead parameters.

    All frozen backbone params (non-LN) are excluded from every group —
    they have requires_grad=False so optimizer.step() skips them anyway,
    but explicit exclusion keeps the param group sizes clean for logging.
    """
    import torch.optim as optim

    opt_cfg   = config['optimizer']
    opt_name  = opt_cfg['type']
    adam_cfg  = opt_cfg['adam']
    lr_cfg    = opt_cfg.get('module_lr', {})

    # Pull the actual model out of DDP wrapper if needed
    m = model.module if hasattr(model, 'module') else model

    # ------------------------------------------------------------------
    # Group 1: Backbone LayerNorm params (GenD regime)
    # ------------------------------------------------------------------
    backbone_ln_params = []
    if hasattr(m, 'spatial_extractor'):
        backbone_ln_params += m.spatial_extractor.get_trainable_params()
    if hasattr(m, 'temporal_extractor') and hasattr(m.temporal_extractor, 'get_trainable_params'):
        backbone_ln_params += m.temporal_extractor.get_trainable_params()
    if hasattr(m, 'frequency_extractor') and hasattr(m.frequency_extractor, 'get_trainable_params'):
        backbone_ln_params += m.frequency_extractor.get_trainable_params()

    # Deduplicate (in case any param appears in multiple branches)
    seen_ids = set()
    deduplicated_ln = []
    for p in backbone_ln_params:
        if id(p) not in seen_ids and p.requires_grad:
            seen_ids.add(id(p))
            deduplicated_ln.append(p)

    # ------------------------------------------------------------------
    # Group 2: Projection heads (always trainable new modules)
    # ------------------------------------------------------------------
    proj_params = []
    for attr in ('temporal_proj', 'spatial_proj', 'frequency_proj'):
        if hasattr(m, attr):
            proj_params += [p for p in getattr(m, attr).parameters()
                            if p.requires_grad and id(p) not in seen_ids]
            for p in getattr(m, attr).parameters():
                seen_ids.add(id(p))

    # ------------------------------------------------------------------
    # Group 3: Fusion
    # ------------------------------------------------------------------
    fusion_params = [p for p in m.fusion.parameters()
                     if p.requires_grad and id(p) not in seen_ids]
    for p in m.fusion.parameters():
        seen_ids.add(id(p))

    # ------------------------------------------------------------------
    # Group 4: Classifier (MultiTaskHead)
    # ------------------------------------------------------------------
    cls_params = [p for p in m.multitaskhead.parameters()
                  if p.requires_grad and id(p) not in seen_ids]
    for p in m.multitaskhead.parameters():
        seen_ids.add(id(p))

    # ------------------------------------------------------------------
    # Optional groups for enabled modules
    # ------------------------------------------------------------------
    optional_params = []
    for attr in ('causal_module', 'sparse_ae', 'semantic_grounding'):
        if hasattr(m, attr):
            ps = [p for p in getattr(m, attr).parameters()
                  if p.requires_grad and id(p) not in seen_ids]
            optional_params += ps
            for p in getattr(m, attr).parameters():
                seen_ids.add(id(p))

    # ------------------------------------------------------------------
    # Assemble param groups
    # ------------------------------------------------------------------
    base_lr = adam_cfg['lr']
    param_groups = []

    if deduplicated_ln:
        param_groups.append({
            'params': deduplicated_ln,
            'lr':     lr_cfg.get('backbone_layernorms', 1e-5),
            'name':   'backbone_layernorms',
        })

    if proj_params:
        param_groups.append({
            'params': proj_params,
            'lr':     lr_cfg.get('projection_heads', base_lr),
            'name':   'projection_heads',
        })

    if fusion_params:
        param_groups.append({
            'params': fusion_params,
            'lr':     lr_cfg.get('fusion', base_lr * 2),
            'name':   'fusion',
        })

    if cls_params:
        param_groups.append({
            'params': cls_params,
            'lr':     lr_cfg.get('classifier', base_lr * 2),
            'name':   'classifier',
        })

    if optional_params:
        param_groups.append({
            'params': optional_params,
            'lr':     base_lr,
            'name':   'optional_modules',
        })

    # Log what ended up in each group
    import logging
    log = logging.getLogger(__name__)
    for g in param_groups:
        count = sum(p.numel() for p in g['params'])
        log.info(f"  Optimizer group '{g['name']}': {count:,} params @ lr={g['lr']}")

    if opt_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=base_lr,                       # fallback for groups without explicit lr
            weight_decay=adam_cfg['weight_decay'],
            betas=(adam_cfg['beta1'], adam_cfg['beta2']),
            eps=adam_cfg['eps'],
            amsgrad=adam_cfg['amsgrad'],
        )
    else:
        raise NotImplementedError(
            f"Per-module param groups only implemented for adam. Got: {opt_name}. "
            f"Add SGD/SAM support following the same pattern above."
        )

    return optimizer

# def choose_optimizer(model, config):
#     opt_name = config['optimizer']['type']
#     if opt_name == 'sgd':
#         optimizer = optim.SGD(
#             params=model.parameters(),
#             lr=config['optimizer'][opt_name]['lr'],
#             momentum=config['optimizer'][opt_name]['momentum'],
#             weight_decay=config['optimizer'][opt_name]['weight_decay'],
#         )
#     elif opt_name == 'adam':
#         optimizer = optim.Adam(
#             params=model.parameters(),
#             lr=config['optimizer'][opt_name]['lr'],
#             weight_decay=config['optimizer'][opt_name]['weight_decay'],
#             betas=(config['optimizer'][opt_name]['beta1'],
#                    config['optimizer'][opt_name]['beta2']),
#             eps=config['optimizer'][opt_name]['eps'],
#             amsgrad=config['optimizer'][opt_name]['amsgrad'],
#         )
#     elif opt_name == 'sam':
#         optimizer = SAM(
#             model.parameters(),
#             optim.SGD,
#             lr=config['optimizer'][opt_name]['lr'],
#             momentum=config['optimizer'][opt_name]['momentum'],
#         )
#     else:
#         raise NotImplementedError(
#             'Optimizer {} is not implemented'.format(config['optimizer']))
#     return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
    elif config['lr_scheduler'] == 'cosine_warmup':
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR)
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = config['nEpochs']
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=config.get('lr_eta_min', 0))
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])
    elif config['lr_scheduler'] == 'linear':
        return LinearDecayLR(
            optimizer, config['nEpochs'], int(config['nEpochs'] / 4))
    else:
        raise NotImplementedError(
            'Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError(
            'metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # ------------------------------------------------------------------ #
    #  Config loading                                                      #
    # ------------------------------------------------------------------ #
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)
    config['local_rank'] = local_rank
    config['ddp'] = args.ddp

    # ------------------------------------------------------------------ #
    #  DDP initialisation                                                  #
    # ------------------------------------------------------------------ #
    if config['ddp']:
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=2),
        )
        torch.cuda.set_device(local_rank)

    # ---- dry-run overrides ----
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False

    # ---- CLI overrides ----
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    # ---- active_branches CLI override ----
    # Step 3 fix: fused_dim = projection_dim × num_active_branches.
    # Previously used sum(raw_branch_dims) which was wrong after per-branch
    # projection heads were added. build_backbone() also patches this at model
    # init, but we set it here so config inspection before model construction
    # (e.g. logger printout) shows the correct value.
    if args.active_branches:
        config['active_branches'] = args.active_branches
        proj_dim = config['fusion']['projection_dim']
        config['fusion']['fused_dim'] = proj_dim * len(args.active_branches)

    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    # ------------------------------------------------------------------ #
    #  Logger                                                              #
    # ------------------------------------------------------------------ #
    timenow  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = (f"_{config['task_target']}"
                if config.get('task_target', None) else "")
    branch_str = '_'.join(
        b[0].upper() for b in sorted(config.get('active_branches',
                                                 ['temporal', 'spatial', 'frequency']))
    )
    task_str = f"{task_str}_branches_{branch_str}"

    logger_path = os.path.join(
        config['log_dir'],
        config['model_name'] + task_str + '_' + timenow,
    )
    if not config['ddp'] or dist.get_rank() == 0:
        os.makedirs(logger_path, exist_ok=True)
    if config['ddp']:
        dist.barrier()

    logger = create_logger(os.path.join(logger_path, 'training.log'))
    if config['ddp']:
        logger.addFilter(RankFilter(0))
    logger.info('Save log to {}'.format(logger_path))

    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += f"{key}: {value}\n"
    logger.info(params_string)

    # ---- Key runtime values for quick sanity check in logs ----
    logger.info("--------------- Runtime Summary ---------------")
    logger.info(f"  active_branches  : {config.get('active_branches')}")
    logger.info(f"  frame_num        : {config['frame_num']}")
    logger.info(f"  clip_size        : {config.get('clip_size')} (== frame_num, full segment)")
    logger.info(f"  projection_dim   : {config['fusion']['projection_dim']}")
    logger.info(f"  fused_dim        : {config['fusion']['fused_dim']} (patched at runtime by build_backbone)")
    logger.info(f"  balance_classes  : {config.get('balance_classes', False)}")
    logger.info(f"  use_data_aug     : {config.get('use_data_augmentation', False)}")
    logger.info(f"  train_batchSize  : {config['train_batchSize']}")
    logger.info(f"  mixed_precision  : {config.get('mixed_precision', False)}")

    # ------------------------------------------------------------------ #
    #  Reproducibility                                                     #
    # ------------------------------------------------------------------ #
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    if config['ddp']:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # ------------------------------------------------------------------ #
    #  Optimizer / scheduler / metric                                      #
    # ------------------------------------------------------------------ #
    optimizer     = choose_optimizer(model, config)
    scheduler     = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    # ------------------------------------------------------------------ #
    #  Trainer                                                             #
    # ------------------------------------------------------------------ #
    trainer = Trainer(config, model, optimizer, scheduler, logger,
                      metric_scoring, time_now=timenow)

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        if config['ddp'] and hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)

        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing "
                f"{metric_scoring}: {parse_metric_for_print(best_metric)}!")

    logger.info(
        "Stop Training on best Testing metric {}".format(
            parse_metric_for_print(best_metric)))

    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()