# author: Zhiyuan Yan (modified for NeSyDeFake frame-level, Step 5)
#
# Step 5 changes:
#   - Removed temporal branch from optimizer param groups
#   - CLI --active_branches only supports spatial/frequency
#   - Logging updated to reflect frame-level mode

import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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
    help='Override active_branches. E.g. --active_branches spatial frequency'
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
                'Only facexray, fwa, sbi, lsda supported for blend dataset')
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

    if config['ddp']:
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
    Adam optimizer with per-module LR groups.
    Frame-level: spatial + frequency only (no temporal).
    """
    import torch.optim as optim
    import logging
    log = logging.getLogger(__name__)

    opt_cfg  = config['optimizer']
    adam_cfg = opt_cfg['adam']
    lr_cfg   = opt_cfg.get('module_lr', {})
    base_lr  = adam_cfg['lr']

    m = model.module if hasattr(model, 'module') else model
    seen_ids = set()

    def _add(params, group_list):
        for p in params:
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                group_list.append(p)

    # Group 1: Backbone LayerNorm params
    g1 = []
    for attr in ('spatial_extractor', 'frequency_extractor'):
        ext = getattr(m, attr, None)
        if ext is not None and hasattr(ext, 'get_trainable_params'):
            _add(ext.get_trainable_params(), g1)

    # Group 2: always_trainable (freq_norm etc, excluding phase_proj)
    g2 = []
    for attr in ('spatial_extractor', 'frequency_extractor'):
        ext = getattr(m, attr, None)
        if ext is None or not hasattr(ext, 'get_always_trainable_params'):
            continue
        for p in ext.get_always_trainable_params():
            freq_ext = getattr(m, 'frequency_extractor', None)
            if (freq_ext is not None
                    and hasattr(freq_ext, 'phase_proj')
                    and freq_ext.phase_proj is not None
                    and any(id(p) == id(pp)
                            for pp in freq_ext.phase_proj.parameters())):
                continue
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                g2.append(p)

    # Group 3: phase_proj
    g3 = []
    freq_ext = getattr(m, 'frequency_extractor', None)
    if (freq_ext is not None
            and hasattr(freq_ext, 'phase_proj')
            and freq_ext.phase_proj is not None):
        _add(list(freq_ext.phase_proj.parameters()), g3)

    # Group 4: Projection heads
    g4 = []
    for attr in ('spatial_proj', 'frequency_proj'):
        mod = getattr(m, attr, None)
        if mod is not None:
            _add(list(mod.parameters()), g4)

    # Group 5: Fusion
    g5 = []
    if hasattr(m, 'fusion'):
        _add(list(m.fusion.parameters()), g5)

    # Group 6: Classifier
    g6 = []
    if hasattr(m, 'multitaskhead'):
        _add(list(m.multitaskhead.parameters()), g6)

    # Group 7: Optional modules (always built at init for phase-transition activation)
    g7 = []
    for attr in ('causal_module', 'sparse_ae', 'violation_proj'):
        mod = getattr(m, attr, None)
        if mod is not None:
            _add(list(mod.parameters()), g7)

    lr_backbone_ln = lr_cfg.get('backbone_layernorms', 1e-5)
    lr_always      = lr_cfg.get('always_trainable',    base_lr)
    lr_phase_proj  = lr_cfg.get('phase_proj',          base_lr * 3)
    lr_proj_heads  = lr_cfg.get('projection_heads',    base_lr)
    lr_fusion      = lr_cfg.get('fusion',              base_lr * 2)
    lr_classifier  = lr_cfg.get('classifier',          base_lr * 2)

    group_specs = [
        (g1, lr_backbone_ln, 'backbone_layernorms'),
        (g2, lr_always,      'always_trainable'),
        (g3, lr_phase_proj,  'phase_proj'),
        (g4, lr_proj_heads,  'projection_heads'),
        (g5, lr_fusion,      'fusion'),
        (g6, lr_classifier,  'classifier'),
        (g7, base_lr,        'optional_modules'),
    ]

    param_groups = []
    for params, lr, name in group_specs:
        if not params:
            log.info(f"  Optimizer group '{name}': empty -- skipped")
            continue
        count = sum(p.numel() for p in params)
        log.info(f"  Optimizer group '{name}': {count:,} params @ lr={lr:.2e}")
        param_groups.append({'params': params, 'lr': lr, 'name': name})

    if not param_groups:
        raise RuntimeError("choose_optimizer(): no trainable parameters found.")

    optimizer = optim.Adam(
        param_groups, lr=base_lr,
        weight_decay=adam_cfg['weight_decay'],
        betas=(adam_cfg['beta1'], adam_cfg['beta2']),
        eps=adam_cfg['eps'], amsgrad=adam_cfg['amsgrad'],
    )
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
    elif config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])
    elif config['lr_scheduler'] == 'cosine_warmup':
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR)
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = config['nEpochs']
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs,
            eta_min=config.get('lr_eta_min', 0))
        return SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
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

    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)
    config['local_rank'] = local_rank
    config['ddp'] = args.ddp

    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)

    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False

    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    if args.active_branches:
        valid = [b for b in args.active_branches if b in ('spatial', 'frequency')]
        if len(valid) != len(args.active_branches):
            print(f"WARNING: temporal branch not supported. Using: {valid}")
        config['active_branches'] = valid
        proj_dim = config['fusion']['projection_dim']
        config['fusion']['fused_dim'] = proj_dim * len(valid)

    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    timenow  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = (f"_{config['task_target']}"
                if config.get('task_target', None) else "")
    branch_str = '_'.join(
        b[0].upper() for b in sorted(config.get('active_branches',
                                                 ['spatial', 'frequency'])))
    task_str = f"{task_str}_branches_{branch_str}"

    logger_path = os.path.join(
        config['log_dir'],
        config['model_name'] + task_str + '_' + timenow)
    if not config['ddp'] or dist.get_rank() == 0:
        os.makedirs(logger_path, exist_ok=True)
    if config['ddp']:
        dist.barrier()

    logger = create_logger(os.path.join(logger_path, 'training.log'))
    if config['ddp']:
        logger.addFilter(RankFilter(0))
    logger.info('Save log to {}'.format(logger_path))

    logger.info("--------------- Configuration ---------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    logger.info("--------------- Runtime Summary ---------------")
    logger.info(f"  MODE             : FRAME-LEVEL (no temporal)")
    logger.info(f"  active_branches  : {config.get('active_branches')}")
    logger.info(f"  video_mode       : {config.get('video_mode', False)}")
    logger.info(f"  frame_num        : {config['frame_num']}")
    logger.info(f"  projection_dim   : {config['fusion']['projection_dim']}")
    logger.info(f"  fused_dim        : {config['fusion']['fused_dim']}")
    logger.info(f"  balance_classes  : {config.get('balance_classes', False)}")
    logger.info(f"  train_batchSize  : {config['train_batchSize']}")
    logger.info(f"  mixed_precision  : {config.get('mixed_precision', False)}")
    logger.info(f"  semantic_features: {config.get('load_semantic_features', False)}")

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    if config['ddp']:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            static_graph=True)

    optimizer      = choose_optimizer(model, config)
    scheduler      = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    trainer = Trainer(config, model, optimizer, scheduler, logger,
                      metric_scoring, time_now=timenow)

    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        if config['ddp'] and hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)

        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders)
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing "
                f"{metric_scoring}: {parse_metric_for_print(best_metric)}!")

    logger.info("Stop Training on best Testing metric {}".format(
        parse_metric_for_print(best_metric)))

    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()