# author: Zhiyuan Yan (modified for NeSyDeFake frame-level, Step 5)
#
# Step 5 changes:
#   - Removed temporal branch from optimizer param groups
#   - CLI --active_branches only supports spatial/frequency
#   - Logging updated to reflect frame-level mode

import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import shutil
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
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Override manualSeed. E.g. --seed 123'
)
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='Resume from a last_checkpoint.pth. Accepts either the checkpoint '
         'path or the log folder containing it. Training continues in the '
         'same folder (same logs, same best_*.pth).'
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

    # Group 1: Backbone LayerNorm params (spatial CLIP LNs)
    g1 = []
    ext = getattr(m, 'spatial_extractor', None)
    if ext is not None and hasattr(ext, 'get_trainable_params'):
        _add(ext.get_trainable_params(), g1)

    # Group 2: always_trainable
    g2 = []
    if ext is not None and hasattr(ext, 'get_always_trainable_params'):
        for p in ext.get_always_trainable_params():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                g2.append(p)

    # Group 3: Projection head (spatial)
    g3 = []
    if getattr(m, 'spatial_proj', None) is not None:
        _add(list(m.spatial_proj.parameters()), g3)

    # Group 4: Classifier
    g4 = []
    if hasattr(m, 'multitaskhead'):
        _add(list(m.multitaskhead.parameters()), g4)

    # Group 5: Concept branch (Ablation 3+)
    g5_concept = []
    if hasattr(m, 'concept_branch'):
        _add(list(m.concept_branch.parameters()), g5_concept)

    # Group 6: Causal branch (Ablation 4) + evidence fusion (gates, CMEF)
    g6_causal = []
    if hasattr(m, 'causal_branch'):
        _add(list(m.causal_branch.parameters()), g6_causal)
    if hasattr(m, 'evidence_fusion'):
        _add(list(m.evidence_fusion.parameters()), g6_causal)

    # Group 7: DISCERN v2 branches (manifold / process) + their fusion gates.
    # Without this the v2 stack is built and runs in the forward pass but never receives
    # gradient updates — training proceeds normally while the branch stays frozen at its
    # random initialisation, and the rung silently measures noise. This repo has hit the
    # same class of bug before (causal_attn_fusion + concept_head missing from the
    # optimizer, fixed in v3), which is why it is an explicit group rather than a catch-all.
    g7_v2 = []
    if getattr(m, 'v2', None) is not None:
        _add(list(m.v2.parameters()), g7_v2)

    lr_backbone_ln = lr_cfg.get('backbone_layernorms', 1e-5)
    lr_always      = lr_cfg.get('always_trainable',    base_lr)
    lr_proj_heads  = lr_cfg.get('projection_heads',    base_lr)
    lr_classifier  = lr_cfg.get('classifier',          base_lr * 2)
    lr_concept     = lr_cfg.get('concept_branch',      base_lr)
    lr_causal_b    = lr_cfg.get('causal_branch',       base_lr)
    lr_discern_v2  = lr_cfg.get('discern_v2',          base_lr)

    group_specs = [
        (g1, lr_backbone_ln, 'backbone_layernorms'),
        (g2, lr_always,      'always_trainable'),
        (g3, lr_proj_heads,  'projection_heads'),
        (g4, lr_classifier,  'classifier'),
        (g5_concept, lr_concept,   'concept_branch'),
        (g6_causal,  lr_causal_b,  'causal_branch'),
        (g7_v2,      lr_discern_v2, 'discern_v2'),
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

    # Coverage guard: every parameter with requires_grad must land in some group.
    # The groups above are hand-enumerated per module, so adding a new trainable module
    # without adding its group leaves it frozen at initialisation while training proceeds
    # and logs normally — the run looks healthy and the module contributes noise. That has
    # bitten this repo twice (causal_attn_fusion + concept_head in v3; the DISCERN v2 stack
    # in D1-VM). Failing loudly here is cheap; discovering it after a multi-hour run is not.
    covered = {id(p) for group in param_groups for p in group['params']}
    orphans = [(n, p.numel()) for n, p in m.named_parameters()
               if p.requires_grad and id(p) not in covered]
    if orphans:
        total = sum(n for _, n in orphans)
        preview = ", ".join(f"{n} ({c:,})" for n, c in orphans[:8])
        raise RuntimeError(
            f"choose_optimizer(): {len(orphans)} trainable parameters ({total:,} values) "
            f"are in NO optimizer group and would train as frozen noise: {preview}"
            f"{' ...' if len(orphans) > 8 else ''}. Add them to a group in group_specs.")

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

    if args.seed is not None:
        config['manualSeed'] = args.seed

    if args.active_branches:
        valid = [b for b in args.active_branches if b in ('spatial', 'frequency')]
        if len(valid) != len(args.active_branches):
            print(f"WARNING: temporal branch not supported. Using: {valid}")
        config['active_branches'] = valid
        proj_dim = config['fusion']['projection_dim']
        config['fusion']['fused_dim'] = proj_dim * len(valid)

    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    # ---- Resolve --resume: path can be a .pth file or a log folder ----
    resume_ckpt_path = None
    if args.resume:
        r = os.path.abspath(args.resume)
        if os.path.isdir(r):
            resume_ckpt_path = os.path.join(r, 'last_checkpoint.pth')
        else:
            resume_ckpt_path = r
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(
                f"--resume: checkpoint not found at {resume_ckpt_path}")

    # Derive experiment name from detector config filename (e.g. nesy_defake_ablation1)
    config_name = os.path.splitext(os.path.basename(args.detector_path))[0]

    if resume_ckpt_path is not None:
        # Reuse the existing log folder so logs, TB events, and best_*.pth continue in place
        logger_path = os.path.dirname(resume_ckpt_path)
        timenow = os.path.basename(logger_path).replace(f'{config_name}_', '') \
            or datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    else:
        timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger_path = os.path.join(
            config['log_dir'], 'train',
            f'{config_name}_{timenow}')

    if not config['ddp'] or dist.get_rank() == 0:
        os.makedirs(logger_path, exist_ok=True)
        # Save detector config for reproducibility (skip on resume — it's already there)
        if resume_ckpt_path is None:
            try:
                shutil.copy2(args.detector_path,
                             os.path.join(logger_path, 'detector_config.yaml'))
            except Exception as e:
                print(f"[warn] failed to save detector config: {e}")
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

    # ── Symbolic-stream frozen-set assertion ─────────────────────────────
    # If the concept branch is configured to use the gap-selected v8
    # retained predicate set AND the concept branch is actually built
    # for this ablation, verify at startup that the loaded module
    # exactly matches configs/retained_predicates.yaml. This is required
    # by the predicate-selection protocol — see
    # results/predicate_selection_decision.md for context.
    #
    # Ablations like `visual_edl_only` and `no_causal` legitimately leave
    # the v8_retained marker in the config (so the verifier still ranks
    # them as part of the retained-set family) but skip building the
    # concept branch via `ablation_mode`. In those cases there is nothing
    # to assert against — silently skip.
    cb_cfg = config.get('concept_branch', {}) or {}
    if cb_cfg.get('consistency_rules_version') == 'v8_retained':
        cb_module = getattr(model, 'concept_branch', None)
        if cb_module is None:
            logger.info(
                'Symbolic stream: concept_branch not built under '
                f'ablation_mode={config.get("ablation_mode")!r}; '
                'skipping retained-set verification.')
        elif not hasattr(cb_module, 'consistency_rules'):
            raise RuntimeError(
                'concept_branch.consistency_rules_version=v8_retained '
                'requires a consistency_rules module on the built concept '
                'branch — none was found.')
        else:
            rules = cb_module.consistency_rules
            if not hasattr(rules, 'verify_against_yaml'):
                raise RuntimeError(
                    'concept_branch.consistency_rules_version=v8_retained '
                    'expected RetainedConsistencyRules but found '
                    f'{type(rules).__name__}.')
            rules.verify_against_yaml()
            logger.info(
                f'Symbolic stream: verified {rules.k} retained predicates '
                f'against {rules.yaml_path}')

    if config['ddp']:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            static_graph=True)

    optimizer      = choose_optimizer(model, config)
    scheduler      = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    trainer = Trainer(config, model, optimizer, scheduler, logger,
                      metric_scoring, log_dir=logger_path, time_now=timenow)

    best_metric = None
    # Early stopping state
    es_cfg = config.get('early_stopping', {})
    es_enabled = es_cfg.get('enabled', False)
    es_patience = es_cfg.get('patience', 15)
    es_min_delta = es_cfg.get('min_delta', 0.001)
    es_best_score = float('-inf')
    es_wait = 0

    # ---- Resume from checkpoint (after trainer + optimizer are fully built) ----
    if resume_ckpt_path is not None:
        completed_epoch, extra = trainer.load_resume_state(resume_ckpt_path)
        config['start_epoch'] = completed_epoch + 1
        es_best_score = extra.get('es_best_score', float('-inf'))
        es_wait = extra.get('es_wait', 0)
        logger.info(
            f"Resuming training at epoch {config['start_epoch']} "
            f"(es_best_score={es_best_score:.6f}, es_wait={es_wait})")

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

            # Periodic checkpoint saving removed — only best checkpoint is saved

            # Early stopping check on avg AUC
            if es_enabled and 'avg' in best_metric:
                current_score = best_metric['avg'].get(metric_scoring, 0)
                if current_score > es_best_score + es_min_delta:
                    es_best_score = current_score
                    es_wait = 0
                else:
                    es_wait += 1
                    logger.info(
                        f"  Early stopping: no improvement for {es_wait}/{es_patience} epochs "
                        f"(best={es_best_score:.6f}, current={current_score:.6f})")
                    if es_wait >= es_patience:
                        logger.info(f"  Early stopping triggered at epoch {epoch}")
                        trainer.save_resume_state(
                            epoch,
                            {'es_best_score': es_best_score, 'es_wait': es_wait})
                        break

        # Save resume checkpoint at end of every epoch (atomic — safe if killed)
        trainer.save_resume_state(
            epoch,
            {'es_best_score': es_best_score, 'es_wait': es_wait})

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