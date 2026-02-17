# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

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
                # DistributedSampler so every rank gets a unique shard
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
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'],
                   config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError(
            'Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


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
    #  DDP initialisation FIRST, before any logging or data loading       #
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
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    # ---- Logger: create on ALL ranks but filter output to rank 0 ----
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = (f"_{config['task_target']}"
                if config.get('task_target', None) else "")
    logger_path = os.path.join(
        config['log_dir'],
        config['model_name'] + task_str + '_' + timenow,
    )
    # Only rank 0 creates the directory and log file to avoid race conditions
    if not config['ddp'] or dist.get_rank() == 0:
        os.makedirs(logger_path, exist_ok=True)
    if config['ddp']:
        dist.barrier()   # ensure dir exists before non-0 ranks proceed

    logger = create_logger(os.path.join(logger_path, 'training.log'))
    if config['ddp']:
        logger.addFilter(RankFilter(0))
    logger.info('Save log to {}'.format(logger_path))

    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += f"{key}: {value}\n"
    logger.info(params_string)

    # ---- reproducibility ----
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    # ---- data ----
    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    # ---- model ----
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    if config['ddp']:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # Only set True if your model genuinely has unused params;
            # it adds ~10% overhead per step otherwise.
            # find_unused_parameters=False,
        )

    # ---- optimizer / scheduler / metric ----
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    # ---- trainer ----
    trainer = Trainer(config, model, optimizer, scheduler, logger,
                      metric_scoring, time_now=timenow)

    # ---- training loop ----
    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        # Let the sampler know the epoch for correct shuffling across ranks
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