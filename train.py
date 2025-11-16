#!/usr/bin/env python
import os
import random
import logging
import shutil
import cv2 as cv
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from matplotlib import pyplot as plt
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import FewShotSeg
from dataloaders.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=my_weight)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'thresh_loss': 0, 'masked_loss': 0}

    i_iter = 0

    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):
            if sample['support_images'].shape[-1] == 257:  # SABS
                for key in ['support_images', 'support_fg_labels', 'query_images', 'query_labels']:
                    tensor = sample[key]
                    ndim = tensor.ndim
                    slices = (slice(None),) * (ndim - 2) + (slice(0, 256), slice(0, 256))
                    sample[key] = tensor[slices]
            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)
            ###################### 我们这里将query_images进行复制并将其中的掩码区域置空，为了后面进行MAE（即送入网络得到预测）
            a = query_images[0][:, 0:1, :, :].clone()
            min_value = a.min().item()
            mask = (query_labels != 0).unsqueeze(0)
            a[mask] = min_value
            repeated_a = a.repeat(1, 3, 1, 1)
            query_images.append(repeated_a)
            remove_mask = (query_labels == 1).float().unsqueeze(1)
            ##########
            # Compute outputs and losses.
            query_pred, qry_fts_out, align_loss = model(support_images, support_fg_mask, query_images, train=True)

            query_loss = criterion(query_pred[0].unsqueeze(0), query_labels)
            query_loss1 = criterion(query_pred[1].unsqueeze(0), query_labels)
            query_loss2 = criterion(query_pred[2].unsqueeze(0), query_labels)
            query_loss3 = criterion(query_pred[3].unsqueeze(0), query_labels)

            loss1 = (a - qry_fts_out) ** 2
            loss1 = loss1.mean(dim=1)  # [N, H, W], mean loss per pixel
            masked_loss1 = loss1 * remove_mask.squeeze(1)  # mask shape: [N, H, W]
            masked_loss = masked_loss1.sum() / (remove_mask.sum() + 1e-6)

            loss = query_loss + 0.3 * query_loss1 + 0.2 * query_loss2 + 0.1 * query_loss3 + align_loss + 0.1 * masked_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            masked_loss = masked_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)
            _run.log_scalar('masked_loss', masked_loss)
            _run.log_scalar('align_loss', align_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['masked_loss'] += masked_loss
            log_loss['align_loss'] += align_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                masked_loss = log_loss['masked_loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['masked_loss'] = 0
                log_loss['align_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, masked_loss: {masked_loss}, align_loss: {align_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            i_iter += 1
    _log.info('End of training.')
    return 1