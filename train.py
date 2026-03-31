#!/usr/bin/env python
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import random
import logging
import shutil
import cv2 as cv
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import FewShotSeg
from dataloaders.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex


@ex.automain
def main(_run, _config, _log):
    """Train the model with segmentation, reconstruction, and alignment objectives."""
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', 'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('Create model...')
    model = FewShotSeg().cuda()
    model.train()

    _log.info('Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [
        (ii + 1) * _config['max_iters_per_load']
        for ii in range(_config['n_steps'] // _config['max_iters_per_load'] - 1)
    ]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    _log.info('Load data...')
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']
    log_loss = {'total_loss': 0, 'seg_loss': 0, 'align_loss': 0, 'masked_loss': 0}
    i_iter = 0

    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):
            if sample['support_images'].shape[-1] == 257:
                for key in ['support_images', 'support_fg_labels', 'query_images', 'query_labels']:
                    tensor = sample[key]
                    ndim = tensor.ndim
                    slices = (slice(None),) * (ndim - 2) + (slice(0, 256), slice(0, 256))
                    sample[key] = tensor[slices]

            support_images = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_fg_labels']]
            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            masked_query = query_images[0][:, 0:1, :, :].clone()
            min_value = masked_query.min().item()
            mask = (query_labels != 0).unsqueeze(1)
            masked_query[mask] = min_value
            repeated_masked_query = masked_query.repeat(1, 3, 1, 1)
            query_images.append(repeated_masked_query)
            remove_mask = (query_labels == 1).float().unsqueeze(1)

            query_pred, qry_fts_out, align_loss = model(support_images, support_fg_mask, query_images, train=True)

            query_target = query_labels.float().unsqueeze(1)
            stage_probs = [query_pred[i:i + 1, 1:2] for i in range(query_pred.shape[0])]
            seg_loss = sum(F.binary_cross_entropy(stage_prob, query_target) for stage_prob in stage_probs) / len(stage_probs)

            rec_loss = (masked_query - qry_fts_out).pow(2)
            rec_loss = rec_loss.mean(dim=1, keepdim=True)
            masked_loss = (rec_loss * remove_mask).sum() / (remove_mask.sum() + 1e-6)

            loss = seg_loss + masked_loss + align_loss

            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            scheduler.step()

            seg_loss_value = seg_loss.detach().cpu().numpy()
            masked_loss_value = masked_loss.detach().cpu().numpy()
            align_loss_value = align_loss.detach().cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('seg_loss', seg_loss_value)
            _run.log_scalar('masked_loss', masked_loss_value)
            _run.log_scalar('align_loss', align_loss_value)

            log_loss['total_loss'] += loss.item()
            log_loss['seg_loss'] += seg_loss_value
            log_loss['masked_loss'] += masked_loss_value
            log_loss['align_loss'] += align_loss_value

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                seg_loss_avg = log_loss['seg_loss'] / _config['print_interval']
                masked_loss_avg = log_loss['masked_loss'] / _config['print_interval']
                align_loss_avg = log_loss['align_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['seg_loss'] = 0
                log_loss['masked_loss'] = 0
                log_loss['align_loss'] = 0

                _log.info(
                    f'step {i_iter + 1}: total_loss: {total_loss}, '
                    f'seg_loss: {seg_loss_avg}, masked_loss: {masked_loss_avg}, align_loss: {align_loss_avg}'
                )

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(
                    model.state_dict(),
                    os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth')
                )

            i_iter += 1

    _log.info('End of training.')
    return 1
