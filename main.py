import os
import time
import argparse
import copy
import time
import numpy as np
import random
import mmcv
import torch
from torch.utils.data import DataLoader
from functools import partial
from mmcv import Config, DictAction
from mmcv.parallel import collate
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, build_runner, build_optimizer
from mmcls import __version__
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger
from mmcls.apis import multi_gpu_test, single_gpu_test

from samplers import RandomIdentitySampler, RandomClassIdentitySampler

from mmcls.datasets import (DATASETS, BaseDataset, ClassBalancedDataset,
                            ConcatDataset, MultiLabelDataset, RepeatDataset)

from recipes.file import *

from datetime import datetime

from custom_dataset import CustomDataset

from losses import TripletLoss, QuadrupletLoss
from heads import IdentHead
from necks import DimNeck
#from transforms import ColorJitter
from metrics import compute_distance_matrix, eval_market1501

custom_imports = dict(
    imports=['losses'])


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def my_build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1, shuffle=True, seed=None, sampler_type='identities', **kwargs):
    rank, world_size = get_dist_info()

    # Omitted code for distributed training
    if sampler_type == 'identities':
        sampler = RandomIdentitySampler(dataset, samples_per_gpu, 4) if shuffle else None
    elif sampler_type == 'identities_classes':
        sampler = RandomClassIdentitySampler(dataset, batch_size=samples_per_gpu, objects_per_class=4, num_instances=4) if shuffle else None
    else:
        raise NotImplementedError(sampler_type)

    batch_size = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if 'dist' in kwargs:
        del kwargs['dist']
    if 'round_up' in kwargs:
        del kwargs['round_up']

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def get_config(base_config_path: str, base_checkpoint_path: str, output_path='./outputs/output') -> Config:
    assert os.path.exists(base_config_path)
    assert os.path.exists(base_checkpoint_path)

    cfg = Config.fromfile(base_config_path)

    cfg.load_from = base_checkpoint_path

    # Set up working dir to save files and logs.
    cfg.work_dir = output_path

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.log_config = dict(
        interval=10,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.model.backbone.frozen_stages = 3

    #cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}
    #cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'QuadrupletLoss', 'loss_weight': 1.0}}
    #cfg.model.head = {'type': 'MultiHead', 'losses': {'cls': {'type': 'CrossEntropyLoss', 'loss_weight': 0.5}, 'ident': {'type': 'TripletLoss', 'loss_weight': 0.5}}, 'topk': (1, 5)}
    #cfg.model.neck = {'type': 'DimNeck', 'out_channels': 512}

    cfg.evaluation.interval = 1#20
    cfg.checkpoint_config.interval = 1 #20

    cfg.total_epochs = 20
    cfg.runner.max_epochs = cfg.total_epochs

    cfg.data.samples_per_gpu = 32
    cfg.data.samples_per_gpu = 32

    #cfg.optimizer = {'type': 'Adam', 'lr': 0.0003, 'weight_decay': 5e-04, 'betas': (0.9, 0.99)}
    #cfg.optimizer.lr = 0.01
    #cfg.lr_config = {'policy': 'step', 'step': [230, 270]}

    # optimizer
    # cfg.optimizer = dict(
    #     type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
    # cfg.optimizer_config = dict(grad_clip=None)
    # # learning policy
    # cfg.lr_config = dict(
    #     policy='CosineAnnealing',
    #     min_lr=0,
    #     warmup='linear',
    #     warmup_iters=2500,
    #     warmup_ratio=0.25)
    # cfg.runner = dict(type='EpochBasedRunner', max_epochs=300)

    cfg.sampler_type = 'identities'

    return cfg


def my_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        my_build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            seed=cfg.seed,
            sampler_type=cfg.sampler_type) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    assert cfg.get('runner') is not None

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def training_pipeline(type='classes', output_path='./outputs/output', lr=None, model_type=None):
    data_path = f'/media/amidemo/Data/blendered/orig/all_model_images_{type}_cropped/'
    dataset_class = DATASETS.get('CustomDataset')

    base_config_path = '../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
    base_checkpoint_path = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'

    cfg = get_config(base_config_path, base_checkpoint_path, output_path)

    # Add a color transform
    cfg.train_pipeline.insert(1, {'type': 'ColorJitter', 'brightness': 0.2, 'contrast': 0.15, 'saturation': 0})

    if lr:
        cfg.optimizer.lr = lr

    dataset = dataset_class(
        data_prefix=data_path,
        pipeline=cfg.train_pipeline,
        ann_file=os.path.join(data_path, 'annotations.txt'),
        classes=None,
        test_mode=False)

    if model_type == 'tri':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}
    elif model_type == 'quad':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'QuadrupletLoss', 'loss_weight': 1.0, 'super_classes': dataset.super_classes}}
        cfg.sampler_type = 'identities_classes'
    elif model_type == 'cls':
        cfg.model.head = {'type': 'ClsHead', 'loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0}}

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {False}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta['seed'] = cfg.seed

    model = build_classifier(cfg.model)

    # add an attribute for visualization convenience
    my_train_model(
        model,
        dataset,
        cfg,
        distributed=False,
        validate=False,
        timestamp=timestamp,
        meta=meta)


def training_pipeline_cats_dogs(output_path='./outputs/output', lr=None, model_type=None):
    data_path = '/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset'
    dataset_class = DATASETS.get('OxfordCatsDogsBreeds')

    base_config_path = '../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
    base_checkpoint_path = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'

    cfg = get_config(base_config_path, base_checkpoint_path, output_path)

    # Add a color transform
    cfg.train_pipeline.insert(1, {'type': 'ColorJitter', 'brightness': 0.2, 'contrast': 0.15, 'saturation': 0})

    if lr:
        cfg.optimizer.lr = lr

    dataset = dataset_class(
        data_prefix=data_path,
        pipeline=cfg.train_pipeline,
        ann_file=os.path.join(data_path, 'annotations', 'annotations', 'trainval.txt'),
        classes=None,
        test_mode=False)

    if model_type == 'tri':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}
        cfg.sampler_type = 'identities_classes'
    elif model_type == 'quad':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'QuadrupletLoss', 'loss_weight': 1.0,
                                                        'super_classes': dataset.super_classes}}
        cfg.sampler_type = 'identities_classes'
    elif model_type == 'cls':
        cfg.model.head = {'type': 'LinearClsHead', 'num_classes': -1, 'in_channels': 2048, 'loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0}, 'topk': (1, 5)}
        cfg.sampler_type = 'identities_classes'

    # After the creation of the dataset we know the number of classes and we can override the config for the
    # classification head
    if model_type == 'cls':
        cfg.model.head['num_classes'] = len(dataset_class.CLASSES) + 1  # real + 1

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {False}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta['seed'] = cfg.seed

    model = build_classifier(cfg.model)

    # add an attribute for visualization convenience
    my_train_model(
        model,
        dataset,
        cfg,
        distributed=False,
        validate=False,
        timestamp=timestamp,
        meta=meta)

def evaluation_pipeline(type='classes', eval_type='reid', model_type=None, output_path='./outputs/output',
                        epoch=None):
    def init_data_loader(data_path):
        # Test overriding not a subset
        if eval_type == 'reid':
            cfg.test_pipeline[5] = {'type': 'Collect', 'keys': ['img', 'gt_label']}

        dataset = dataset_class(
            data_prefix=data_path,
            pipeline=cfg.test_pipeline,
            ann_file=os.path.join(data_path, 'annotations.txt'),
            classes=None,
            test_mode=False)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False)
        return dataset, data_loader

    def get_results(data_loader, dataset, model):
        model.eval()
        results = []
        labels = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        for i, data in enumerate(data_loader):
            labels.extend(data.pop('gt_label'))
            with torch.no_grad():
                result = model(return_loss=False, **data)
                if isinstance(result, tuple) or isinstance(result, list):
                    result = result[1]

            batch_size = len(result)
            results.extend(result)

            batch_size = data['img'].size(0)
            for _ in range(batch_size):
                prog_bar.update()

        results = torch.vstack(results)
        labels = torch.vstack(labels)
        return results, labels

    train_data_path = f'/media/amidemo/Data/blendered/orig/all_model_images_{type}_cropped/'
    test_data_path = f'/media/amidemo/Data/blendered/strength0.1/all_test_images_{type}_cropped/'
    dataset_class = DATASETS.get('CustomDataset')

    base_config_path = '../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
    # base_checkpoint_path = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'

    if epoch:
        base_checkpoint_path = f'{output_path}/epoch_{epoch}.pth'
    else:
        base_checkpoint_path = f'{output_path}/latest.pth'

    checkpoint_path = base_checkpoint_path

    cfg = get_config(base_config_path, base_checkpoint_path, output_path)

    if model_type == 'tri':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}
    elif model_type == 'quad':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'QuadrupletLoss', 'loss_weight': 1.0}}
        cfg.sampler_type = 'identities_classes'
    elif model_type == 'cls':
        #cfg.model.head = {'type': 'ClsHead', 'loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0}}
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}

    train_dataset, train_data_loader = init_data_loader(train_data_path)
    test_dataset, test_data_loader = init_data_loader(test_data_path)


    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    if eval_type == 'reid':
        train_results, train_labels = get_results(train_data_loader, train_dataset, model)
        test_results, test_labels = get_results(test_data_loader, test_dataset, model)

        dists = compute_distance_matrix(test_results, train_results)
        all_cmc, mAP = eval_market1501(dists.cpu().numpy(), test_labels.cpu().numpy().ravel(), train_labels.cpu().numpy().ravel(), None, None, max_rank=50)
        print(all_cmc, mAP)
    elif eval_type == 'cls':
        outputs = single_gpu_test(model, test_data_loader, show=True)
        results = test_dataset.evaluate(outputs)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
    else:
        raise NotImplementedError('Operation not supported')


def evaluation_pipeline_cats_dogs(eval_type='reid', model_type=None, output_path='./outputs/output', epoch=None):
    def init_data_loader(data_path, annotation_relative):
        # Test overriding not a subset
        if eval_type == 'reid':
            cfg.test_pipeline[5] = {'type': 'Collect', 'keys': ['img', 'gt_label']}

        dataset = dataset_class(
            data_prefix=data_path,
            pipeline=cfg.test_pipeline,
            ann_file=os.path.join(data_path, annotation_relative),
            classes=None,
            test_mode=False)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False)
        return dataset, data_loader

    def get_results(data_loader, dataset, model):
        model.eval()
        results = []
        labels = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        for i, data in enumerate(data_loader):
            labels.extend(data.pop('gt_label'))
            with torch.no_grad():
                result = model(return_loss=False, **data)
                if isinstance(result, tuple) or isinstance(result, list):
                    result = result[1]

            batch_size = len(result)
            results.extend(result)

            batch_size = data['img'].size(0)
            for _ in range(batch_size):
                prog_bar.update()

        results = torch.vstack(results)
        labels = torch.vstack(labels)
        return results, labels

    data_path = '/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset'
    dataset_class = DATASETS.get('OxfordCatsDogsBreeds')

    base_config_path = '../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
    # base_checkpoint_path = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'

    if epoch:
        base_checkpoint_path = f'{output_path}/epoch_{epoch}.pth'
    else:
        base_checkpoint_path = f'{output_path}/latest.pth'

    checkpoint_path = base_checkpoint_path

    cfg = get_config(base_config_path, base_checkpoint_path, output_path)

    if model_type == 'tri':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}
    elif model_type == 'quad':
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'QuadrupletLoss', 'loss_weight': 1.0}}
        cfg.sampler_type = 'identities_classes'
    elif model_type == 'cls':
        #cfg.model.head = {'type': 'ClsHead', 'loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0}}
        cfg.model.head = {'type': 'IdentHead', 'loss': {'type': 'TripletLoss', 'loss_weight': 1.0}}

    train_dataset, train_data_loader = init_data_loader(data_path, './annotations/annotations/trainval.txt')
    test_dataset, test_data_loader = init_data_loader(data_path, './annotations/annotations/test.txt')


    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    if eval_type == 'reid':
        train_results, train_labels = get_results(train_data_loader, train_dataset, model)
        test_results, test_labels = get_results(test_data_loader, test_dataset, model)

        dists = compute_distance_matrix(test_results, train_results)
        all_cmc, mAP = eval_market1501(dists.cpu().numpy(), test_labels.cpu().numpy().ravel(), train_labels.cpu().numpy().ravel(), None, None, max_rank=50)
        print(all_cmc, mAP)
    elif eval_type == 'cls':
        outputs = single_gpu_test(model, test_data_loader, show=True)
        results = test_dataset.evaluate(outputs)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
    else:
        raise NotImplementedError('Operation not supported')


def test_sampler(output_tag=''):
    # Model config & checkpoint
    base_config_path = '../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
    base_checkpoint_path = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'

    # Dataset
    dataset_id = 'oxford-cats-dogs-breeds'

    # Config creation
    cfg = get_config(base_config_path, base_checkpoint_path, output_tag)
    cfg.sampler_type = 'identities_classes'

    data_path = '/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset'
    dataset_class = DATASETS.get('OxfordCatsDogsBreeds')
    dataset = dataset_class(
        data_prefix=data_path,
        pipeline=cfg.train_pipeline,
        ann_file=os.path.join(data_path, 'annotations', 'annotations', 'trainval.txt'),
        classes=None,
        test_mode=False)

    # Build dataset
    #dataset = build_dataset(cfg.data.train)

    dl = my_build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        sampler_type=cfg.sampler_type,
        seed=cfg.seed)

    for i, batch in enumerate(dl):
        print(i, batch['gt_label'])


def test():
    # targets = torch.Tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]).int()
    # id2c = torch.Tensor([0, 1, 0, 1, 2, 2]).int()
    # batch_classes = torch.zeros_like(targets)
    #
    # for i, t in enumerate(targets):
    #     batch_classes[i] = id2c[t-1]
    #
    # print(batch_classes)

    targets = torch.Tensor([9,  9,  9,  9,  6,  6,  6,  6, 11, 11, 11, 11, 12, 12, 12, 12,  2,  2,
         2,  2, 13, 13, 13, 13, 10, 10, 10, 10,  3,  3,  3,  3])
    batch_classes = torch.Tensor([2, 2, 2, 2, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 1, 6, 6, 6, 6,
        3, 3, 3, 3, 2, 2, 2, 2])

    n = 32
    mask_ids = targets.expand(n, n).eq(targets.expand(n, n).t())
    mask_classes = batch_classes.expand(n, n).eq(batch_classes.expand(n, n).t())

    mask_neg_ids = torch.logical_xor(mask_ids, mask_classes)
    mask_neg_classes = torch.logical_not(mask_classes)


    dist_ap, dist_ani, dist_anc = [], [], []
    for i in range(n):
        # res1 = dist[i][mask_ids[i]]
        # dist_ap.append(res1.max().unsqueeze(0))
        # res2 = dist[i][mask_neg_ids[i]]
        # dist_ani.append(res2.min().unsqueeze(0))
        # res3 = dist[i][mask_neg_classes[i]]
        # dist_anc.append(res3.min().unsqueeze(0))
        print(i, mask_neg_ids[i].int())


def main():
    #lrs = [0.0003, 0.0003*0.5, 0.0003*0.1, 0.0003*0.01, 0.0003*0.001]
    # lrs = [0.1, 0.01, 0.001]
    # for lr in lrs:
    #     training_pipeline('ids', output_path=f'./outputs/output_tri_{datetime.now():%Y-%m-%d_%H-%M-%S}', lr=lr, model_type='tri')
    # for lr in lrs:
    #     training_pipeline('ids', output_path=f'./outputs/output_quad_{datetime.now():%Y-%m-%d_%H-%M-%S}', lr=lr, model_type='quad')
    # for lr in lrs:
    #     training_pipeline('ids', output_path=f'./outputs/output_cls_{datetime.now():%Y-%m-%d_%H-%M-%S}', lr=lr, model_type='cls')

    # training_pipeline('ids', output_path=f'./outputs/output_tri_{datetime.now():%Y-%m-%d_%H-%M-%S}', lr=0.1,
    #                   model_type='tri')

    # cats n dogs
    model_type = 'tri'
    # output_path = f'./outputs/output_{model_type}_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    # training_pipeline_cats_dogs(output_path=output_path, lr=0.01, model_type=model_type)

    #
    output_path = './outputs/output_base'
    evaluation_pipeline_cats_dogs(eval_type='reid', output_path=output_path, model_type=model_type, epoch=None)

    #symlink_force(output_path, f'./outputs/output_{model_type}_latest')

    # types = ['tri', 'quad', 'cls']
    # # #types = ['cls']
    # #
    # for t in types:
    #     path = f'./outputs/exps_sgd/{t}'
    #     for p in os.listdir(path):
    #         output_path = os.path.join(path, p)
    #         evaluation_pipeline('ids', eval_type='reid', output_path=output_path, model_type=t, epoch=20)

    # for e in range(1, 21):
    #     evaluation_pipeline('ids', eval_type='reid', output_path='./outputs/output_tri_2021-03-17_16-01-47', model_type='tri', epoch=e)

    # evaluation_pipeline('ids', eval_type='reid', output_tag=f'_quad_first')

    #test_sampler()
    # test()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

