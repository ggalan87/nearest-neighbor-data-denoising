from pathlib import Path
import subprocess
import argparse
import os
import numpy as np
import h5py
import pathlib
import itertools
import multiprocessing as mp
from tqdm import tqdm
import torch
import torchvision
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


try:
    # https://github.com/pytorch/accimage
    torchvision.set_image_backend('accimage')
except:
    pass
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms

# Not used for now
# from vast.tools import logger as vastlogger

try:
    from pl_bolts.models import self_supervised

    pl_bolts = True
except:
    pl_bolts = False

try:
    import timm
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    DeiT_support = True
except:
    DeiT_support = False

# mmclassification & mmcv
from mmcv import Config
from mmcls.datasets import (build_dataset, build_dataloader, DATASETS)
import custom_dataset


# Adapted to python from https://github.com/Vastlab/vast/blob/main/tools/FeatureExtractors/extract_all_pytorch_model.sh


def get_config(base_config_path: str, output_path='./outputs/output') -> Config:
    assert os.path.exists(base_config_path)

    cfg = Config.fromfile(base_config_path)

    # Set up working dir to save files and logs.
    cfg.work_dir = output_path

    return cfg


def init_data_loader(cfg, data_path, annotation_relative, dataset_class):
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
        round_up=False,
        pin_memory=False,
        drop_last=False
    )
    return dataset, data_loader


def extract(architecture, dataset_part, dataset, dataset_root, image_names: list):
    avail_pretrained_models = timm.list_models(pretrained=True)

    if architecture not in avail_pretrained_models:
        print(f'Model {architecture} is not available.')
        return

    # Zero in the number of classes is to disable the classification part of the model, i.e. feature extractor only
    model = timm.create_model(architecture, pretrained=True, num_classes=0)
    model.eval()

    # dataset_tag, dataset_folder = dataset
    # print(f'Starting\t {architecture}\t {dataset_part} \t {dataset_tag}')
    #
    # output_dir = Path(dataset_root) / dataset_folder / 'features' / architecture
    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_file_path = output_dir / f'{dataset_part}.hdf5'
    #
    # cfg = get_config(base_config_path='../../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py')
    # cfg.test_pipeline[5] = {'type': 'Collect', 'keys': ['img', 'gt_label']}
    #
    # data_path = '/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset'
    # dataset_class = DATASETS.get(dataset_tag)
    #
    # dataset, data_loader = init_data_loader(cfg, data_path, './annotations/annotations/trainval.txt',
    #                                         dataset_class)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    feats = []
    with torch.no_grad():
        for img in image_names:
            img_path = img
            #print(os.path.exists(img_path))
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

            # tensor[:, :, int(-tensor.shape[2]/2):tensor.shape[2], :tensor.shape[3]] = \
            #     torch.zeros((1, 3, int(tensor.shape[2]/2), tensor.shape[3]))

            feat = model(tensor).squeeze()
            feats.append(feat)

    dists = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            dist = ((feats[i]-feats[j])**2).sum(axis=0)
            dists.append((dist, i, j))

    dists.sort(key=lambda x: int(x[0]))
    for dist, i, j in dists:
        print(i, j, dist)


def batch_extract(architectures: list, dataset_parts: list, datasets: list, dataset_root: str):
    cmd = 'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l'
    n_gpus = int(subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode())

    avail_pretrained_models = timm.list_models(pretrained=True)

    for architecture in architectures:

        if architecture not in avail_pretrained_models:
            print(f'Model {architecture} is not available.')
            continue

        model = timm.create_model(architecture, pretrained=True, num_classes=0)

        for dataset_tag, dataset_folder in datasets:
            for dataset_part in dataset_parts:
                print(f'Starting\t {architecture}\t {dataset_part} \t {dataset_tag}')

                output_dir = Path(dataset_root) / dataset_folder / 'features' / architecture
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file_path = output_dir / f'{dataset_part}.hdf5'

                cfg = get_config(base_config_path='../../mmclassification/configs/resnet/resnet50_b32x8_imagenet.py')
                cfg.test_pipeline[5] = {'type': 'Collect', 'keys': ['img', 'gt_label']}

                data_path = '/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset'
                dataset_class = DATASETS.get(dataset_tag)

                dataset, data_loader = init_data_loader(cfg, data_path, './annotations/annotations/trainval.txt',
                                                        dataset_class)

                with torch.no_grad():
                    with h5py.File(output_file_path, "w") as hf:
                        base_group = hf.create_group('feat')
                        base_group.create_dataset('image_names', data=np.empty(shape=(0, 0), dtype=h5py.string_dtype(
                            encoding='utf-8')), compression='gzip', chunks=True, maxshape=(None, 1))
                        base_group.create_dataset('gt_labels', data=np.empty(shape=(0, 0), dtype=np.int64),
                                                  compression='gzip',
                                                  chunks=True, maxshape=(None, 1))

                        # TODO: Pass the features size in the maxshape
                        base_group.create_dataset('data', data=np.empty(shape=(0, 0), dtype=np.float32),
                                                  compression='gzip',
                                                  chunks=True, maxshape=(None, None))

                        for i, batch in enumerate(data_loader):
                            print(f'Batch: {i}')
                            features = model(batch['img'])
                            gt_labels = batch['gt_label']
                            img_metas = batch['img_metas']

                            # Append the features
                            base_group['data'].resize((base_group['data'].shape[0] + features.shape[0]), axis=0)
                            base_group['data'][-features.shape[0]:] = features

                            # Append the gt labels
                            base_group['gt_labels'].resize((base_group['gt_labels'].shape[0] + gt_labels.shape[0]), axis=0)
                            base_group['gt_labels'][-gt_labels.shape[0]:] = gt_labels

                            # Append the image names
                            orig_filenames = []
                            for elem in img_metas.data[0]:
                                orig_filenames.append(elem['ori_filename'])

                            base_group['image_names'].resize((base_group['image_names'].shape[0] + len(orig_filenames)), axis=0)
                            base_group['image_names'][-len(orig_filenames):] = orig_filenames

                hf.close()


if __name__ == '__main__':
    # Options
    # TODO: Move options to script argument
    architectures = ['efficientnet_b3']
    dataset_parts = ['train']
    datasets = [('OxfordCatsDogsBreeds', 'cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset')]
    dataset_root = '/media/amidemo/Data/'
    batch_extract(architectures, dataset_parts, datasets, dataset_root)
