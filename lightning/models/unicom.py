import hashlib
import os
import urllib
import warnings
from typing import Optional, Type, Dict, Literal, Tuple, Union, List
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses, miners
from lightning.models.model_base import LitModelBase, Flatten
from warnings import warn
from torchmetrics.functional import accuracy

from torch_metric_learning import noise_reducers
from .backbones.vision_tranformer import load_model_and_transform, build_model

_MODELS = {
    "ViT-B/32": "https://github.com/deepglint/unicom/releases/download/b32/FP16-ViT-B-32.pt",
    "ViT-B/16": "https://github.com/deepglint/unicom/releases/download/b16/FP16-ViT-B-16.pt",
    "ViT-L/14": "https://github.com/deepglint/unicom/releases/download/l14/FP16-ViT-L-14.pt",
    "ViT-L/14@336px": "https://github.com/deepglint/unicom/releases/download/l14_336px/FP16-ViT-L-14-336px.pt",
}

_SHA256 = {
    "FP16-ViT-B-32.pt": "f9d5696a9b58dbbbefee2d31615ca59084f2895a0fdd2ca4c235e0f9b2793f7a",
    "FP16-ViT-B-16.pt": "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    "FP16-ViT-L-14.pt": "ff3ab62ff782876460099e6e0ee17b73a7c01109de2fffd595f16f4129404bbd",
    "FP16-ViT-L-14-336px.pt": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
}


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L43
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = _SHA256[filename]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class LitUnicom(LitModelBase):
    def __init__(self,
                 # Common model options
                 batch_size=256,
                 num_classes=1000,
                 num_channels=3,
                 use_pretrained_weights=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[Type[_LRScheduler]] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 loss_class: Union[Type[losses.BaseMetricLossFunction], Type[losses.BaseLossWrapper]] = losses.TripletMarginLoss,
                 loss_kwargs: Optional[Dict] = None,
                 miner_class: Type[miners.BaseMiner] = miners.BatchHardMiner,
                 miner_kwargs: Optional[Dict] = None,
                 noise_reducer_class: Type[noise_reducers.DefaultNoiseReducer] = noise_reducers.DefaultNoiseReducer,
                 noise_reducer_kwargs: Optional[Dict] = None,
                 # Custom options
                 model_variant: Literal['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] = 'ViT-B/32'
                 ):
        super().__init__(batch_size=batch_size,
                         num_classes=num_classes,
                         num_channels=num_channels,
                         use_pretrained_weights=use_pretrained_weights,
                         loss_class=loss_class,
                         loss_kwargs=loss_kwargs,
                         miner_class=miner_class,
                         miner_kwargs=miner_kwargs,
                         noise_reducer_class=noise_reducer_class,
                         noise_reducer_kwargs=noise_reducer_kwargs)

        self.save_hyperparameters()

    def _create_model(self) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
        """

        @return: backbone and linear layer
        """
        name = self.hparams.model_variant
        if name in _MODELS:
            model_path = _download(
                _MODELS[name], os.path.expanduser("~/.cache/unicom"))
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {available_models()}")
        with open(model_path, 'rb') as opened_file:
            state_dict = torch.load(opened_file, map_location="cpu")

        model = build_model(name)

        if self.hparams.use_pretrained_weights:
            state_dict_fp32 = {}
            for k, v in state_dict.items():
                state_dict_fp32[k] = v.float()

            model.load_state_dict(state_dict)

        if name == "ViT-B/32":
            embedding_size = 512
        else:
            embedding_size = 768

        fc = \
            nn.Sequential(
                Flatten(),
                nn.Linear(embedding_size, self.hparams.num_classes)
            )

        # TODO: Original fc is more sophisticated, but I don't currently adopt it
        return model, fc, embedding_size
