from typing import Union
from pathlib import Path
import torch
from pytorch_lightning import LightningModule
from torchviz import make_dot


class ModelVisualizer:
    def __init__(self, output_path: Path):
        self._output_path = output_path

    def visualize(self, model: LightningModule, sample: torch.Tensor, verbose=False):
        output = model(sample)
        make_dot(output, params=dict(model.named_parameters()), show_attrs=verbose, show_saved=verbose)\
            .render(directory=self._output_path, filename=f'{model.__class__.__name__}', format="png")
