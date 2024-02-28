from typing import List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import pl_bolts.datamodules.fashion_mnist_datamodule as fashionmnist_datamodule
from lightning.data.datasets import FashionMNIST


class FashionMNISTDataModule(fashionmnist_datamodule.FashionMNISTDataModule):
    dataset_cls = FashionMNIST

