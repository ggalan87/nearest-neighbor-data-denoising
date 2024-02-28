from typing import List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import pl_bolts.datamodules.cifar10_datamodule as cifar10_datamodule
from lightning.data.datasets import CIFAR10


class CIFAR10DataModule(cifar10_datamodule.CIFAR10DataModule):
    dataset_cls = CIFAR10