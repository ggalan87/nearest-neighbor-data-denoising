from .identity_datamodule import IdentityDataModule
from .controlled_populations_datamodule import ControlledPopulationsArgs, ControlledPopulationsDataModule
from .noisymnist_datamodule import NoisyMNISTDataModule, NoisyMNISTSubsetDataModule, NoisyMNISTArgs, \
    NoisyMNISTSubsetArgs
from .mnist_datamodule import MNISTDataModule, MNISTSubsetDataModule, MNISTSubsetArgs
from .fashionmnist_datamodule import FashionMNISTDataModule
from .cars_datamodule import CarsDataModule, CarsArgs
from .cars98n_datamodule import Cars98NDataModule
from .birds_datamodule import BirdsDataModule, BirdsArgs
from .online_products_datamodule import OnlineProductsDataModule, OnlineProductsArgs
from .food101n_datamodule import Food101NDataModule, Food101NArgs
from .noisycifar_datamodule import NoisyCIFAR10DataModule, NoisyCIFARArgs, NoisyCIFAR10Args
from .cifar_datamodule import CIFAR10DataModule

# also provide some aliases
KMNISTDataModule = MNISTDataModule
