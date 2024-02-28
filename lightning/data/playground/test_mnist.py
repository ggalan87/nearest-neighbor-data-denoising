from typing import Tuple, Optional, Callable, Any, Dict
import random
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST, FashionMNIST
from lightning.data.datasets import DirtyMNIST

mnist_dataset = DirtyMNIST('./datasets', download=True, dirty_probability=1.0)
elem = mnist_dataset[0]
plt.imshow(elem['image'])
plt.show()
