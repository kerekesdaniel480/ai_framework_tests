
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


print("PyTorch version:", torch.__version__, "\n")

import torch
x = torch.rand(5, 3)
print(x)