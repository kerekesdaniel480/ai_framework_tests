
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#import os

data_dir = '/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog/train'

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("PyTorch version:", torch.__version__, "\n")

import torch
x = torch.rand(5, 3)
print(x)