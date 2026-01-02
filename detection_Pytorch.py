# Object Detection Implementation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

train_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/train")
test_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/test")
valid_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/valid")

train=pd.read_csv("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/train/_annotations.csv")

train[['xmin', 'ymin', 'xmax', 'ymax']] = train[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
train.drop_duplicates(subset='filename', inplace=True, ignore_index=True)

def display_image(img, bbox_coorsd=[], pred_coords=[], norm=False):
    if norm:
        img *= 255.
        img = img.astype(np.uint8)

    img = img.copy()

    if len(bbox_coorsd) == 4:
        xmin, ymin, xmax, ymax = bbox_coorsd
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)

    if len(pred_coords) == 4:
        xmin, ymin, xmax, ymax = pred_coords
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def display_image_from_file(name, bbox_coorsd=[], path=train_path):
    img = cv.imread(str(path / name))
    display_image(img, bbox_coorsd=bbox_coorsd)

def display_from_dataframe(row, path=train_path):
    display_image_from_file(row['filename'], bbox_coorsd=(row.xmin, row.ymin, row.xmax, row.ymax), path=path)

def display_grid(df=train, n_item=3):
    plt.figure(figsize=(20, 10))

    rand_incidents = [np.random.randint(0, df.shape[0]) for _ in range(n_item)]

    for pos, index in enumerate(rand_incidents):
        plt.subplot(1, n_item, pos + 1)
        display_from_dataframe(df.loc[index, :])

# PyTorch Dataset osztály 
class DogDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.path / row['filename']
        img = cv.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv.resize(img, (640, 640)) / 255.0  # Resize és normalizálás
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W) PyTorch-hoz
        bbox = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32)
        return img, bbox

# DataLoader létrehozása (batch-ek, shuffling)
dataset = DogDataset(train, train_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Teszt - egy batch kiolvasása
"""
for images, bboxes in dataloader:
    print(images.shape, bboxes.shape)  # (batch_size, 3, 640, 640), (batch_size, 4)
    break
    """

# PyTorch Modell 
class DogDetectionModel(nn.Module):
    def __init__(self):
        super(DogDetectionModel, self).__init__()
        self.conv_layers = nn.Sequential()
        for i in range(10):  # Vissza 10-re, padding=1 miatt
            n_filters = 2**(i+3)  # Ugyanaz a szűrő szám, mint TF-ben
            self.conv_layers.add_module(f'conv_{i}', nn.Conv2d(3 if i == 0 else 2**(i+2), n_filters, kernel_size=3, padding=1))
            self.conv_layers.add_module(f'bn_{i}', nn.BatchNorm2d(n_filters))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool2d(2, padding=1))  # padding=1 'same' effektus
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Mindig 1x1-re pool-ol
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 256),  # Vissza 4096-ra, padding=1 miatt
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # 1x1-re
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# Modell példányosítása
model = DogDetectionModel()
print(model)  

criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-3)  


def compute_accuracy(preds, targets, threshold=10):  
    diff = torch.abs(preds - targets)
    correct = (diff < threshold).all(dim=1).float()
    return correct.mean().item()


# Test függvények 
def test_model(model, dataloader_iter):
    try:
        images, bboxes = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)  
        images, bboxes = next(dataloader_iter)
    
    with torch.no_grad():
        pred_bbox = model(images).cpu().numpy()[0]  
    
    img = images[0].cpu().numpy().transpose(1, 2, 0)  
    gt_coords = bboxes[0].cpu().numpy()
    
    display_image(img, pred_coords=pred_bbox, norm=True)

def test(model):
    dataloader_iter = iter(dataloader)
    
    plt.figure(figsize=(15, 7))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        test_model(model, dataloader_iter)
    plt.show()  


def show_test_images(model, epoch):
    print(f"Epoch {epoch+1} vége - Teszt képek:")
    test(model)

test(model)

# Training loop 
device = torch.device('cpu')  
model.to(device)

epochs = 10
steps_per_epoch = 500  

for epoch in range(epochs):
    model.train()  
    running_loss = 0.0
    for step, (images, bboxes) in enumerate(dataloader):
        if step >= steps_per_epoch:
            break
        images, bboxes = images.to(device), bboxes.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / steps_per_epoch
    print(f'Epoch {epoch+1}/{epochs} vége, Átlagos Loss: {avg_loss:.4f}')
    
    show_test_images(model, epoch)

# Modell mentés 
torch.save(model.state_dict(), 'dog_detection.pth')
print("Modell mentve: dog_detection.pth")

