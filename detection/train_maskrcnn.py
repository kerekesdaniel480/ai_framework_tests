import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import torch.optim as optim
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm

class CocoCarDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        
        # Csak az autó kategória (id=2 a COCO-ban)
        self.category_id = 1  # a car_subset már csak autókat tartalmaz, így ez mindig 1
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # Kép betöltése
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Annotációk betöltése
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Boxok és maszkok előkészítése
        boxes = []
        masks = []
        
        for ann in anns:
            if ann['area'] > 0:  # Csak valid annotációk
                # Box koordináták
                bbox = ann['bbox']  # [x, y, width, height] format
                # Konvertálás [x1, y1, x2, y2] formátumra
                x1, y1, w, h = bbox
                bbox = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
                
                # Validálás
                if w > 0 and h > 0:
                    boxes.append(bbox)
                    
                    # Maszk generálása
                    mask = self.coco.annToMask(ann)
                    masks.append(mask)
        
        # Ha nincs annotáció
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Labels (mind autó - 1)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        # Image ID
        image_id = torch.tensor([img_id])
        
        # Area
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Konvertálás tensor formátumra
        img = torchvision.transforms.ToTensor()(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)

def get_model():
    # Mask R-CNN model betöltése pretrained weights és backbone nélkül
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    
    # Osztályok számának beállítása (háttér + autó = 2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # Mask predictor módosítása
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, 2
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        print(f"Batch Loss: {losses.item():.4f}")

def main():
    # Adatok elérési útjai
    data_dir = "detection/coco/images/train2017"
    ann_file = "detection/coco/annotations/instances_train_car.json"
    
    # Dataset és DataLoader létrehozása
    dataset = CocoCarDataset(data_dir, ann_file)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Device beállítása (CPU)
    device = torch.device('cpu')
    
    # Model létrehozása és áthelyezése device-ra
    model = get_model()
    model.to(device)
    
    # Optimizer beállítása
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Output könyvtár létrehozása
    output_dir = "detection/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training
    num_epochs = 5
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, data_loader, device)
        
        # Model mentése
        torch.save(model.state_dict(), 
                  os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
        print(f"Model saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()