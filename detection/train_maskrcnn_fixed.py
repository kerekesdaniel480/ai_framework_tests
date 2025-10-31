import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import cv2
import datetime
import psutil
import time
import traceback
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F

class CarDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None, max_size=800, car_only=True):
        self.root = root
        self.coco = COCO(annFile)
        # Optionally restrict to car-only annotations or use full COCO labels
        self.car_only = car_only
        self.transform = transform

        if self.car_only:
            # Determine which category id corresponds to 'car' in this annotation file
            car_cat_ids = []
            for c in self.coco.loadCats(self.coco.getCatIds()):
                if str(c.get('name','')).lower() == 'car':
                    car_cat_ids.append(c['id'])

            # Only keep images that have at least one valid 'car' annotation
            all_ids = list(sorted(self.coco.imgs.keys()))
            valid_ids = []
            for img_id in all_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                keep = False
                for ann in anns:
                    try:
                        if car_cat_ids and ann.get('category_id', -1) not in car_cat_ids:
                            continue
                        bbox = ann.get('bbox', None)
                        if not bbox or len(bbox) < 4:
                            continue
                        bw = float(bbox[2])
                        bh = float(bbox[3])
                        if bw > 10 and bh > 10:
                            keep = True
                            break
                    except Exception:
                        continue
                if keep:
                    valid_ids.append(img_id)

            self.ids = valid_ids
            self.car_cat_ids = car_cat_ids
        else:
            # Full COCO: include all images that have at least one annotation
            all_ids = list(sorted(self.coco.imgs.keys()))
            valid_ids = []
            for img_id in all_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if len(ann_ids) > 0:
                    valid_ids.append(img_id)
            self.ids = valid_ids

            # Build category id -> contiguous label mapping for training (labels start at 1)
            cat_ids = sorted(self.coco.getCatIds())
            self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        self.max_size = max_size
        
    def __getitem__(self, idx):
        max_retries = 10
        current_idx = idx
        
        for _ in range(max_retries):
            try:
                coco = self.coco
                img_id = self.ids[current_idx]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                coco_annotation = coco.loadAnns(ann_ids)
                
                path = coco.loadImgs(img_id)[0]['file_name']
                img_path = os.path.join(self.root, path)
                img = Image.open(img_path).convert('RGB')
                
                w, h = img.size
                scale = min(1.0, self.max_size / max(w, h))
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                
                boxes = []
                masks = []
                valid_anns = []
                
                for ann in coco_annotation:
                    try:
                        if self.car_only:
                            # check using detected car category ids
                            if self.car_cat_ids and ann.get('category_id', -1) not in self.car_cat_ids:
                                continue
                            
                        xmin = float(ann['bbox'][0])
                        ymin = float(ann['bbox'][1])
                        w = float(ann['bbox'][2])
                        h = float(ann['bbox'][3])
                        
                        if w > 10 and h > 10:  # Minimális méretű objektumok szűrése
                            if scale < 1.0:
                                xmin *= scale
                                ymin *= scale
                                w *= scale
                                h *= scale
                            
                            xmax = xmin + w
                            ymax = ymin + h
                            
                            if xmax > xmin and ymax > ymin:
                                boxes.append([xmin, ymin, xmax, ymax])
                                mask = coco.annToMask(ann)
                                if scale < 1.0:
                                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                                masks.append(mask)
                                valid_anns.append(ann)
                    except:
                        continue
                
                if len(boxes) == 0:
                    current_idx = (current_idx + 1) % len(self)
                    continue
                
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                
                num_objs = len(boxes)
                if self.car_only:
                    labels = torch.ones((num_objs,), dtype=torch.int64)
                else:
                    # map COCO category ids to contiguous labels
                    lbls = []
                    for ann in valid_anns:
                        cid = ann.get('category_id', -1)
                        lbl = self.cat_id_to_label.get(cid, 0)
                        lbls.append(lbl)
                    labels = torch.as_tensor(lbls, dtype=torch.int64)
                image_id = torch.tensor([current_idx])
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
                
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["masks"] = masks
                target["image_id"] = image_id
                target["area"] = area
                target["iscrowd"] = iscrowd
                
                if self.transform is not None:
                    img = self.transform(img)
                
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                return img, target
                
            except Exception as e:
                print(f"Error processing image {current_idx}: {str(e)}")
                traceback.print_exc()
                current_idx = (current_idx + 1) % len(self)
        
        raise RuntimeError(f"Failed to load any valid image after {max_retries} attempts starting from index {idx}")
    
    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_prediction(image, prediction, output_path, confidence_threshold=0.7):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    masks = prediction['masks'].cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # Különböző színek a maszkokhoz
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    detected_objects = 0
    for mask, box, score, color in zip(masks, boxes, scores, colors):
        if score > confidence_threshold:
            detected_objects += 1
            # Bounding box
            plt.gca().add_patch(
                patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
            )
            
            # Maszk megjelenítése
            mask = mask[0]
            mask_colored = np.zeros((mask.shape[0], mask.shape[1], 4))
            mask_colored[mask > 0.5] = color
            mask_colored[..., 3] = mask * 0.5  # átlátszóság
            plt.imshow(mask_colored)
            
            # Score megjelenítése
            plt.text(
                box[0], box[1] - 5,
                f'Car: {score:.2f}',
                bbox=dict(facecolor='white', alpha=0.8),
                color=tuple(color[:3]),
                fontsize=8
            )
    
    plt.title(f'Detected {detected_objects} cars')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    return detected_objects

def get_instance_segmentation_model(num_classes):
    # Erősebb backbone használata
    backbone = resnet_fpn_backbone('resnet101', weights=None)
    
    # Modell létrehozása jobb paraméterekkel
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=400,  # Nagyobb minimum méret
                     max_size=800,
                     box_detections_per_img=100,  # Több detekció engedélyezése
                     box_score_thresh=0.05,  # Alacsonyabb kezdeti küszöb
                     box_nms_thresh=0.5)  # Szigorúbb NMS
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_mask = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_classifier'].item()
        running_loss_box_reg += loss_dict['loss_box_reg'].item()
        running_loss_mask += loss_dict['loss_mask'].item()
        
        losses.backward()
        
        # Gradient clipping a stabilitás érdekében
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}]')
            print(f'Loss: {losses.item():.4f}')
            print(f'Classifier: {loss_dict["loss_classifier"].item():.4f}')
            print(f'Box Reg: {loss_dict["loss_box_reg"].item():.4f}')
            print(f'Mask: {loss_dict["loss_mask"].item():.4f}')
    
    avg_loss = running_loss / len(data_loader)
    avg_loss_classifier = running_loss_classifier / len(data_loader)
    avg_loss_box_reg = running_loss_box_reg / len(data_loader)
    avg_loss_mask = running_loss_mask / len(data_loader)
    
    return {
        'loss': avg_loss,
        'loss_classifier': avg_loss_classifier,
        'loss_box_reg': avg_loss_box_reg,
        'loss_mask': avg_loss_mask
    }

def main():
    # Adatok és könyvtárak beállítása
    data_dir = "detection/coco"
    # Option A: use original COCO annotations and COCO-pretrained weights
    USE_COCO_PRETRAINED = True
    if USE_COCO_PRETRAINED:
        train_ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
        train_img_dir = os.path.join(data_dir, "images/train2017")
    else:
        train_ann_file = os.path.join(data_dir, "annotations/instances_train_car.json")
        train_img_dir = os.path.join(data_dir, "images/train2017")
    
    # Output könyvtár létrehozása
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("detection/outputs", f"run_{timestamp}")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Dataset és DataLoader
    # if using COCO-pretrained and original COCO annotations, load full COCO dataset
    dataset = CarDataset(train_img_dir, train_ann_file, car_only=not USE_COCO_PRETRAINED)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, 
                           num_workers=0, collate_fn=collate_fn)
    
    # Eszköz beállítása
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Modell létrehozása és áthelyezése az eszközre
    if USE_COCO_PRETRAINED:
        # load torchvision COCO-pretrained Mask R-CNN (weights will be downloaded if needed)
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            model = maskrcnn_resnet50_fpn(weights=weights)
            print('Loaded COCO-pretrained Mask R-CNN')
        except Exception:
            # fallback to older API
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            model = maskrcnn_resnet50_fpn(pretrained=True)
            print('Loaded COCO-pretrained Mask R-CNN (fallback)')
    else:
        num_classes = 2  # háttér + autó
        model = get_instance_segmentation_model(num_classes)
    model.to(device)
    
    # Optimalizáló beállítása
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    
    # Learning rate scheduler
    # For a quick sanity run keep epochs small; raise this for full training
    num_epochs = 1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.2
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # Train one epoch
        epoch_stats = train_one_epoch(model, optimizer, data_loader, 
                                    device, epoch)
        
        lr_scheduler.step()
        
        # Modell mentése
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_stats
        }
        torch.save(checkpoint, os.path.join(output_dir, f'model_epoch_{epoch}.pth'))
        
        print(f'Epoch {epoch} stats:')
        for k, v in epoch_stats.items():
            print(f'{k}: {v:.4f}')
        
        # Teszt egy képen
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                images, targets = next(iter(data_loader))
                prediction = model([images[0].to(device)])[0]
                img = images[0].cpu().numpy().transpose(1, 2, 0)
                visualize_prediction(img, prediction, 
                                  os.path.join(vis_dir, f'train_vis_epoch{epoch}.jpg'))
            model.train()
    
    print('Training completed')

if __name__ == "__main__":
    main()