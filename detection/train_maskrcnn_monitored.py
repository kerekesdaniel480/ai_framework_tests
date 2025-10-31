import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import cv2
import datetime
import psutil
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class CarDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None, max_size=800):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.max_size = max_size
        
    def __getitem__(self, idx):
        try:
            coco = self.coco
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            coco_annotation = coco.loadAnns(ann_ids)
            
            path = coco.loadImgs(img_id)[0]['file_name']
            img_path = os.path.join(self.root, path)
            img = Image.open(img_path).convert('RGB')
            
            # Store original image path for visualization
            original_path = img_path
            
            w, h = img.size
            scale = min(1.0, self.max_size / max(w, h))
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            num_objs = len(coco_annotation)
            if num_objs == 0:
                return self.__getitem__((idx + 1) % len(self))
            
            boxes = []
            masks = []
            valid_anns = []
            
            for ann in coco_annotation:
                try:
                    xmin = float(ann['bbox'][0])
                    ymin = float(ann['bbox'][1])
                    w = float(ann['bbox'][2])
                    h = float(ann['bbox'][3])
                    
                    if w > 1 and h > 1:
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
                return self.__getitem__((idx + 1) % len(self))
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            
            num_objs = len(boxes)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["original_path"] = original_path
            
            if self.transform is not None:
                img = self.transform(img)
            
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            return img, target
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
    
    def __len__(self):
        return len(self.ids)

class ResourceMonitor:
    def __init__(self, log_dir):
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.start_time = time.time()
        self.log_dir = log_dir
        
    def update(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.Process().memory_percent()
        current_time = time.time() - self.start_time
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        self.timestamps.append(current_time)
    
    def plot_and_save(self):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.timestamps, self.cpu_usage)
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('CPU Usage (%)')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.timestamps, self.memory_usage)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'resource_usage.png'))
        plt.close()

def visualize_prediction(image, prediction, output_path):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    masks = prediction['masks'].cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    for mask, box, score in zip(masks, boxes, scores):
        if score > 0.5:  # csak a jó predikciók
            plt.gca().add_patch(
                patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
            )
            
            mask = mask[0]  # első csatorna
            mask_colored = np.zeros_like(image)
            mask_colored[:, :, 0] = mask * 1.0  # piros maszk
            plt.imshow(mask_colored, alpha=0.3)
            
            plt.text(
                box[0], box[1],
                f'Car: {score:.2f}',
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=8
            )
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_instance_segmentation_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', weights=None)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=100,
                     max_size=800)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, resource_monitor, vis_dir, print_freq=10):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        try:
            # Erőforrás monitoring
            resource_monitor.update()
            
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                       for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f'Loss is {losses}, skipping batch')
                continue
                
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            # Predikciók vizualizálása training közben
            if i % print_freq == 0:
                avg_loss = total_loss / max(1, num_batches)
                print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] Avg Loss: {avg_loss:.4f}')
                print('Per-task losses:')
                for k, v in loss_dict.items():
                    print(f'{k}: {v.item():.4f}')
                
                # Teszt predikció az aktuális batch első képére
                model.eval()
                with torch.no_grad():
                    prediction = model([images[0]])[0]
                model.train()
                
                # Eredeti kép betöltése és vizualizáció
                img_path = targets[0]['original_path']
                orig_img = cv2.imread(img_path)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                vis_path = os.path.join(vis_dir, f'epoch_{epoch}_batch_{i}.png')
                visualize_prediction(orig_img, prediction, vis_path)
        
        except Exception as e:
            print(f"Error in training batch {i}: {str(e)}")
            continue

def main():
    # Könyvtárak létrehozása
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"detection/outputs/run_{timestamp}"
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Erőforrás monitor inicializálása
    resource_monitor = ResourceMonitor(output_dir)
    
    data_dir = "detection/coco/images/train2017"
    ann_file = "detection/coco/annotations/instances_train_car.json"
    
    dataset = CarDataset(data_dir, ann_file, max_size=800)
    
    dataset_size = len(dataset)
    train_size = min(10, dataset_size)
    indices = torch.randperm(dataset_size)[:train_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    device = torch.device('cpu')
    
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 
                              lr=0.0001,
                              momentum=0.9, 
                              weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=3,
                                                 gamma=0.1)
    
    num_epochs = 1
    print(f"Starting training with {train_size} images for {num_epochs} epochs...")
    print(f"Outputs will be saved to: {output_dir}")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, resource_monitor, vis_dir)
        lr_scheduler.step()
        
        checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")
    
    # Erőforrás használat plot mentése
    resource_monitor.plot_and_save()
    print(f"Resource usage plots saved to {output_dir}/resource_usage.png")

if __name__ == "__main__":
    main()