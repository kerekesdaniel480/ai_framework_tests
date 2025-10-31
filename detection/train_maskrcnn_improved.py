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
                
                # Store original image path for visualization
                original_path = img_path
                
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
                        if ann['category_id'] != 3:  # Csak autók (COCO category_id=3)
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
                labels = torch.ones((num_objs,), dtype=torch.int64)
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
                target["original_path"] = original_path
                
                if self.transform is not None:
                    img = self.transform(img)
                
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                return img, target
                
            except Exception as e:
                print(f"Error processing image {current_idx}: {str(e)}")
                current_idx = (current_idx + 1) % len(self)
        
        # Ha minden próbálkozás sikertelen volt, dobjunk kivételt
        raise RuntimeError(f"Failed to load any valid image after {max_retries} attempts starting from index {idx}")
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["original_path"] = original_path
            
            # Adataugmentáció
            if np.random.random() > 0.5:
                img = F.hflip(img)
                target["boxes"][:, [0, 2]] = img.size[0] - target["boxes"][:, [2, 0]]
                target["masks"] = torch.flip(target["masks"], [2])
            
            # Kontraszt és fényerő változtatás
            if np.random.random() > 0.5:
                img = F.adjust_brightness(img, brightness_factor=np.random.uniform(0.8, 1.2))
                img = F.adjust_contrast(img, contrast_factor=np.random.uniform(0.8, 1.2))
            
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
    
    # Különböző színek a maszkokhoz
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    for mask, box, score, color in zip(masks, boxes, scores, colors):
        if score > 0.5:  # csak a jó predikciók
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
            
            mask = mask[0]  # első csatorna
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0.5] = color[:3]  # csak az RGB csatornák
            plt.imshow(mask_colored, alpha=0.3)
            
            plt.text(
                box[0], box[1] - 5,
                f'Car: {score:.2f}',
                bbox=dict(facecolor='white', alpha=0.8),
                color=tuple(color[:3]),
                fontsize=8
            )
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, resource_monitor, vis_dir, print_freq=10):
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
        
        if resource_monitor:
            resource_monitor.update()
            
        # Visualization during training
        if i % 100 == 0:
            model.eval()
            with torch.no_grad():
                prediction = model(images[:1])[0]
                img = images[0].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                visualize_prediction(img, prediction, 
                                  os.path.join(vis_dir, f'train_vis_epoch{epoch}_iter{i}.jpg'))
            model.train()
    
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
    train_ann_file = os.path.join(data_dir, "annotations/instances_train_car.json")
    train_img_dir = os.path.join(data_dir, "images/train2017")
    
    # Output könyvtár létrehozása
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("detection/outputs", f"run_{timestamp}")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Dataset és DataLoader
    dataset = CarDataset(train_img_dir, train_ann_file)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, 
                           num_workers=0, collate_fn=collate_fn)
    
    # Eszköz beállítása
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Modell létrehozása és áthelyezése az eszközre
    num_classes = 2  # háttér + autó
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    
    # Optimalizáló beállítása
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    
    # Learning rate scheduler
    num_epochs = 10
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.2
    )
    
    # Resource monitor inicializálása
    resource_monitor = ResourceMonitor(output_dir)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # Train one epoch
        epoch_stats = train_one_epoch(model, optimizer, data_loader, 
                                    device, epoch, resource_monitor, vis_dir)
        
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
    
    # Resource használat mentése
    resource_monitor.plot_and_save()
    
    print('Training completed')

if __name__ == "__main__":
    main()