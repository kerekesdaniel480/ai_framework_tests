import os
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(checkpoint_path, num_classes=2):
    # Backbone és modell létrehozása
    backbone = resnet_fpn_backbone('resnet50', weights=None)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=100,
                     max_size=800)
    
    # Checkpoint betöltése
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image_path, max_size=800):
    # Kép betöltése és előfeldolgozása
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    
    # Méretezés, ha szükséges
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Konvertálás tensor-rá
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    return image, image_tensor

def visualize_prediction(image, prediction, output_path, confidence_threshold=0.85):
    # image: PIL Image or numpy uint8 HxWx3
    if isinstance(image, Image.Image):
        img_np = np.array(image).astype(np.uint8)
    else:
        img_np = image.astype(np.uint8)

    masks = prediction['masks'].cpu().numpy()  # N x 1 x H x W
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    h, w = img_np.shape[:2]
    overlay = img_np.astype(np.float32).copy()
    output = img_np.astype(np.float32).copy()

    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))[:, :3] * 255

    detected_objects = 0
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if score < confidence_threshold:
            continue
        detected_objects += 1
        # mask: 1 x H x W
        m = mask[0]
        # resize mask if needed
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

        m_bin = (m > 0.5).astype(np.uint8)
        color = colors[i % len(colors)]
        alpha = 0.5

        # Blend color onto overlay where mask is 1
        for c in range(3):
            overlay[:, :, c] = np.where(m_bin == 1,
                                        overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                        overlay[:, :, c])

        # Draw bbox on output
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color=tuple(map(int, color)), thickness=2)
        cv2.putText(overlay, f'Car:{score:.2f}', (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1)

    # Finalize and save
    out_img = np.clip(overlay, 0, 255).astype(np.uint8)
    plt.figure(figsize=(12, 8))
    plt.imshow(out_img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    return detected_objects

def main():
    # Modell checkpoint betöltése
    checkpoint_path = "detection/outputs/run_20251031_151954/model_epoch_0.pth"
    model = load_model(checkpoint_path)
    
    # Test könyvtár beállítása
    test_dir = "detection/coco/images/val2017"
    output_dir = "detection/outputs/test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Néhány teszt kép feldolgozása
    test_images = os.listdir(test_dir)[:5]  # első 5 kép tesztelése
    device = torch.device('cpu')
    
    successful_detections = []
    
    for img_name in test_images:
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        print(f"Processing {img_name}...")
        img_path = os.path.join(test_dir, img_name)
        
        # Kép előfeldolgozása
        original_image, image_tensor = preprocess_image(img_path)
        
        # Predikció
        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]
        
        # Eredmény mentése
        output_path = os.path.join(output_dir, f"pred_{img_name}")
        num_detections = visualize_prediction(original_image, prediction, output_path)
        
        if num_detections > 0:
            successful_detections.append({
                'image': img_name,
                'detections': num_detections,
                'output_path': output_path
            })
        
        print(f"Found {num_detections} cars in {img_name}")
    
    # Összegzés
    print("\nSuccessful detections:")
    for det in successful_detections:
        print(f"Image: {det['image']}, Cars detected: {det['detections']}")
        print(f"Result saved to: {det['output_path']}")

if __name__ == "__main__":
    main()