import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import transforms

def get_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    
    # Box predictor módosítása
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask predictor módosítása
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      hidden_layer,
                                                      num_classes)
    return model

def load_model(model_path):
    device = torch.device('cpu')
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_image(model, image_path, device, confidence_threshold=0.5):
    # Kép betöltése és előfeldolgozása
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    # Predikció
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    # Eredmények feldolgozása
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_masks = prediction[0]['masks'].cpu().numpy()
    
    # Confidence threshold alkalmazása
    mask = pred_scores > confidence_threshold
    boxes = pred_boxes[mask]
    scores = pred_scores[mask]
    masks = pred_masks[mask]
    
    return boxes, scores, masks

def visualize_prediction(image_path, boxes, scores, masks):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Maszkok és boxok rajzolása
    for box, score, mask in zip(boxes, scores, masks):
        # Maszk rajzolása
        mask = mask[0]  # Squeeze the channel dimension
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0.5] = [255, 0, 0]  # Piros szín a maszknak
        image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)
        
        # Box rajzolása
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Score kiírása
        cv2.putText(image, f'Car: {score:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def main():
    # Model betöltése
    model_path = 'detection/outputs/model_epoch_5.pth'  # Az utolsó epoch modellje
    model, device = load_model(model_path)
    
    # Test képek könyvtára
    test_dir = 'detection/coco/images/val2017'
    output_dir = 'detection/outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Néhány teszt kép feldolgozása
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        print(f"Processing {image_file}...")
        
        # Predikció
        boxes, scores, masks = predict_image(model, image_path, device)
        
        # Vizualizáció
        result_image = visualize_prediction(image_path, boxes, scores, masks)
        
        # Eredmény mentése
        output_path = os.path.join(output_dir, f'pred_{image_file}')
        cv2.imwrite(output_path, result_image)
        print(f"Prediction saved to {output_path}")

if __name__ == '__main__':
    main()