import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detection.test_maskrcnn import load_model, preprocess_image, visualize_prediction
import torch

checkpoint_path = "detection/outputs/run_20251031_151954/model_epoch_0.pth"
model = load_model(checkpoint_path)

test_dir = "detection/coco/images/val2017"
output_dir = "detection/outputs/test_results_lowth"
os.makedirs(output_dir, exist_ok=True)

test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]

device = torch.device('cpu')
for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    original_image, image_tensor = preprocess_image(img_path)
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]
    out_path = os.path.join(output_dir, 'pred_'+img_name)
    num = visualize_prediction(original_image, prediction, out_path, confidence_threshold=0.5)
    print(img_name, 'detections', num)
