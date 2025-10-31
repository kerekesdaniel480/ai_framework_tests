import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
import cv2
import torch
from detection.test_maskrcnn import load_model, preprocess_image

CHECKPOINT = 'detection/outputs/run_20251031_151954/model_epoch_0.pth'
IMG_NAME = '000000182611.jpg'
TEST_DIR = 'detection/coco/images/val2017'
OUT_DIR = 'detection/outputs/analysis_examples'

os.makedirs(OUT_DIR, exist_ok=True)

model = load_model(CHECKPOINT)
model.eval()
device = torch.device('cpu')

orig, tensor = preprocess_image(os.path.join(TEST_DIR, IMG_NAME))
with torch.no_grad():
    pred = model([tensor.to(device)])[0]

boxes = pred['boxes'].cpu().numpy()
scores = pred['scores'].cpu().numpy()
masks = pred['masks'].cpu().numpy()

H, W = np.array(orig).shape[:2]

print('Total detections:', len(scores))

# Save original overlay with low threshold for reference
from detection.test_maskrcnn import visualize_prediction as vis_fn
vis_fn(orig, pred, os.path.join(OUT_DIR, IMG_NAME + '_orig_overlay.jpg'), confidence_threshold=0.3)

# Filtering heuristics
min_score = 0.7
min_bbox_area = 2000  # pixels

keep_idxs = []
for i, s in enumerate(scores):
    if s < min_score:
        continue
    box = boxes[i].astype(int)
    x1,y1,x2,y2 = box
    bw = max(0, x2-x1); bh = max(0, y2-y1)
    if bw*bh < min_bbox_area:
        continue
    # compute mask area
    mask = masks[i,0]
    if mask.shape != (H,W):
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask>0.5).astype(np.uint8)
    mask_area = int(mask_bin.sum())
    # require mask_area to be at least 200 pixels
    if mask_area < 200:
        continue
    keep_idxs.append(i)

print('Kept after filtering:', len(keep_idxs))

# Build filtered prediction dict
filtered_pred = {k: v[keep_idxs] for k, v in pred.items()} if len(keep_idxs)>0 else {'boxes': torch.zeros((0,4)), 'scores': torch.zeros((0,)), 'labels': torch.zeros((0,), dtype=torch.int64), 'masks': torch.zeros((0,1,H,W))}

# Save filtered overlay
# adapt visualize function from test_maskrcnn to accept our filtered_pred
from detection.test_maskrcnn import visualize_prediction as vis_fn2
# Our vis function expects tensors - convert filtered_pred items to tensors if needed
if isinstance(filtered_pred.get('boxes'), np.ndarray):
    import torch
    filtered_pred_t = {
        'boxes': torch.from_numpy(filtered_pred['boxes']).float(),
        'scores': torch.from_numpy(filtered_pred['scores']).float(),
        'labels': torch.from_numpy(filtered_pred['labels']).long() if 'labels' in filtered_pred else torch.zeros((0,),dtype=torch.int64),
        'masks': torch.from_numpy(filtered_pred['masks']).float()
    }
else:
    filtered_pred_t = filtered_pred

# The vis function saves the image
vis_fn2(orig, filtered_pred_t, os.path.join(OUT_DIR, IMG_NAME + '_filtered_overlay.jpg'), confidence_threshold=0.0)

print('Saved original and filtered overlays to', OUT_DIR)
