"""Run COCO-pretrained Mask R-CNN on a small testset and save overlays.

This script uses torchvision's Mask R-CNN with COCO weights (downloads once
into the torch cache). It reuses `preprocess_image` and `visualize_prediction`
from `detection.test_maskrcnn` which are already present in the repo.
"""
import os
import sys
import torch

# allow running the script directly from the repo root
sys.path.append(os.path.abspath('.'))
from torchvision.models.detection import maskrcnn_resnet50_fpn
try:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
    WEIGHTS = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
except Exception:
    WEIGHTS = None

from detection.test_maskrcnn import preprocess_image, visualize_prediction


def run_on_images(image_names, images_root="detection/coco/images/val2017", out_dir="detection/outputs/pretrained_testset", confidence_threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    # load pretrained model (weights enum if available)
    if WEIGHTS is not None:
        model = maskrcnn_resnet50_fpn(weights=WEIGHTS)
    else:
        # fallback to older API that may accept pretrained=True
        model = maskrcnn_resnet50_fpn(pretrained=True)

    model.eval()
    device = torch.device('cpu')

    for name in image_names:
        img_path = os.path.join(images_root, name)
        if not os.path.isfile(img_path):
            print(f"Skipping missing image: {img_path}")
            continue
        try:
            img, tensor = preprocess_image(img_path)
            with torch.no_grad():
                pred = model([tensor.to(device)])[0]
            out_path = os.path.join(out_dir, f"{name}_pretrained_overlay.jpg")
            visualize_prediction(img, pred, out_path, confidence_threshold=confidence_threshold)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed on {name}: {e}")


if __name__ == "__main__":
    # small set of images used previously in this repo
    test_images = [
        '000000182611.jpg',
        '000000335177.jpg',
        '000000278705.jpg',
        '000000463618.jpg',
        '000000568981.jpg',
    ]
    run_on_images(test_images)
