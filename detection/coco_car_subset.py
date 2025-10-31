# make_coco_car_subset.py
import json
from pycocotools.coco import COCO

def make_car_subset(orig_json, out_json, include_negatives=True):
    coco = COCO(orig_json)
    cat_id = coco.getCatIds(catNms=['car'])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    imgs = [coco.loadImgs(i)[0] for i in img_ids]
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)

    # Optionally include other images without car annotations
    if include_negatives:
        # add some images without car (example: take up to N random images that don't have car)
        all_img_ids = set(coco.getImgIds())
        neg_ids = list(all_img_ids - set(img_ids))
        # keep at most same number as positives (adjust as needed)
        neg_keep = neg_ids[:len(img_ids)]
        imgs += [coco.loadImgs(i)[0] for i in neg_keep]

    out = {
        'images': imgs,
        'annotations': anns,
        'categories': [coco.loadCats(cat_id)]
    }
    with open(out_json, 'w') as f:
        json.dump(out, f)
    print(f"Saved {len(imgs)} images and {len(anns)} anns to {out_json}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python make_coco_car_subset.py /path/to/instances_val2017.json out_val_car.json")
    else:
        make_car_subset(sys.argv[1], sys.argv[2])