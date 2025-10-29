from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/coco/annotations/instances_train2017.json",
    save_dir="/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/coco/labels/train2017",
    use_keypoints=True,
)