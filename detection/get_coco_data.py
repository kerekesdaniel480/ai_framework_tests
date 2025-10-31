from pycocotools.coco import COCO
c = COCO('/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/detection/coco/annotations/instances_train2017.json')
cats = c.loadCats(c.getCatIds('car'))
print(cats)