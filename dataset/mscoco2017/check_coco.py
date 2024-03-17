import json

coco_json_path = './annotations_trainval2017/annotations/instances_val2017.json'
coco_labels = json.load(open(coco_json_path,'r'))
print(coco_labels['info'])
print(coco_labels.keys())
