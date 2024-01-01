import os
from pycocotools.coco import COCO

coco_json_path = './mscoco2017/annotations_trainval2017/annotations/instances_val2017.json'
coco_img_path = './val2017/val2017'

coco = COCO(annotation_file=coco_json_path)
print(list(coco.imgs.keys())[1])
# boxes = []
# for i,item in enumerate(coco.anns.items()):
#     if item[1]['image_id'] == 37777:
#         print(item[1]['bbox'],item[1]['category_id'])
    # print(item)
    # if i == 9:
    #     break
