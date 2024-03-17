import os
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

coco_json_path = './annotations_trainval2017/annotations/instances_val2017.json'
coco_img_path = './val2017/val2017'

coco = COCO(annotation_file=coco_json_path)
coco_img_keys = coco.imgs.keys()
ids = list(sorted(coco.imgs.keys()))
print(f'number of images in coco_val_2017 is {len(ids)}')

coco_cats_values = coco.cats.values()
coco_labels = dict([(v['id'],v['name']) for v in coco.cats.values()])
print(coco_labels)

for img_id in ids[:3]:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)

    img_name = coco.loadImgs(img_id)[0]['file_name']
    img = Image.open(os.path.join(coco_img_path,img_name)).convert('RGB')
    draw = ImageDraw.Draw(img)

    for target in targets:
        x,y,w,h = target['bbox']
        x1,y1,x2,y2 = x,y,int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        draw.text((x1,y1),coco_labels[target['category_id']])

    plt.imshow(img)
    plt.show()




