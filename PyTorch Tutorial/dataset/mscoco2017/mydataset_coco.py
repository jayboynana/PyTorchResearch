import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

class coco_dataset(Dataset):
    """
    custom coco dataset
    """
    def __init__(self,root,transforms = None,train = True):
        self.root = root
        self.transforms = transforms
        self.annotations = os.path.join(root,'annotations_trainval2017/annotations')
        if train:
            self.coco_images = os.path.join(root,'train2017')
            self.coco_annotations = os.path.join(self.annotations,'instances_train2017.json')
        else:
            self.coco_images = os.path.join(root, 'val2017')
            self.coco_annotations = os.path.join(self.annotations,'instances_val2017.json')
        assert os.path.exists(self.coco_annotations),f'{self.coco_annotations} is not exist!'

        self.coco = COCO(self.coco_annotations)

    def __len__(self):
        return len(self.coco.imgs.keys())

    def __getitem__(self, idx):
        # id2image = dict([(i,j) for i,j in enumerate(list(self.coco.imgs.keys()))])
        # img_id = id2image[idx]
        img_id = list(self.coco.imgs.keys())[idx]
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        targets = self.coco.loadAnns(ann_ids)

        image = read_image(os.path.join(self.coco_images,img_name))
        boxes = [t['bbox'] for t in targets]
        labels = [t['category_id'] for t in targets]
        iscrowd = [t['iscrowd'] for t in targets]
        area = [t['area'] for t in targets]

        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels = torch.as_tensor(labels,dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd,dtype=torch.int64)
        area = torch.as_tensor(area,dtype=torch.float32)
        image_id = torch.tensor([idx])

        target = {}
        target['image_id'] = image_id
        target['original_id'] = img_id
        target['boxes'] = boxes
        target['labels'] = labels
        target['iscrowd'] = iscrowd
        target['area'] = area

        if self.transforms is not None:
            image,target = self.transforms(image,target)

        return image,target

def draw_boxes(image,boxes,labels):
    boxes_convert = box_convert(boxes,in_fmt='xywh',out_fmt='xyxy')
    draw_boxes_image = draw_bounding_boxes(image,boxes_convert,labels=labels)
    return draw_boxes_image

def show(image,labels,cats_dict):
    draw_labels = []
    for label in labels:
        for v in cats_dict:
            if v['id'] == label:
                draw_labels.append(v['name'])
    print(draw_labels)
    plt.figure(figsize=(10,6))
    draw_boxes_image = draw_boxes(image, target['boxes'], draw_labels)
    draw_boxes_image = draw_boxes_image.detach()
    draw_boxes_image = F.to_pil_image(draw_boxes_image)
    plt.imshow(draw_boxes_image)
    plt.show()

if __name__ == "__main__":
    root = 'E:/pythonProject/PyTorch Tutorial/dataset/mscoco2017'
    coco_dataset_val = coco_dataset(root=root,train=False)
    image,target = coco_dataset_val[1000]
    # print(image)
    for item in target.items():
        print(item)

    cats_dict = coco_dataset_val.coco.cats.values()
    show(image,target['labels'],cats_dict)
    # show(draw_boxes_image)
    # coco = COCO(os.path.join(root,'annotations_trainval2017/annotations/instances_val2017.json'))
    # ann_id = coco.getAnnIds(imgIds='37777')
    # coco.showAnns(ann_id,draw_bbox=True)