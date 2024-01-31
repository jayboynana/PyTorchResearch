import torch
from PIL import Image
from torch.utils.data import Dataset

class Mydataset(Dataset):
    """
    Custom own dataset
    """
    def __init__(self,image_path,class_path,transform = None):
        self.image_path = image_path
        self.class_path = class_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = Image.open(self.image_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.image_path[item]} is not mode of RGB!")
        if self.transform is not None:
            img = self.transform(img)

        label = self.class_path[item]

        return img,label

    @staticmethod
    def collate_fn(batch):
        images,labels = tuple(zip(*batch))
        images = torch.stack(images,dim=0)
        labels = torch.as_tensor(labels)
        return images,labels


