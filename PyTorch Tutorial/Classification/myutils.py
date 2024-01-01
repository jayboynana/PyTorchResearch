import json
import os
import random
from tqdm import tqdm

def read_split_data(root,val_rate = 0.2):

    random.seed(0)
    assert os.path.exists(root), f'{root} does not exist!'

    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    classes.sort()

    classes_indices = dict((k,v) for v,k in enumerate(classes))
    indices_classes = dict((k,v) for v,k in classes_indices.items())

    json_classes = json.dumps(indices_classes,indent=4)
    with open('indices2classes.json','w') as json_file:
        json_file.write(json_classes)

    train_images_path = []
    train_images_label = []
    val_iamges_path = []
    val_iamges_label = []
    every_class_num = []
    supported = ['.JPG','.jpg','.PNG','.png']

    for cla in tqdm(classes,desc='it is splitting!'):
        class_path = os.path.join(root,cla)
        images = [os.path.join(class_path,i) for i in os.listdir(class_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = classes_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images,k= int(len(images)*val_rate))

        for img in images:
            if img in val_path:
                val_iamges_path.append(img)
                val_iamges_label.append(image_class)
            else:
                train_images_path.append(img)
                train_images_label.append(image_class)

    print(f'there are {sum(every_class_num)} images in this dataset')
    print(f'{len(train_images_path)} images for training')
    print(f'{len(val_iamges_path)} images for validation')

    return train_images_path,train_images_label,val_iamges_path,val_iamges_label



if __name__ == "__main__":
    flowers_path = 'E:/jupyterlab/PyTorch Tutorial/Classification/flower_photos'
    train_images_path,train_images_label,val_images_path,val_images_label = read_split_data(flowers_path)








