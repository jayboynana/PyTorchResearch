import os
import math
import argparse
from collections import defaultdict
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from myutils import read_split_data
from mydataset import Mydataset
from model.vit_model import vit_base_patch16_224_in21k
# from ViT import vit_base_patch16_224_in21k
from engine import train_one_epoch,evaluate

def get_args_parser():

    parser = argparse.ArgumentParser(description='PyTorch Classification Training',add_help=True)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data',type=str,default='E:/PyTorchProject/PyTorch Tutorial/Classification/flower_photos',help='path to dataset')
    parser.add_argument('--model_name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='.checkpoints/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device',type=str,default='cuda:0',help='gpu device')

    return parser.parse_args()

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    train_dataset = Mydataset(train_images_path, train_images_label,transform=data_transform['train'])
    val_dataset = Mydataset(val_images_path,val_images_label,transform=data_transform['val'])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = vit_base_patch16_224_in21k(num_classes=args.num_classes,has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    loss_function = torch.nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    info = defaultdict(list)
    for i in ['train_loss','train_acc','val_loss','val_acc']:
        info[i] = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                loss_function = loss_function,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        val_loss, val_acc = evaluate(model=model,
                                     loss_function = loss_function,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        info['train_loss'].append(round(train_loss,2))
        info['train_acc'].append(round(train_acc,2))
        info['val_loss'].append(round(val_loss,2))
        info['val_acc'].append(round(val_acc,2))

        print(info)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    args = get_args_parser()
    main(args)