import sys
from tqdm import tqdm
import torch

def train_one_epoch(model, loss_function, optimizer, data_loader, device, epoch):
    model.train()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        sample_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        pred_classes = torch.argmax(pred,dim=1)
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (len(data_loader)), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, loss_function, data_loader, device, epoch):
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        sample_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (len(data_loader)), accu_num.item() / sample_num