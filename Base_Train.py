from model.conv4 import ConvNet
import yaml
import easydict
import argparse
import os
import torch
import torch.nn as nn
from util.DataSet_FSL import MiniImageNetDataSet
import torchvision
import torchvision.transforms as T
from model.resnet12 import resnet12
from model.resnet import *
import warnings
warnings.filterwarnings("ignore")

def train():
    Epoch_num = config.train.Epoch_num
    ce_loss = torch.nn.CrossEntropyLoss()
    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(),lr=config.train.lr)
    elif config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train.lr,momentum=0.9,weight_decay=5e-4)
    Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.markstone, gamma=0.4)
    best_dev = 0

    for epoch in range(Epoch_num):
        net.train()
        correct_train,total_train = 0,0
        for step,(image,label) in enumerate(trainloader):
            image = image.to(device)
            label = label.to(device)
            out = net(image)
            loss = ce_loss(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,pred = torch.max(out,1)
            correct_train += (pred == label).sum().item()
            total_train += label.size(0)
            if step%100 == 0:
                print("Epoch {} of {}  Step {} of {}: loss_train: {:.4f}  acc_train: {:.4f}".format(epoch,Epoch_num,step,len(trainloader),loss.item(),correct_train/total_train))
                total_train = 0
                correct_train = 0

        Schuler.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for step, (img, label) in enumerate(devloader):
                img = img.to(device)
                label = label.to(device)
                predict = net(img)
                _, pre = torch.max(predict, 1)
                correct += (pre == label).sum().item()
                total += label.size(0)
            if correct / total > best_dev:
                best_dev = correct / total
                # torch.save(net.module.state_dict(), "checkpoint/area_pred_param_{:.4f}.ckpt".format(best_dev))
                if args.mosaic:
                    torch.save(net.state_dict(), "checkpoint/BaseModel_" + args.net + "_mosaic.ckpt")
                else:
                    torch.save(net.state_dict(), "checkpoint/BaseModel_"+args.net+"_best2.ckpt")
        print(
            "Epoch {} of {}:   Dev_Acc : {:.4f}  Best_Dev_Acc:{:.4f}".format(epoch, config.train.Epoch_num,
                                                                             correct / total,
                                                                             best_dev))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Msi area predict ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, default=None)
    # parser.add_argument('--configs', type=str, default='0.001', help='learn_rate')
    # parser.add_argument("--local_rank", default=-1, type=int)
    # parser.add_argument("--isDDP", default=1, type=int)
    parser.add_argument("--mosaic", action="store_true", default=False)
    parser.add_argument("--net", default="conv64", type=str)
    parser.add_argument("--model_continue",type=int,default=0)
    args = parser.parse_args()

    config = yaml.load(open("config/Base_train.yaml"),Loader=yaml.FullLoader)
    config = easydict.EasyDict(config)

    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    device = torch.device('cuda')
    if args.net == "conv64":
        net = ConvNet(num_classes=64).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint/BaseModel_param_0.6.ckpt")
            net.load_state_dict(state_dict)
    elif args.net == "resnet12":
        net = resnet12(num_classes=64).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint/BaseModel_resnet12_mosaic.ckpt")
            net.load_state_dict(state_dict)
    elif args.net == "resnet18":
        net = resnet18(num_classes=64).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint/BaseModel_resnet18_mosaic.ckpt")
            net.load_state_dict(state_dict)
    # print(net)
    transform_train = torchvision.transforms.Compose([
        T.Resize([84,84]),
        T.RandomCrop(84, padding=2),
        T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transform_mosaic = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.RandomCrop(42, padding=1),
        # T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        T.Resize([84,84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    DataSet_train = MiniImageNetDataSet(root=config.data.root,mode="train_Base",mosaic=args.mosaic,transform=transform_train,transform_mosaic=transform_mosaic)
    DataSet_val = MiniImageNetDataSet(root=config.data.root, mode="val_Base",mosaic=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=config.data.trainloader.batch_size, shuffle=True,
                                              num_workers=config.data.trainloader.worker)
    devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size, shuffle=True,
                                            num_workers=config.data.testloader.worker)
    train()







