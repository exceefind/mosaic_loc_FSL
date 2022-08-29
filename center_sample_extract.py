import time

from model.conv4 import ConvNet
import yaml
import easydict
import argparse
import os
import torch
import torch.nn as nn
from util.DataSet import MiniImageNetDataSet
import torchvision
import torchvision.transforms as T
from model.resnet12 import resnet12
from model.resnet import *
import numpy as np
import warnings
from util.loss import prototypical_Loss
warnings.filterwarnings("ignore")

def train():
    Epoch_num = config.train.Epoch_num
    if args.no_train:
        Epoch_num = 1
    ce_loss = torch.nn.CrossEntropyLoss()
    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(),lr=config.train.lr)
    elif config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train.lr,momentum=0.9,weight_decay=5e-4)
    Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.markstone, gamma=0.4)
    best_dev = 0
    dataset = args.yaml if args.yaml else "mini_imagenet"
    prototype = torch.zeros((config.train.num_classes, net.fc.in_features),requires_grad=False)
    # sample_num = np.zeros(config.train.num_classes)
    for epoch in range(Epoch_num):
        tic = time.time()
        net.train()
        if args.no_train:
            net.eval()
            torch.no_grad()
        correct_train,total_train = 0,0
        cls_prototype = torch.zeros((config.train.num_classes, net.fc.in_features))
        sample_num = np.zeros(config.train.num_classes)
        feat_list = []
        label_list = []
        j = 0
        dict_path = {}
        for step,(img_path,image,label) in enumerate(trainloader):
            # 记录下图片地址:
            for path in img_path:
                dict_path[j] = path
                j += 1
            image = image.to(device)
            label = label.to(device)
            # print(step)
            feat,out = net(image,is_feat=True)
            feat_list.append(feat.detach().cpu())
            label_list.append(label.detach().cpu())
            if args.no_train is False:
                loss = ce_loss(out,label)
                if args.center_loss:
                    loss_center,prototype = prototypical_Loss(feat,label,prototype,epoch,center=True)
                    loss += 0.5 * loss_center
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _,pred = torch.max(out,1)
                correct_train += (pred == label).sum().item()
                total_train += label.size(0)
                if step%100 == 0 and step != 0:
                    print("Epoch {} of {}  Step {} of {}: loss_train: {:.4f}  acc_train: {:.4f}".format(epoch,Epoch_num,step,len(trainloader),loss.item(),correct_train/total_train))
                    total_train = 0
                    correct_train = 0

            for i in np.unique(label.cpu().data):
                lab_list = np.where(label.detach().cpu().data == i)[0]
                sample_num[i] += lab_list.size
                cls_prototype[i, :] += sum(feat.detach().cpu()[lab_list, :], 0)

        # Schuler.step()
        # net.eval()
        correct = total = 0
        feat_tensor = torch.cat(feat_list,dim=0).reshape(-1,net.fc.in_features)
        label_tensor = torch.cat(label_list,dim=0)
        center_sample = []
        for l in range(config.train.num_classes):
            cls_prototype[l, :] /= sample_num[l]
            # print(feat_tensor[label_tensor==l].shape)
            dist = torch.sum(torch.pow(feat_tensor[label_tensor==l]-cls_prototype[l, :],2),1)
            value,ind = torch.sort(dist)
            label_index = torch.where(label_tensor==l)[0]
            # print(ind[:10])
            glo_ind = label_index[ind[:100]]
            center_sample.append(glo_ind)

        center_sample = torch.cat(center_sample,dim=0).reshape(-1,center_sample[0].shape[0])
        np.save("sample/center_sample.npy",center_sample.numpy())
        np.save("sample/dict_path.npy",dict_path)
        # print(center_sample.shape)

        # with torch.no_grad():
        #     for step, (img_path,img, label) in enumerate(trainloader):
        #         img = img.to(device)
        #         label = label.to(device)
        #         feat,predict = net(img,is_feat=True)
        #         _, pre = torch.max(predict, 1)
        #         correct += (pre == label).sum().item()
        #         total += label.size(0)
        #     if epoch%10 ==0:
        #         torch.save(net.state_dict(), "checkpoint_new/BaseModel_" + args.net + "_" + dataset + "_mosaic_last.ckpt")
        #     if correct / total > best_dev:
        #
        #         for i in range(config.train.num_classes):
        #             cls_prototype[i, :] /= sample_num[i]
        #         if args.mosaic:
        #             np.save("prototype/Base_" + args.net + "_" + dataset + "_mosaic_0602.npy", cls_prototype.numpy())
        #         else:
        #             if args.clean:
        #                 np.save("prototype/Base_" + args.net + "_" + dataset + "_pt_clean.npy", cls_prototype.numpy())
        #             else:
        #                 np.save("prototype/Base_" + args.net + "_" + dataset + "_pt_proto_color.npy", cls_prototype.numpy())
        #         best_dev = correct / total
        #         # torch.save(net.module.state_dict(), "checkpoint/area_pred_param_{:.4f}.ckpt".format(best_dev))
        #         if args.no_train is False:
        #             if args.mosaic:
        #                 torch.save(net.state_dict(), "checkpoint_new/BaseModel_" + args.net + "_"+dataset+"_mosaic.ckpt")
        #             else:
        #                 if args.clean:
        #                     torch.save(net.state_dict(),
        #                                "checkpoint_new/BaseModel_" + args.net + "_" + dataset + "_best_clean.ckpt")
        #                 else:
        #                     torch.save(net.state_dict(), "checkpoint_new/BaseModel_"+args.net+ "_"+dataset+"_proto.ckpt")
        # print(
        #     "Epoch {} of {}:   Dev_Acc : {:.4f}  Best_Dev_Acc:{:.4f}".format(epoch, config.train.Epoch_num,
        #                                                                      correct / total,
        #                                                                      best_dev))
        # toc = time.time()
        # print("elapse:  {:.2f} min".format((toc-tic)/60))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Base stage: pretrain model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument("--mosaic", action="store_true", default=False)
    parser.add_argument("--net", default="conv64", type=str)
    parser.add_argument("--model_continue",type=int,default=0)
    parser.add_argument("--yaml", type=str, default=None)
    parser.add_argument("--no_train", default=False, action="store_true")
    parser.add_argument("--center_loss", default=False, action="store_true")
    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument('--baseline', default=False, action='store_true')
    args = parser.parse_args()
    dataset = args.yaml if args.yaml else "mini_imagenet"
    config = yaml.load(open("config/Base_train_new.yaml"),Loader=yaml.FullLoader)
    if args.yaml:
        # print(args.yaml)
        config = yaml.load(open("config/Base_train_"+args.yaml+".yaml"), Loader=yaml.FullLoader)
    config = easydict.EasyDict(config)
    # print(config.train)

    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    device = torch.device('cuda')
    if args.net == "conv64":
        net = ConvNet(num_classes=config.train.num_classes).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint_new/BaseModel_param_0.6.ckpt")

            net.load_state_dict(state_dict)
    elif args.net == "resnet12":
        net = resnet12(num_classes=config.train.num_classes,baseline = args.baseline).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint_new/BaseModel_resnet12_mini_imagenet_best_clean.ckpt")
            net.load_state_dict(state_dict)
    elif args.net == "resnet18":
        net = resnet18(num_classes=config.train.num_classes).to(device)
        if args.model_continue == 1:
            state_dict = torch.load("checkpoint_new/BaseModel_resnet18_mosaic.ckpt")
            net.load_state_dict(state_dict)
    # print(net)
    transform_train = torchvision.transforms.Compose([
        T.Resize([84,84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transform_mosaic = torchvision.transforms.Compose([
        T.Resize([84, 84]),
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

    DataSet_train = MiniImageNetDataSet(root=config.data.root,mode="train_train",mosaic=args.mosaic,mosaic_k=1,
                                        transform=transform_train,transform_mosaic=transform_mosaic,is_returnName=True)
    DataSet_val = MiniImageNetDataSet(root=config.data.root, mode="train_test",mosaic=False, transform=transform)

    # print(len(DataSet_train))
    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=config.data.trainloader.batch_size, shuffle=False,
                                              num_workers=config.data.trainloader.worker)
    devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size, shuffle=False,
                                            num_workers=config.data.testloader.worker)
    train()







