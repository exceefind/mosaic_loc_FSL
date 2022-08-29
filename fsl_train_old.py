import random

from model.conv4_old_version import ConvNet
import yaml
import easydict
import argparse
import os
import torch
import torch.nn as nn
from util.DataSet_FSL import MiniImageNetDataSet
from util.DataSet_mosaic import MiniImageNetDataSet_mosaic
import torchvision
import torchvision.transforms as T
import logging
import time

from util.Mosaic_fsl import getMosaic_pseudo
from util.loss import ce_loss

def train():
    Epoch_num = config.train.Epoch_num
    ce_loss_loc = torch.nn.CrossEntropyLoss()
    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(),lr=config.train.lr,weight_decay=float(config.train.weight_decay))
    elif config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train.lr,momentum=0.9,weight_decay=float(config.train.weight_decay))
    Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.markstone, gamma=config.train.decay_lr)
    best_dev = 0
    beta_moving = config.train.beta_moving
    Dev_Acc_moving = 0
    info = ""
    logging.info("Few-shot model start training...")
    for epoch in range(Epoch_num):
        net.train()
        correct_train,total_train = 0,0

        for step,(image,label,lable_loc) in enumerate(trainloader):
            image = image.to(device)
            label = label.to(device)
            label_loc = lable_loc.to(device)
            out,out_loc = net(image)
            loss_cls = ce_loss(out,label)
            loss_loc = ce_loss_loc(out_loc,label_loc)
            loss = config.train.alpha*loss_cls + config.train.beta * loss_loc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if len(label.size()) != 1:
                __, label = torch.max(label, 1)
            _,pred = torch.max(out,1)
            correct_train += (pred == label).sum().item()
            total_train += label.size(0)
            if step%100 == 0:
                info = "Epoch {} of {}  \t Step {} of {}: \t loss_train: {:.4f} \t loss_classification: {:.4f} \tloss_Location: {:.4f}".format(epoch,Epoch_num,step,len(trainloader),loss.item(),loss_cls.item(),loss_loc.item())
                logging.info(info)
                # print(info)
                # print("train_acc: {}".format(correct_train/total_train))
                # total_train = 0
                # correct_train = 0
        Schuler.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for step, (img, label) in enumerate(devloader):
                img = img.to(device)
                label = label.to(device)
                predict,predict_loc = net(img)
                _, pre = torch.max(predict, 1)
                correct += (pre == label).sum().item()
                total += label.size(0)
            dev_acc = correct / total
            Dev_Acc_moving = beta_moving * Dev_Acc_moving + (1-beta_moving) * dev_acc
            if  dev_acc> best_dev:
                best_dev = dev_acc
                save_path = "checkpoint/FSLModel_param_best_"+str(epoch)+".ckpt"
                # torch.save(net.module.state_dict(), "checkpoint/area_pred_param_{:.4f}.ckpt".format(best_dev))
                # torch.save(net.state_dict(), save_path)
                # logging.info("Epoch {} of {}:  \t  model_acc : {} \t  model_save: {}".format(epoch,config.train.Epoch_num,dev_acc,save_path))
        info = "Epoch {} of {}:  \t Dev_Acc : {:.4f}  \t Dev_Acc_moving: {:.4f} \t Best_Dev_Acc:{:.4f}".format(epoch, config.train.Epoch_num,
                                                                             dev_acc,Dev_Acc_moving,best_dev)
        logging.info(info)
    print(info)
    logging.info("Few-shot model start finish!")
    logging.info("-"*100+"experiment finish!"+"-" * 100)
def FSL_sample():
    # print("start sample novel class for Few-shot Learning ...")
    path = "workship/miniImageNet/spilt/Fewshot.txt"
    cls_dict = {}
    cls_name = open(path, "r").read().splitlines()
    # cls_keys = cls_name
    cls_keys = random.sample(cls_name, args.n_way)
    cls_keys = ['Rottweiler', 'mobilehome', 'numbfish', 'ostrich', 'crocodile']
    # cls_keys = ['firetruck', 'horn', 'butterfly', 'husky', 'dishrag']
    # cls_keys = ['butternutsquash', 'runningshoe', 'horn', 'piano', 'malamute']
    logging.info("Sample novel class ...")
    logging.info(cls_keys)
    # cls_keys = cls_name
    # print(cls_keys)
    i = 0
    for name in cls_keys:
        if name != "":
            cls_dict[i] = name
            i += 1
    print(cls_dict)
    return cls_dict

if __name__ == '__main__':

    date = time.strftime("%Y_%m_%d", time.localtime())
    logging.basicConfig(filename="log/FSL_MiniImageNet_"+date+".log",level=logging.INFO,format="%(levelname)s: %(asctime)s : %(message)s")
    parser = argparse.ArgumentParser(description='FSL  ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--gpu', type=int, nargs='+', default=None)
    # parser.add_argument("--model_continue",type=int,default=0)
    parser.add_argument("--n_way",type=int,default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--use_mosaic", type=bool, default=True)
    parser.add_argument("--sim_way",type=int,default=0)
    parser.add_argument("--dist_scale", type=float, default=1)
    parser.add_argument("--num_sim",type=int, default=3)
    parser.add_argument("--randomMosaic",type=bool, default=True)
    parser.add_argument("--mosaic_baseSize",type=float,default=0.5)
    # parser.add_argument("--mosaic_sizeBias", type=float, default=0)
    parser.add_argument("--mosaic_num", type=int, default=100)
    parser.add_argument("--Novel_Mosaic_rate", type=float, default=0.25)
    parser.add_argument("--Location",type=bool,default=True)
    parser.add_argument("--pretrained", type=bool, default=True)
    args = parser.parse_args()
    logging.info("-" * 100+"experiment start!"+"-" * 100)
    logging.info("FSL_learning: \t n_way: {}  \t k_shot: {}".format(args.n_way,args.k_shot))
    logging.info(args.__dict__)
    # logging.info("use_mosaic: " +("True" if args.use_mosaic else "False"))
    # if args.use_mosaic:
    #     logging.info("mosaic : num of similiar class = {}".format(args.num_sim))
    # logging.info("use_Location: " + ("True" if args.Location else "False"))
    # 抽样小样本类
    cls_fsl_dict = FSL_sample()
    # 根据小样本类构造相似类:
    getMosaic_pseudo(cls_fsl_dict, args.num_sim,args.sim_way,args.dist_scale)

    config = yaml.load(open("config/fsl_train.yaml"),Loader=yaml.FullLoader)
    config = easydict.EasyDict(config)

    logging.info("model parameters :  ")
    logging.info(config.train)

    gpu_device = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    device = torch.device('cuda')

    net = ConvNet(out="prob",Location=True, num_cls=5).to(device)
    # net_base = ConvNet(out="prob",num_cls=80)
    # # if args.model_continue == 1:
    state_dict = torch.load("checkpoint/BaseModel_param_best.ckpt")
    # net_base.load_state_dict(state_dict)
    # # net.add_module("fc",nn.Linear(1600,5))

    # 读取参数 预训练的参数和自己模型的参数
    pretrained_dict = state_dict
    model_dict = net.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉，通过严格对应层名获取对应参数
    dict_not_include = ["fc.weight","fc.bias","fc_loc.weight","fc_loc.bias"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if( k in model_dict and k not in dict_not_include)}
    # 更新现有、对等的model_dict
    model_dict.update(pretrained_dict)
    # 加载参数model.state_dict到自己的模型中
    if args.pretrained:
        net.load_state_dict(model_dict)

    img_size = 84

    transform_train = torchvision.transforms.Compose([
        T.Resize([84,84]),
        T.RandomCrop(args.mosaic_baseSize * 84),
        # T.RandomRotation(90),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])

    transform_orginal = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.RandomCrop( 84, padding=2),
        T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    logging.info(transform_train.__dict__)

    transform = torchvision.transforms.Compose([
        T.Resize([84,84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    DataSet_train = MiniImageNetDataSet_mosaic(root=config.data.root,mode="train_fsl",cls_dict=cls_fsl_dict,mosaic_novel_rate=args.Novel_Mosaic_rate,mosaic_cls= args.num_sim
                                               ,mosaic=True,mosaic_k=args.mosaic_num,Location=args.Location,transform=transform_train,transform_original=transform_orginal)
    DataSet_val = MiniImageNetDataSet_mosaic(root=config.data.root, mode="val_fsl",cls_dict=cls_fsl_dict
                                             , transform=transform)

    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=config.data.trainloader.batch_size, shuffle=True,
                                              num_workers=config.data.trainloader.worker)
    devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size, shuffle=True,
                                            num_workers=config.data.testloader.worker)
    train()







