import csv
import math
import random

import numpy.random
from tqdm import tqdm

from model.conv4 import ConvNet
from model.resnet12 import resnet12
from model.resnet import *
import yaml
import easydict
import argparse
import os
import torch
import torch.nn as nn
from util.DataSet import MiniImageNetDataSet
from util.DataSet_mosaic import MiniImageNetDataSet_mosaic
import torchvision
import torchvision.transforms as T
import logging
import time
import torch.nn.functional as F
import numpy as np
from collections import Counter

from util.Mosaic_fsl import getMosaic_pseudo,Mosaic_cls
from util.loss import ce_loss,prototypical_Loss,euclidean_dist,loc_loss,Few_loss
import warnings

warnings.filterwarnings("ignore")

# random.seed(20220518)
random.seed(20220520)

def train():
    tic = time.time()
    train_param = net.parameters()
    # train_param = net.fc.parameters()
    # print(net.fc.state_dict())
    backbone_param = filter(lambda x:id(x) not in list(map(id,net.fc.parameters())) +list(map(id,net.fc_loc.parameters())),net.parameters())
    head = filter(lambda x:id(x) not in list(map(id,backbone_param)),net.parameters())
    params = [{'params':head,'lr':args.lr_base},
             {'params':backbone_param,'lr':args.lr_base*config.train.scale_backbone}]

    if args.train_head:
        params = head
    Epoch_num = config.train.Epoch_num if args.Epoch is None else args.Epoch
    ce_loss_loc = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()
    if config.train.optimizer == "Adam":
        # optimizer = torch.optim.Adam(train_param,lr=args.lr_base,weight_decay=float(config.train.weight_decay))
        optimizer = torch.optim.Adam(params, weight_decay=float(config.train.weight_decay))
    elif config.train.optimizer == "SGD":
        # optimizer = torch.optim.SGD(train_param, lr=args.lr_base,momentum=0.9,weight_decay=float(config.train.weight_decay))
        optimizer = torch.optim.SGD(params,lr=args.lr_base ,momentum=config.train.momentum,
                                    weight_decay=float(config.train.weight_decay))

    # Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.markstone, gamma=config.train.decay_lr)
    Schuler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=int(Epoch_num/4),T_mult=3)
    best_dev = 0
    beta_moving = config.train.beta_moving
    Dev_Acc_moving = 0
    info = ""
    logging.info("Few-shot model start training...")
    sum_acc = 0
    prototypes = torch.zeros((args.n_way, feat_dim),requires_grad=False).to(device)
    for epoch in range(Epoch_num):
        # net.train()
        correct_train,total_train = 0,0
        for step,(image,mosaic_img,label,lable_loc) in enumerate(trainloader):
            # net.train()
            image = image.to(device)
            mosaic_img = mosaic_img.to(device)
            label = label.to(device)
            label_loc = lable_loc.to(device)
            if args.proto_loss:
                feat ,out = net(image,is_feat=True)
                _, out_loc = net(mosaic_img, feat, is_loc=True)
                loss_cls,prototypes_update = prototypical_Loss(feat,label,prototypes,epoch,temperature=config.train.temperature)
                # print(prototypes)
                prototypes = prototypes_update.to(device)
            else:
                # print(torch.any(torch.isnan(image)))
                feat ,out = net(image,is_feat=True)
                _,out_loc = net(mosaic_img,feat,is_loc= True)
                # print(out)
                if args.Few_loss:
                    loss_cls = Few_loss(out,label)
                else:
                    loss_cls = ce_loss(out,label,temperature=config.train.temperature)
                # loss_center, prototypes = prototypical_Loss(feat, label, prototypes, epoch)
                # print(prototypes)
                # prototypes = prototypes.to(device)
                # loss_cls += 0*loss_center
            if args.boundingbox:
                loss_loc = mse_criterion(out_loc,label_loc)
            else:
                # print(label_loc)
                # print(out_loc)
                # sigmoid 之后，在进行ce loss 会导致全1，完全没有任何优化
                # loss_loc = loc_loss(out_loc,label_loc)
                #  -------1--------
                # out_loc = F.sigmoid(out_loc)
                # loss_loc = mse_criterion(out_loc, label_loc)
                # label_loc =label_loc.long()
                loss_loc = loc_loss(out_loc,label_loc)
                # print(loss_loc)
            loss = config.train.alpha*loss_cls + args.beta * loss_loc
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if len(label.size()) != 1:
                __, label = torch.max(label, 1)
            _,pred = torch.max(out,1)
            correct_train += (pred == label).sum().item()
            total_train += label.size(0)
            if step%50 == 0 :
                info = "Epoch {} of {} :  \t  Step {} of {}: \t loss_train: {:.4f} \t loss_classification: {:.4f} \tloss_Location: {:.4f}".format(epoch,Epoch_num,step,len(trainloader),loss.item(),loss_cls.item(),loss_loc.item())
                logging.info(info)
                # Schuler.step()
                # net.eval()
                # correct = total = 0
                # with torch.no_grad():
                #     for step, (img, label) in enumerate(devloader):
                #         img = img.to(device)
                #         label = label.to(device)
                #         if args.proto_loss:
                #             feat_predict, _, predict_loc = net(img, is_feat=True)
                #             predict = F.log_softmax(-euclidean_dist(feat_predict, prototypes), dim=1)
                #         else:
                #             predict, predict_loc = net(img)
                #         _, pre = torch.max(predict, 1)
                #         correct += (pre == label).sum().item()
                #         total += label.size(0)
                #     dev_acc = correct / total
                #     sum_acc += dev_acc
                #     Dev_Acc_moving = beta_moving * Dev_Acc_moving + (1 - beta_moving) * dev_acc
                #     if dev_acc > best_dev:
                #         best_dev = dev_acc
                # info = "Epoch {} of {}:  \t Dev_Acc : {:.4f}  \t Dev_Acc_moving: {:.4f} \t Best_Dev_Acc:{:.4f}".format(
                #     epoch, Epoch_num,
                #     dev_acc, Dev_Acc_moving, best_dev)
                # print(info)
                # print(info)
                # print("train_acc: {}".format(correct_train/total_train))
                total_train = 0
                correct_train = 0
        Schuler.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for step, (img, label) in enumerate(devloader):
                img = img.to(device)
                label = label.to(device)

                if args.proto_loss:
                    feat_predict,_ = net(img,is_feat=True)
                    # _, out_loc = net(img, feat, is_loc=True)
                    predict = F.log_softmax(-euclidean_dist(feat_predict,prototypes),dim=1)
                else:
                    predict= net(img)
                _, pre = torch.max(predict, 1)
                # print(pre)
                # print(pre)
                correct += (pre == label).sum().item()
                total += label.size(0)
            dev_acc = correct / total
            sum_acc += dev_acc
            Dev_Acc_moving = beta_moving * Dev_Acc_moving + (1-beta_moving) * dev_acc
            if  dev_acc> best_dev:
                best_dev = dev_acc
                # save_path = "checkpoint/FSLModel_param_best_"+str(epoch)+".ckpt"
                # torch.save(net.module.state_dict(), "checkpoint/area_pred_param_{:.4f}.ckpt".format(best_dev))
                # torch.save(net.state_dict(), save_path)
                # logging.info("Epoch {} of {}:  \t  model_acc : {} \t  model_save: {}".format(epoch,config.train.Epoch_num,dev_acc,save_path))
        info = "Epoch {} of {}:  \t Dev_Acc : {:.4f}  \t Best_Dev_Acc:{:.4f}".format(epoch, Epoch_num,
                                                                          dev_acc,best_dev)
        if epoch%10 == 0:
            print(info)
        logging.info(info)

    file = open("record/_"+str(args.id)+'_'+str(args.k_shot)+"_"+str(args.net)+"_"+".txt","a+")
    # file.write(str(test_acc)+"\t"+str(dev_acc)+"\n")
    file.write(str(dev_acc) + "\n")
    file.close()
    toc = time.time()
    acc_fsl.append(dev_acc)
    print("Dev_Acc:  {:.4f} \t  elapse:  {:.2f} min".format(dev_acc,(toc-tic)/60))
    logging.info("Few-shot model start finish!")
    logging.info("-"*100+"experiment finish!"+"-" * 100)

def FSL_sample():
    # 首先: 统计小样本类的总个数、各类下的样本个数
    fsl_stage = "val" if args.fsl_val else "test"
    if args.fsl_val:
        path = config.data.root + "/val.csv"
    else:
        path = config.data.root + "/test.csv"
    images_list,label_list = [] , []
    label_dict = {}
    num_dict = []
    ind = 0
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            image = config.data.root + "/images/" + r[0]
            if r[1] not in label_dict.keys():
                label_dict[r[1]] = ind
                num_dict.append(0)
                ind += 1
            num_dict[label_dict[r[1]]] += 1
            label_list.append(r[1])
            images_list.append(image)
    # print(Counter(num_dict)[600])
    # assert Counter(num_dict)[600]==16
    # 其次：抽样出n way、k shot、q shot的场景
    cls_num =  len(set(label_list))
    support_lsit = []
    query_list = []
    fsl_cls = random.sample(range(cls_num),args.n_way)
    logging.info('Sample FSL classes:   ')
    # fsl_cls = [17, 0, 11, 4, 2]
    # fsl_cls = [5, 1, 14, 3, 0]
    # fsl_cls = [8, 4, 10, 12, 13]
    # fsl_cls = [6, 9, 14, 5, 13]
    # fsl_cls = [5, 12, 0, 10, 1]

    # hard: 0.66
    # fsl_cls = [3 ,0, 2, 13, 14]
    # mid: 0.74
    # fsl_cls = [3, 9, 12, 6, 1]
    # easy: 0.81
    # fsl_cls = [11, 10, 1, 4, 6]
    # fsl_cls = [10, 6, 2, 12, 3]
    # fsl_cls = [3, 4, 1, 8, 6]
    print(fsl_cls)
    logging.info(fsl_cls)
    # new_dict = {v : k for k,v in label_dict.items()}
    # for c in fsl_cls:
    #     print(new_dict[c])
    # print(label_list)
    for cls_id in fsl_cls:
        start = sum(num_dict[:cls_id])
        nspc = num_dict[cls_id]
        fsl_sample = random.sample(range(start, start+nspc-1),args.k_shot+args.q_shot)

        # count ,sum_ = 0,0
        # for index in fsl_sample:
        #     lab = label_list[index]
        #     sum_ += label_dict[lab]
        #     count += 1
        # print(sum_/len(fsl_sample))

        fsl_sup = fsl_sample[:args.k_shot]
        # print(images_list[fsl_sample[0]])
        # print(len(fsl_sup))
        fsl_query = fsl_sample[args.k_shot:]
        # print(len(fsl_query))
        support_lsit.extend(fsl_sup)
        query_list.extend(fsl_query)

    # 查看fsl是否采样出错
    # count =0
    # sum_ = 0
    # for index in query_list:
    #     lab = label_list[index]
    #     sum_ += label_dict[lab]
    #     if count==99:
    #         count = 0
    #         print(sum_/100)
    #         sum_ = 0
    #     else:
    #         count += 1
    # 然后：指定样本index 构成FSL support 和query数据集


    transform_mosaic = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),
        # T.RandomCrop(args.mosaic_baseSize * img_size,padding=2),
        # T.RandomHorizontalFlip(),
        # T.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
        # # T.RandomRotation(45),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform_train = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),
        # T.RandomCrop(img_size, padding=2),
        # T.RandomHorizontalFlip(),
        # T.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
        # # T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # old setting
    # transform_mosaic = torchvision.transforms.Compose([
    #     T.Resize([img_size, img_size]),
    #     T.RandomCrop(args.mosaic_baseSize * img_size, padding=1),
    #     # T.RandomRotation(90),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    # ])
    # transform_train = torchvision.transforms.Compose([
    #     T.Resize([img_size, img_size]),
    #     T.RandomCrop(img_size, padding=2),
    #     T.RandomRotation(90),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    # ])



    # logging.info(transform_train.__dict__)

    transform = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    DataSet_train = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl", n_way=args.n_way,sample_idx=support_lsit,
                                               mosaic_novel_rate=args.Novel_Mosaic_rate, mosaic_cls=args.num_sim
                                               , mosaic=False, mosaic_k=args.mosaic_num,
                                               Location=False,transform=transform_train,transform_mosaic=transform_mosaic,
                                               mosaic_origin_num=args.mosaic_ori_num, boundingbox=args.boundingbox)
    DataSet_val = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl", sample_idx=query_list,
                                             transform=transform)

    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=config.data.trainloader.batch_size,
                                              num_workers=config.data.trainloader.worker)
    devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size,
                                            num_workers=config.data.testloader.worker)
    # mosaic_nk, pseudo_nk = Mosaic_cls(net, dataloader=trainloader,num_sim=args.num_sim,sim_way=args.sim_way,sim_mix=args.sim_mix,
    #                                   dist_scale=args.dist_scale,num_classes=args.n_way,path_dict_proto="prototype/Base_"+args.net+"_"+args.dataset+"_mosaic_0602.npy")
    # mosaic_nk, pseudo_nk = Mosaic_cls(net, dataloader=trainloader, num_sim=args.num_sim, sim_way=args.sim_way,
    #                                   sim_mix=args.sim_mix,
    #                                   dist_scale=args.dist_scale, num_classes=args.n_way,
    #                                   path_dict_proto="prototype/Base_" + args.net + "_" + args.dataset + "_pt_proto_color.npy")

    DataSet_mosaic = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl",n_way=args.n_way, sample_idx=support_lsit,
                                        mosaic_novel_rate=args.Novel_Mosaic_rate, mosaic_cls=args.num_sim
                                        , mosaic=True, mosaic_k=args.mosaic_num,mosaic_nk=None, pseudo_nk=None,
                                        Location=args.Location,transform=transform_train,transform_mosaic=transform_mosaic,
                                        mosaic_origin_num=args.mosaic_ori_num, boundingbox=args.boundingbox,size_bias=args.size_bias,
                                         grid=args.grid,fsl_label=fsl_cls,random_mosaic=args.random_mosaic)

    mosaic_loader = torch.utils.data.DataLoader(DataSet_mosaic, batch_size=config.data.testloader.batch_size, shuffle=True,
                                            num_workers=config.data.testloader.worker)
    return mosaic_loader,devloader

def load_model(net):
    reset(net)
    state_dict = torch.load(dict_path)
    # 读取参数 预训练的参数和自己模型的参数
    pretrained_dict = state_dict
    # pretrained_dict = torch.load('archive/data.pkl')

    model_dict = net.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉，通过严格对应层名获取对应参数
    dict_not_include = ["fc.weight","fc.weight_g","fc.weight_v", "fc.bias", "fc_loc.weight", "fc_loc.bias", "classifier.weight", "classifier.bias"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in dict_not_include)}
    # 更新现有、对等的model_dict
    model_dict.update(pretrained_dict)
    # 加载参数model.state_dict到自己的模型中
    if args.no_pretrained is False:
        net.load_state_dict(model_dict)

    return net

def reset(net):
    for m in net.modules():
        if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,a=math.sqrt(5))
            if m.bias is not None:
                fan_in,_ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound  = 1 /math.sqrt(fan_in)
                nn.init.uniform_(m.bias,-bound,bound)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSL  ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument("--fsl_val",default=False,action='store_true')
    parser.add_argument("--n_way",type=int,default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_shot", type=int, default=100)
    parser.add_argument("--num_sim",type=int, default=3)
    parser.add_argument("--net", default="conv64", type=str)
    parser.add_argument("--not_use_mosaic", action='store_false', default=True)
    parser.add_argument("--sim_way",type=int,default=0)
    parser.add_argument("--dist_scale", type=float, default=1)
    # 添加bounding box的目标定位  辅助任务
    parser.add_argument("--boundingbox", action="store_true", default=False)
    parser.add_argument("--sim_mix", action="store_true", default=False)
    parser.add_argument("--randomMosaic",type=bool, default=True)
    parser.add_argument("--mosaic_baseSize",type=float,default=0.5)
    parser.add_argument("--size_bias", type=float, default=0,help="mosaic_baseSize")
    parser.add_argument("--mosaic_num", type=int, default=100)
    parser.add_argument("--Novel_Mosaic_rate", type=float, default=0.25)
    parser.add_argument("--Location",action='store_false',default=True)
    parser.add_argument("--no_pretrained", action='store_true', default=False)
    parser.add_argument("--id",default=None,type=int)
    parser.add_argument("--Epoch", default=None, type=int)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--lr_base", default=0.002, type=float)
    parser.add_argument("--train_head", action='store_true', default=False)
    parser.add_argument("--proto_loss",action='store_true',default=False,)
    parser.add_argument("--have_Original", action='store_false', default=True)
    parser.add_argument("--mosaic_ori_num",default=0,type=int)
    parser.add_argument("--idx_task", default=None, type=int)
    parser.add_argument("--num_task", default=1, type=int)
    parser.add_argument("--dataset",default="mini_imagenet",type=str)
    parser.add_argument("--Few_loss", default=False, action="store_true")
    parser.add_argument('--grid',default=2,type=int)
    parser.add_argument('--baseline', default=False, action='store_true')
    parser.add_argument("--random_mosaic",default=False,action='store_true')
    parser.add_argument("--yaml",default=None,type=str)
    args = parser.parse_args()
    img_size = 84
    date = time.strftime("%Y_%m_%d", time.localtime())
    if args.id is not None:
        logging.basicConfig(filename="log/FSL_MiniImageNet_"+date+"_"+str(args.id)+".log", level=logging.INFO,
                            format="%(levelname)s: %(asctime)s : %(message)s")
    else:
        logging.basicConfig(filename="log/FSL_MiniImageNet_" + date + ".log", level=logging.INFO,
                        format="%(levelname)s: %(asctime)s : %(message)s")

    logging.info("-" * 100+"experiment start!"+"-" * 100)
    logging.info("FSL_learning: \t n_way: {}  \t k_shot: {}".format(args.n_way,args.k_shot))
    logging.info(args.__dict__)

    if args.yaml == None:
        config = yaml.load(open("config/fsl_train_new.yaml"), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open("config/"+args.yaml), Loader=yaml.FullLoader)
    config = easydict.EasyDict(config)

    logging.info("model parameters :  ")
    logging.info(config.train)

    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    device = torch.device('cuda')

    if args.net == "conv64":
        net = ConvNet(num_classes=5,Location=True).to(device)
        dict_path = "checkpoint/BaseModel_Conv64_best.ckpt"
        feat_dim = 64
    elif args.net == "resnet12":
        net = resnet12(num_classes=args.n_way,Location=True,grid=args.grid,baseline = args.baseline).to(device)
        # dict_path = "checkpoint/BaseModel_resnet12_0.8.ckpt"
        dict_path = "checkpoint_new/BaseModel_resnet12_mini_imagenet_mosaic.ckpt"
        dict_path = "checkpoint_new/BaseModel_resnet12_mini_imagenet_clean_last.ckpt"
        # dict_path = "checkpoint_new/BaseModel_resnet12_mini_imagenet_best_proto.ckpt"
        # dict_path = "checkpoint_new/BaseModel_resnet12_mini_imagenet_best_clean.ckpt"
        # baseliine++ :0.72
        # dict_path = "checkpoint_new/BaseModel_resnet12_" + args.dataset + "_best_clean.ckpt"
        # dict_path = "checkpoint_new/BaseModel_resnet12_" + args.dataset + "_proto.ckpt"
        feat_dim = 640
    elif args.net == "resnet18":
        net = resnet18(num_classes=5, Location=True).to(device)
        dict_path = "checkpoint/BaseModel_resnet18_best.ckpt"
        feat_dim = 512
    # lr_list = [0.001,0.0005,0.0001,0.00005]

    acc_fsl = []
    acc_sat = np.zeros(10,dtype=int)
    acc_sick = 0
    for i in range(args.num_task):
        net = load_model(net)
        # 根据小样本类构造相似类:
        # 抽样小样本类
        trainloader,devloader = FSL_sample()
        # getMosaic_pseudo(cls_fsl_dict, args.num_sim, args.sim_way, args.dist_scale, model=args.net, dict_path=dict_path)
        train()
        # 统计 0.5--1的acc情况
        acc = acc_fsl[-1]
        idx = int((acc-0.5)/0.05)
        if acc >= 0.5:
            acc_sat[idx] += 1
        else:
            acc_sick += 1
        print("Acc 分布: ")
        print("-"*180)
        print("  [--,0.50]\t [0.5,0.55]\t [0.55,0.6]\t [0.6,0.65]\t [0.65,0.7]\t [0.7,0.75]\t [0.75,0.8]\t [0.8,0.85]\t [0.85,0.9]\t [0.9,0.95]\t [0.95,1.0] ")
        print("-"*180)
        print("     {}    \t     {}    \t    {}     \t     {}    \t     {}    \t      {}    \t    {}    \t     {}    \t     {}    \t     {}    \t     {}    ".format(
            acc_sick,acc_sat[0],acc_sat[1],acc_sat[2],acc_sat[3],acc_sat[4],acc_sat[5],acc_sat[6],acc_sat[7],acc_sat[8],acc_sat[9]))
        print("-"*180)
        print("{} tasks Average Acc: {:.2f}".format(len(acc_fsl),np.mean(acc_fsl)*100))
    print("{} Tasks average Acc:  {:.2f} ".format(args.num_task,np.mean(acc_fsl)*100))






