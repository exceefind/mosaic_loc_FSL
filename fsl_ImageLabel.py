import csv
import math
import random
import numpy.random
from tqdm import tqdm
from model.conv4 import ConvNet
from model.resnet12 import resnet12
from model.MLP_Net import mlp_Net
from model.resnet import *
import yaml
import easydict
import argparse
import os
import torch
import torch.nn as nn
from util.DataSet_preload import MiniImageNetDataSet
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
# random.seed(20220520)
# random.seed(6628)
test_mosaic = 5

fsl_file = []
support_file = []
query_file = []
task_index = 0
save_sample = False
show_step = 1

if save_sample == False:
    fsl_file = np.loadtxt("sample/fsl_file.txt",dtype=int)
    support_file = np.loadtxt("sample/support_file.txt",dtype=int)
    query_file = np.loadtxt("sample/query_file.txt",dtype=int)

def train():

    params = [{'params':prompt_net.parameters(),'lr':args.lr_base},
             {'params':backbone_net.parameters(),'lr':args.lr_base},
              {'params':mlp_net.parameters(),'lr':args.lr_base*config.train.lr_scale}]

    if args.train_head:
        params = [{'params':mlp_net.parameters(),'lr':args.lr_base*config.train.lr_scale}]
    Epoch_num = config.train.Epoch_num if args.Epoch is None else args.Epoch
    mse_criterion = torch.nn.MSELoss()

    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, weight_decay=float(config.train.weight_decay))
    elif config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(params,lr=args.lr_base ,momentum=config.train.momentum,
                                    weight_decay=float(config.train.weight_decay))

    Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.markstone, gamma=config.train.decay_lr)
    # Schuler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=int(Epoch_num/4),T_mult=3)
    best_dev = 0
    beta_moving = config.train.beta_moving
    Dev_Acc_moving = 0
    info = ""
    logging.info("Few-shot model start training...")
    sum_acc = 0
    prototypes = torch.zeros((args.n_way, feat_dim),requires_grad=False).to(device)
    for epoch in range(Epoch_num):
        prompt_net.train()
        backbone_net.train()
        mlp_net.train()
        correct_train,total_train = 0,0
        for step,(image,mosaic_img,label,lable_loc) in enumerate(trainloader):
            image = image.to(device)
            mosaic_img = mosaic_img.to(device)
            label = label.to(device)
            label_loc = lable_loc.to(device)
            feat_fsl, out = backbone_net(image, is_feat=True)
            feat_loc, _ = prompt_net(mosaic_img, is_feat=True)
            tensor_concat = torch.cat([feat_fsl,feat_loc],dim=1)
            # print(tensor_concat)
            out_loc = mlp_net(tensor_concat)
            loss_loc = loc_loss(out_loc, label_loc)
            if args.proto_loss:
                loss_cls,prototypes_update = prototypical_Loss(feat_fsl,label,prototypes,epoch,temperature=config.train.temperature)
                prototypes = prototypes_update.to(device)
            else:
                _, prototypes_update = prototypical_Loss(feat_fsl, label, prototypes, epoch,
                                                                temperature=config.train.temperature)
                prototypes = prototypes_update.to(device)
                if args.Few_loss:
                    loss_cls = Few_loss(out,label)
                else:
                    loss_cls = ce_loss(out,label,temperature=config.train.temperature)


            # loss = 0 * loss_cls + 0 * loss_loc
            loss = args.alpha*loss_cls + args.beta * loss_loc
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(label.size()) != 1:
                __, label = torch.max(label, 1)
            _,pred = torch.max(out,1)
            correct_train += (pred == label).sum().item()
            total_train += label.size(0)
        if epoch%10 == 0 :
            info = "Epoch {} of {} :  \t  Step {} of {}: \t loss_train: {:.4f} \t loss_classification: {:.4f} \tloss_Location: {:.4f}".format(epoch,Epoch_num,step,len(trainloader),loss.item(),loss_cls.item(),loss_loc.item())
            logging.info(info)

        Schuler.step()

        prompt_net.eval()
        backbone_net.eval()
        mlp_net.eval()
        correct = total = 0

        with torch.no_grad():

            for step, (img, label) in enumerate(devloader):
                img = img.to(device)
                label = label.to(device)
                feat_predict, _ = backbone_net(img, is_feat=True)
                predict = F.log_softmax(-euclidean_dist(feat_predict,prototypes),dim=1)

                if args.test_mosaic:
                    # feat = feat_predict.expand(config.data.trainloader.batch_size,-1)
                    result = torch.ones(1)*-1
                    result = result.to(device)
                    # print(feat.shape)

                    for step,(mosaic_img,label_loc) in enumerate(dev_mosaic_loader):
                        feat = feat_predict.expand(mosaic_img.shape[0], -1)
                        mosaic_img = mosaic_img.to(device)
                        label_loc = label_loc.to(device)
                        feat_loc, _ = prompt_net(mosaic_img, is_feat=True)
                        out_loc = mlp_net(torch.cat([feat,feat_loc],1))
                        _, pre = torch.max(out_loc, 1)
                        out = torch.zeros_like(label_loc)
                        label_range = torch.arange(0, out.size(0)).long()
                        out[label_range,pre]=1
                        res = torch.sum(torch.mul(out,label_loc),1)
                        result = torch.cat([result,res],0)
                        # print(result)
                    pre = torch.mode(result)[0]

                else:
                    _, pre = torch.max(predict, 1)
                correct += (pre == label).sum().item()
                total += label.size(0)
            dev_acc = correct / total
            sum_acc += dev_acc
            Dev_Acc_moving = beta_moving * Dev_Acc_moving + (1-beta_moving) * dev_acc
            if  dev_acc> best_dev:
                best_dev = dev_acc
                # logging.info("Epoch {} of {}:  \t  model_acc : {} \t  model_save: {}".format(epoch,config.train.Epoch_num,dev_acc,save_path))
        info = "Epoch {} of {}:  \t Dev_Acc : {:.4f}  \t Best_Dev_Acc:{:.4f}".format(epoch, Epoch_num,
                                                                          dev_acc,best_dev)
        if epoch%show_step == 0:
            print(info)
        logging.info(info)

    # file = open("record/_"+str(args.id)+'_'+str(args.k_shot)+"_"+str(args.net)+"_"+".txt","a+")
    # file.write(str(dev_acc) + "\n")
    # file.close()
    toc = time.time()
    acc_fsl.append(dev_acc)
    print("Dev_Acc:  {:.4f} \t  elapse:  {:.2f} min".format(dev_acc,(toc-tic)/60))
    logging.info("Few-shot model start finish!")
    logging.info("-"*100+"experiment finish!"+"-" * 100)

def FSL_sample(only_test = False,center_list = None,fsl_cls = [],query_list=[]):
    # 首先: 统计小样本类的总个数、各类下的样本个数
    fsl_stage = "val" if args.fsl_val else "test"
    if args.fsl_val:
        path = config.data.root + "/val.csv"
    else:
        # print("---------")
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
    # 反转字典的keys和values
    label_dict_new = {value: key for key, value in label_dict.items()}
    # 获取类名
    import json
    map = open('map.txt', 'r')
    js = map.read()
    dic= json.loads(js)
    dic_cls = dict(dic.values())
    map.close()

    transform_mosaic = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),

        T.RandomCrop(args.mosaic_baseSize * img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform_train = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),
        T.RandomCrop(img_size, padding=1),
        T.RandomHorizontalFlip(),

        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # T.RandomRotation(30),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        T.Resize([img_size, img_size]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    if only_test:
        DataSet_val = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage + "_fsl", n_way=args.n_way,
                                          sample_idx=query_list,
                                          test_mosaic_ob=center_list, mosaic_k=test_mosaic, transform=transform,
                                          mosaic_origin_num=args.mosaic_ori_num, grid=args.grid, fsl_label=fsl_cls)
        devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size,
                                                num_workers=config.data.testloader.worker)
        return devloader

    support_list = []
    query_list = []
    fsl_cls = random.sample(range(cls_num),args.n_way)
    logging.info('Sample FSL classes:   ')


    for cls_id in fsl_cls:
        start = sum(num_dict[:cls_id])
        nspc = num_dict[cls_id]
        fsl_sample = random.sample(range(start, start+nspc-1),args.k_shot+args.q_shot)
        fsl_sup = fsl_sample[:args.k_shot]
        fsl_query = fsl_sample[args.k_shot:]
        support_list.extend(fsl_sup)
        query_list.extend(fsl_query)

    if save_sample == False:
        fsl_cls = list(fsl_file[task_index,:])
        support_list = list(support_file[task_index,:])
        query_list = list(query_file[task_index,:])
    else:
        fsl_file.append(np.asarray(fsl_cls))
        support_file.append(np.asarray(support_list))
        query_file.append(np.asarray(query_list))

    if only_test == False:
        print(fsl_cls)
        for c in fsl_cls:
            print(dic_cls[label_dict_new[c]],end=", ")
        print()
        logging.info(fsl_cls)
    # print(support_list)
    # print(query_list)
    # support_list = [3867, 3915, 4001, 3876, 3941, 2497, 2687, 2511, 2848, 2407, 258, 47, 433, 471, 364, 6362, 6022, 6391, 6397, 6258, 3560, 3021, 3092, 3538, 3333]
    # query_list = [3877, 4012, 4057, 3671, 3705, 3640, 3687, 3989, 3963, 3853, 4051, 3869, 3616, 3711, 3775, 3784, 4010, 3966, 3850, 3868, 4132, 4193, 3603, 4030, 4023, 3824, 3623, 4185, 3756, 4148, 3965, 3906, 3607, 4191, 3842, 3688, 3981, 3772, 3958, 3709, 3670, 4055, 3967, 3901, 4125, 4119, 3895, 3692, 3839, 4149, 2939, 2885, 2722, 2542, 2977, 2649, 2911, 2448, 2596, 2925, 2692, 2531, 2957, 2940, 2724, 2948, 2561, 2429, 2731, 2794, 2858, 2795, 2857, 2694, 2946, 2961, 2499, 2617, 2895, 2844, 2881, 2490, 2823, 2861, 2918, 2418, 2910, 2580, 2565, 2923, 2590, 2738, 2780, 2740, 2470, 2503, 2581, 2594, 2842, 2781, 133, 222, 156, 178, 254, 122, 54, 307, 420, 415, 534, 487, 544, 185, 302, 516, 595, 453, 554, 198, 217, 14, 311, 597, 84, 344, 472, 3, 102, 328, 444, 531, 60, 387, 320, 367, 543, 422, 396, 405, 282, 1, 301, 164, 127, 182, 553, 238, 150, 459, 6368, 6255, 6077, 6594, 6230, 6427, 6151, 6547, 6226, 6286, 6566, 6576, 6351, 6339, 6556, 6323, 6257, 6480, 6349, 6191, 6075, 6592, 6431, 6059, 6205, 6396, 6513, 6492, 6490, 6589, 6305, 6591, 6110, 6327, 6098, 6176, 6467, 6086, 6298, 6578, 6253, 6173, 6105, 6559, 6335, 6399, 6112, 6184, 6134, 6180, 3199, 3358, 3583, 3107, 3405, 3049, 3377, 3446, 3441, 3454, 3416, 3032, 3176, 3149, 3348, 3043, 3562, 3315, 3345, 3096, 3517, 3311, 3019, 3031, 3481, 3447, 3577, 3180, 3556, 3030, 3085, 3265, 3105, 3204, 3409, 3045, 3469, 3468, 3486, 3382, 3426, 3010, 3142, 3594, 3597, 3387, 3183, 3527, 3357, 3554]
    DataSet_train = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl", n_way=args.n_way,sample_idx=support_list,
                                               mosaic_novel_rate=args.Novel_Mosaic_rate, mosaic_cls=args.num_sim
                                               , mosaic=False, mosaic_k=args.mosaic_num,
                                               Location=False,transform=transform_train,transform_mosaic=transform_mosaic,
                                               mosaic_origin_num=args.mosaic_ori_num, boundingbox=args.boundingbox)


    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=config.data.trainloader.batch_size,
                                              num_workers=config.data.trainloader.worker)

    # mosaic_nk, pseudo_nk = Mosaic_cls(net, dataloader=trainloader,num_sim=args.num_sim,sim_way=args.sim_way,sim_mix=args.sim_mix,
    #                                   dist_scale=args.dist_scale,num_classes=args.n_way,path_dict_proto="prototype/Base_"+args.net+"_"+args.dataset+"_mosaic_0602.npy")
    mosaic_nk, pseudo_nk = Mosaic_cls(backbone_net, dataloader=trainloader, num_sim=args.num_sim, sim_way=args.sim_way,
                                      sim_mix=args.sim_mix,
                                      dist_scale=args.dist_scale, num_classes=args.n_way,
                                      path_dict_proto="prototype/Base_" + args.net + "_" + args.dataset + "_pt_mosaic.npy")

    # print(mosaic_nk)
    DataSet_mosaic = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl",n_way=args.n_way, sample_idx=support_list,
                                        mosaic_novel_rate=args.Novel_Mosaic_rate, mosaic_cls=args.num_sim
                                        , mosaic=True, mosaic_k=args.mosaic_num,mosaic_nk=mosaic_nk, pseudo_nk=pseudo_nk,
                                        Location=args.Location,transform=transform_train,transform_mosaic=transform_mosaic,
                                        mosaic_origin_num=args.mosaic_ori_num, boundingbox=args.boundingbox,size_bias=args.size_bias,
                                         grid=args.grid,fsl_label=fsl_cls,random_mosaic=args.random_mosaic,is_BaseMosaic=args.base_mos,mosaic_center=True)

    mosaic_loader = torch.utils.data.DataLoader(DataSet_mosaic, batch_size=config.data.testloader.batch_size, shuffle=True,
                                            num_workers=config.data.testloader.worker)


    DataSet_val = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage + "_fsl", sample_idx=query_list,
                                      transform=transform)

    devloader = torch.utils.data.DataLoader(DataSet_val, batch_size=config.data.testloader.batch_size,
                                            num_workers=config.data.testloader.worker)

    Dataset_val_mosaic = MiniImageNetDataSet(root=config.data.root, mode=fsl_stage+"_fsl",n_way=args.n_way, sample_idx=support_list,
                                        mosaic_novel_rate=args.Novel_Mosaic_rate, mosaic_cls=args.num_sim
                                        , mosaic=True, mosaic_k=args.mosaic_num,mosaic_nk=mosaic_nk, pseudo_nk=pseudo_nk,
                                        Location=args.Location,transform=transform_train,transform_mosaic=transform_mosaic,
                                        mosaic_origin_num=args.mosaic_ori_num, boundingbox=args.boundingbox,size_bias=args.size_bias,
                                         grid=args.grid,fsl_label=fsl_cls,random_mosaic=args.random_mosaic,is_BaseMosaic=args.base_mos,isTestMosaic=args.test_mosaic,image_size=args.img_size,mosaic_center=args.mosaic_center)

    dev_mosaic_loader = torch.utils.data.DataLoader(Dataset_val_mosaic, batch_size=config.data.trainloader.batch_size,
                                            num_workers=config.data.testloader.worker)
    # print(len(Dataset_val_mosaic))
    return mosaic_loader,devloader,dev_mosaic_loader

def load_model():

    state_dict = torch.load(dict_path)
    # 读取参数 预训练的参数和自己模型的参数
    pretrained_dict = state_dict
    # pretrained_dict = torch.load('archive/data.pkl')
    mlp_net = mlp_Net(in_dim=feat_dim*2,out_dim=args.grid**2).to(device)
    model_dict = prompt_net.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉，通过严格对应层名获取对应参数
    dict_not_include = ["fc.weight","fc.weight_g","fc.weight_v", "fc.bias", "fc_loc.weight", "fc_loc.bias", "classifier.weight", "classifier.bias"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in dict_not_include)}
    # 更新现有、对等的model_dict
    model_dict.update(pretrained_dict)
    # 加载参数model.state_dict到自己的模型中
    if args.no_pretrained is False:
        prompt_net.load_state_dict(model_dict)
        backbone_net.load_state_dict(model_dict)

    return prompt_net,backbone_net,mlp_net

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
    parser.add_argument("--q_shot", type=int, default=50)
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
    parser.add_argument("--beta", default=1, type=float)
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
    parser.add_argument("--test_mosaic",default=False,action='store_true')
    parser.add_argument("--base_mos", default=False, action='store_true')
    parser.add_argument("--mosaic_center",default=False,action="store_true")
    parser.add_argument("--img_size",default=84,type=int)
    parser.add_argument('--alpha',type=float,default=0.5)
    args = parser.parse_args()
    img_size = args.img_size
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
        config = yaml.load(open("config/fsl_Img_Label.yaml"), Loader=yaml.FullLoader)
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
        prompt_net = ConvNet(num_classes=5,Location=True).to(device)
        backbone_net = ConvNet(num_classes=5,Location=True).to(device)
        dict_path = "checkpoint/BaseModel_Conv64_best.ckpt"
        feat_dim = 64
    elif args.net == "resnet12":
        prompt_net = resnet12(num_classes=args.n_way,Location=True,grid=args.grid,baseline = args.baseline).to(device)
        backbone_net = resnet12(num_classes=args.n_way, Location=True, grid=args.grid, baseline=args.baseline).to(device)
        dict_path = config.train.dict_state
        feat_dim = 640
    elif args.net == "resnet18":
        prompt_net = resnet18(num_classes=5, Location=True).to(device)
        backbone_net = resnet18(num_classes=5, Location=True).to(device)
        dict_path = config.train.dict_state
        feat_dim = 512
    # lr_list = [0.001,0.0005,0.0001,0.00005]

    acc_fsl = []
    acc_sat = np.zeros(10,dtype=int)
    acc_sick = 0
    for i in range(args.num_task):
        task_index = i
        tic = time.time()
        prompt_net,backbone_net,mlp_net = load_model()
        # 根据小样本类构造相似类:
        # 抽样小样本类
        trainloader,devloader,dev_mosaic_loader = FSL_sample()
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
        if i%10 == 0 and save_sample:
            np.savetxt("sample/fsl_file.txt", np.array(fsl_file), fmt='%d')
            np.savetxt("sample/support_file.txt", np.array(support_file), fmt='%d')
            np.savetxt("sample/query_file.txt", np.array(query_file), fmt='%d')
            np.savetxt("sample/acc_proto.txt",np.array(acc_fsl)*100,fmt='%.02f')
    print("{} Tasks average Acc:  {:.2f} ".format(args.num_task,np.mean(acc_fsl)*100))






