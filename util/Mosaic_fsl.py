import os

import numpy as np

# 获取各类中心
import torchvision
import torchvision.transforms as T
import yaml
import easydict
from util.DataSet_FSL import MiniImageNetDataSet
import torch
from model.conv4 import ConvNet
from model.resnet12 import resnet12
from tqdm import tqdm
from util.DataSet_mosaic import MiniImageNetDataSet_mosaic
from model.resnet import resnet18
import logging

def getProtoType(net, dataloader, num_cls, path_dict_proto=None):
    net.eval()
    sample_num = np.zeros(num_cls)
    cls_prototype = np.zeros((num_cls,net.fc.in_features))
    device = next(net.parameters()).device
    # for step, (image, label) in enumerate(tqdm(dataloader)):
    for step, (image, label) in enumerate(dataloader):
        image = image.to(device)
        out ,pred= net(image,is_feat=True)
        out = out.cpu()
        # print(np.unique(out.data.numpy()))
        # label = label.numpy()
        for i in np.unique(label.data):
            lab_list = np.where(label.data == i)[0]
            # print(lab_list.size)
            # print(i)
            sample_num[i] += lab_list.size
            # print(out.data.numpy())
            cls_prototype[i,:] += sum(out.detach().data.numpy()[lab_list,:],0)

    for i in range(num_cls):
        cls_prototype[i,:] /= sample_num[i]
    # if path_dict_proto:
    #     np.save(path_dict_proto,cls_prototype)
    return cls_prototype

# 得到新类与相似旧类的矩阵
# bp:   base_protoyype
# fb:   fewshot_prototype
def getMosaic(bp,fp,k,sim_way,dict_scale,sim_mix=False):
    mat = euclidean_dist(bp,fp,dict_scale)
    mat = mat.T
    expand = 1.6
    mosaic_nk = np.zeros((mat.size(0),k))
    mosaic_nk_2 = np.zeros((mat.size(0),int(expand*k)))
    pseudo_nk = np.zeros((mat.size(0), k, mat.size(0)))
    # print(mat.T)
    mat_norm = mat.clone()
    # print(mat[:,0])

    if sim_way == 0:
        info ="基于最高输出一致性，提取专属相似类"
        logging.info("Base on most similiar output, find special similar class")
        # 基于最高输出相似度提取
        for i in range(mat.size(1)):
            mat_norm[:, i] = torch.exp(mat_norm[:, i] * -1) / torch.sum(torch.exp(mat_norm[:, i] * -1))
        value, idx = torch.sort(mat_norm,descending=True)
    elif sim_way == 1:
        info ="基于最近距离，提取相似类..."
        logging.info("Base on shortest distance, find similiar class")
        # 以最高的相似度提取
        value, idx = torch.sort(mat)
        # print(mat[:,1])
        # print(value)
        # value, idx = torch.sort(torch.sum(mat,0))
        # print(idx)
        # 原生softmax结果
        for i in range(mat.size(1)):
            mat_norm[:, i] = torch.exp(mat[:, i] * -1)/ torch.sum(torch.exp(mat[:, i] * -1))
    elif sim_way == 2:
        info = "基于最近相对距离，提取专属相似类..."
        logging.info("Base on related shortest distance, find special similiar class")
        for i in range(mat.size(1)):
            mat_norm[:, i] = mat_norm[:, i] / torch.sum(mat_norm[:, i])
        value, idx = torch.sort(mat_norm)
        # 原生softmax结果
        for i in range(mat.size(1)):
            mat_norm[:, i] = torch.exp(mat[:, i] * -1) / torch.sum(torch.exp(mat[:, i] * -1))
    # print(info)
    # print(mosaic_nk)

    if sim_mix == False:
        mosaic_nk[:, :] = idx[:,0:k]
    else:
        mosaic_nk_2[:, :] = idx[:, 0:int(expand * k)]
        count = np.zeros((mosaic_nk.shape[0], mosaic_nk_2.shape[1]))
        for i in range(count.shape[0]):
            for j in range(count.shape[1]):
                v = mosaic_nk_2[i, j]
                for p in range(mosaic_nk_2.shape[0]):
                    for q in range(mosaic_nk_2.shape[1]):
                        if mosaic_nk_2[p, q] == v:
                            count[i, j] += 1
        index = np.argsort(-count)
        for i in range(mosaic_nk.shape[0]):
            mosaic_nk[i, :] = mosaic_nk_2[i, index[i, 0:k]]


    for i in range(pseudo_nk.shape[0]):
        for j in range(pseudo_nk.shape[1]):
            # c 表示的是i类的第j个最近类
            c = int(mosaic_nk[i,j])
            # for s in range(pseudo_nk.shape[2]):
            #     pseudo_nk[i][j][s] = mat.data[s][c]
            # pseudo_nk[i][j][:] = np.exp(pseudo_nk[i][j][:]*-1)/sum(np.exp(pseudo_nk[i][j][:]*-1))
            # 保存距离
            # print(mat[:, 1])
            pseudo_nk[i,j,:] = mat.numpy()[:,c]

            # assert False
            # pseudo_nk[i, j, :] = mat_norm.numpy()[:, c]
            # print(i)
            # print(pseudo_nk[i,j,:])


    # print(mosaic_nk)
    # print(pseudo_nk[0][:][:])
    return mosaic_nk,pseudo_nk
    # print(idx)
    # print(sorted(mat[0,:]))

# def getMosaic_pseudo(bp, fp, k):
#     mat = euclidean_dist(bp, fp)
#     mat = mat.T
#     mosaic_nk = np.zeros((mat.size(0), k))
#     pseudo_nk = np.zeros((mat.size(0), k, mat.size(0)))
#     # print(mat.T)
#
#     for i in range(pseudo_nk.shape[0]):
#         for j in range(pseudo_nk.shape[1]):
#             for s in range(pseudo_nk.shape[2]):
#                 c = int(mosaic_nk[i][j])
#                 pseudo_nk[i][j][s] = mat.data[s][c]
#             pseudo_nk[i][j][:] = np.exp(pseudo_nk[i][j][:] * -1) / sum(np.exp(pseudo_nk[i][j][:] * -1))
#     # 未整理完。。。
#     value, idx = torch.max(torch.from_numpy(pseudo_nk),2)
#     print(value)
#     rc = np.where(idx < k)
#     t = 0
#     for i in range(mosaic_nk.shape[0]):
#         for j in range(mosaic_nk.shape[1]):
#             # r = rc[0][t]
#             mosaic_nk[i][j] = rc[1][t]
#             t += 1
#     # print(mosaic_nk)
#     # print(pseudo_nk[0][:][:])
#     return mosaic_nk, pseudo_nk

def euclidean_dist(x, y, scale):
    # x: N x D
    # y: M x D
    # print(x.size)
    x = torch.from_numpy(x)
    y =torch.from_numpy(y)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    # assert d == y.shape(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # To accelerate training, but observe little effect
    return (torch.pow(x - y, 2)*scale).sum(2)

def Mosaic_cls(net,dataloader,num_sim,sim_way,dist_scale,num_classes,path_dict_proto,sim_mix=False):
    fsl_protype = getProtoType(net, dataloader,num_classes)
    base_prototype =  np.load(path_dict_proto)
    mosaic_nk, pseudo_nk = getMosaic(base_prototype, fsl_protype, num_sim, sim_way, dist_scale,sim_mix)
    # print(pseudo_nk[0][0][:])
    return mosaic_nk, pseudo_nk

def getMosaic_pseudo(cls_dict,num_sim,sim_way,dist_scale,model,dict_path):
    gpu_device = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    device = torch.device('cuda')
    if model =="conv64":
        net = ConvNet(num_classes=64).to(device)
        out_dim = 64
    elif model =="resnet12":
        net = resnet12(num_classes=64).to(device)
        out_dim = 640
    elif model =="resnet18":
        net = resnet18(num_classes=64).to(device)
        out_dim = 512
    state_dict = torch.load(dict_path)
    net.load_state_dict(state_dict)

    config = yaml.load(open("./config/Base_train.yaml"), Loader=yaml.FullLoader)
    config = easydict.EasyDict(config)
    transform = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    DataSet_train = MiniImageNetDataSet(root="./"+config.data.root, mode="train_Base", mosaic=False,transform=transform)
    trainloader = torch.utils.data.DataLoader(DataSet_train, batch_size=64,
                                              shuffle=False,
                                              num_workers=config.data.trainloader.worker)
    DataSet_fsl = MiniImageNetDataSet_mosaic(root="./"+config.data.root, mode="train_fsl",mosaic=False,cls_dict=cls_dict, transform=transform)
    dataloader_fsl = torch.utils.data.DataLoader(DataSet_fsl, batch_size=25,
                                                 shuffle=False,
                                                 num_workers=config.data.trainloader.worker)
    path_dict_proto = "util/"+model+"_prototype.npy"
    if os.path.exists(path_dict_proto):
        base_prototype = np.load(path_dict_proto)
    else:
        base_prototype = getProtoType(net, trainloader, 64, out_dim, device,path_dict_proto)
    fsl_protype = getProtoType(net, dataloader_fsl, 5, out_dim, device)

    # base_prototype = np.ones((80, 256))
    # fsl_protype = np.ones((5, 256))
    # print(fsl_protype.shape[0])
    mosaic_nk,pseudo_nk = getMosaic(base_prototype, fsl_protype, num_sim,sim_way,dist_scale)
    # print(pseudo_nk[0][0][:])
    np.save("util/mosaic_nk.npy",mosaic_nk)
    np.save("util/pseudo_nk.npy", pseudo_nk)
    # return mosaic_nk,pseudo_nk

if __name__ == '__main__':

       getMosaic_pseudo(num_cls=3)