import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def ce_loss(out, lab,temperature=1):

    out = F.softmax(out/temperature, 1)
    # print(label.shape)
    # print(out.shape)
    # print(lab.size)
    if len(lab.size()) == 1:
        label = torch.zeros((out.size(0),
                             out.size(1))).long().cuda()
        label_range = torch.arange(0, out.size(0)).long()
        label[label_range, lab] = 1
        lab = label
    loss = torch.mean(torch.sum(-lab*torch.log(out+1e-8),1))

    return loss

def Few_loss(out,lab):
    # 目的似乎是实现poly loss，但实践过程中有误
    # 这个损失意义不大
    out = F.softmax(out, 1)
    eps = 2
    n = 1
    poly_head = torch.zeros(out.size(0),out.size(1)).cuda()
    for i in range(n):
        poly_head += eps*1/(i+1)*torch.pow(1-out,(i+1))
    ce_loss = torch.sum(-lab * torch.log(out + 1e-8) - poly_head,1)
    loss = torch.mean(ce_loss)
    return loss

def loc_loss(out_loc,lab):

    out_loc = F.sigmoid(out_loc)
    # print(out_loc)
    log_loc = (-lab) * torch.log(out_loc + 1e-8)-(1-lab)* torch.log(out_loc + 1e-8)
    # loss = torch.mean(torch.sum(log_loc, 1))
    loss = torch.mean(torch.mean(log_loc, 1))

    # out_loc = out_loc.view(out_loc.size(0),out_loc.size(1),-1,2)
    # out_loc = F.softmax(out_loc,dim=3)

    return loss

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    # unsqueeze 在dim维度进行扩展
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_Loss(feat_out,lab,prototypes,epoch,center=False,temperature = 1):
    temperature = 256
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return label_cpu.eq(c).nonzero()[:].squeeze(1)

    feat_cpu = feat_out.cpu()
    label_cpu = lab.cpu()
    prototypes = prototypes.cpu()

    n_classes = prototypes.size(0)
    # print(label_cpu.size())
    # _,label_cpu = torch.max(label_cpu,1)
    if len(label_cpu.size()) == 1:
        classes = np.unique(label_cpu)
        # print(classes)
        #  map :调用函数supp_idsx  classes作为参数列表
        support_idxs = list(map(supp_idxs,classes))
        prototypes_update = torch.stack([feat_cpu[idx_list].mean(0) for idx_list in support_idxs])
    else:
        classes = range(n_classes)
        count = sum(label_cpu, 0)
        # feat_cpu dim  : 64 * 640
        #  label dim    : 64 * 5
        # print(feat_cpu.T.size())
        # print(label_cpu.size())
        prototypes_update = torch.matmul(feat_cpu.T,label_cpu.float())/torch.tensor(count).float()
        prototypes_update = prototypes_update.T
    # if epoch == 0 :
    beta = 0.9
    # prototypes[classes, :] = prototypes_update.detach()
    prototypes[classes, :] = beta * prototypes[classes, :] + (1-beta) * prototypes_update.detach()

    # if epoch >= 1:
    #     prototypes[classes, :] = 0.9 * prototypes[classes, :] + 0.1 * prototypes_update.detach()
    # else:
    #     prototypes[classes, :] = prototypes_update.detach()


    # print(prototypes)
    # print(prototypes)
    if len(lab.size()) == 1:
        label = torch.zeros((feat_cpu.size(0),
                             n_classes)).long().cuda()
        label_range = torch.arange(0, feat_cpu.size(0)).long()
        label[label_range, lab] = 1
        lab = label
    # if center:
    #     # print(type(prototypes))
    #     dist = 1/2*torch.sum(torch.pow((feat_cpu-torch.matmul(lab.cpu(),prototypes.long())),2))
    #     loss = dist
    # else:
    dists = euclidean_dist(feat_cpu,prototypes)/temperature
    # print(dists.shape)
    log_p_y = F.log_softmax(-dists, dim=1)
    y  = F.softmax(-dists,1)
    # print(dists.size())
    # print(y.size())
    # print(y)

    loss = torch.sum(torch.mean(-lab.cpu() * torch.log(y+1e-8), 1))
    # print(loss)
    return loss,prototypes