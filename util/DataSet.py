import csv
import os.path
import random
import shutil
import time
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import torchvision.transforms as T
# from util.Mosaic_fsl import getMosaic_pseudo
import numpy as np

path = '../workship/temp/test'
img_size = 84
is_save = False

class MiniImageNetDataSet(data.Dataset):
    def __init__(self,root, mode="train_train",n_way = 5,mosaic = False,mosaic_novel_rate=1, mosaic_cls=3,Location=False,sample_idx=None,
                 mosaic_k=2, transform=None,transform_mosaic=None,mosaic_origin_num =1,boundingbox=False,
                 mosaic_nk=None, pseudo_nk=None,size_bias=0,grid=2,fsl_label =None,random_mosaic=False,test_mosaic_ob=None,is_BaseMosaic=False,is_returnName=False,mosaic_center=False):
        self.root = root
        self.transform = transform
        self.transform_mosaic= transform_mosaic
        self.img_path,self.mode = mode.split("_")
        self.mosaic = mosaic
        self.n_way = n_way
        self.mosaic_k = mosaic_k
        self.mosaic_cls = mosaic_cls
        self.Location =Location
        # img_num_dict 记录base中所有的img
        # train_num_dict 记录 base训练阶段的样本数量
        self.img_num_dict = []
        self.train_num_dict = []
        self.boundingbox = boundingbox
        self.mosaic_origin_num = mosaic_origin_num
        self.w = mosaic_novel_rate  # fewshot图片的label占比
        # self.train_num_pc = train_num_pc
        # self.split = 0.9
        self.split = 0.99
        self.sample_idx_list = sample_idx
        self.size_bias = size_bias
        # print(os.path.join(root, self.img_path+ ".csv"))
        images_list, label_list, label_dict = self.getCls(os.path.join(root, self.img_path+ ".csv"))
        self.label_dict=label_dict
        self.label_list = label_list
        self.grid = grid
        self.bound_target = 1
        self.mosaic_nk, self.pseudo_nk = mosaic_nk,pseudo_nk
        self.t = 0
        self.fsl_label = fsl_label
        self.random_mosaic = random_mosaic
        self.test_mosaic_list=None
        self.is_BaseMosaic = is_BaseMosaic
        self.is_returnName = is_returnName
        self.mosaic_center = mosaic_center

        if self.mosaic_center:
            self.dict_path = np.load("sample/dict_path.npy")
            self.center_sample = np.load("sample/center_sample.npy")
        # Base 预训练阶段
        if self.mode != "fsl":
            if mosaic:
                self.origin_samples = self.getSamples(images_list,label_list)
                self.samples = self.getSamples(images_list, label_list, mosaic_k)
            else:
                self.samples = self.getSamples(images_list, label_list)
        else:
            if mosaic:
                images_list_base, label_list_base, label_dict_base = self.getCls(os.path.join(root, "train.csv"))
                self.label_list_base = label_list_base
                self.label_dict_base = label_dict_base
                self.Base_samples = self.getSamples(images_list_base, label_list_base)
                self.origin_samples = self.fsl_samples(images_list,label_list,sample_idx_list=self.sample_idx_list)
                self.samples = self.fsl_samples(images_list, label_list,sample_idx_list=self.sample_idx_list,mosaic_k=mosaic_k)
            else:
                if test_mosaic_ob != None:
                    self.test_mosaic_list = self.fsl_samples(images_list,label_list,sample_idx_list=test_mosaic_ob)
                    self.samples = self.fsl_samples(images_list, label_list, sample_idx_list=self.sample_idx_list,
                                                    mosaic_k=mosaic_k*n_way)
                    if is_save:
                        self.save_temp()
                else:
                    self.samples = self.fsl_samples(images_list, label_list,sample_idx_list=self.sample_idx_list)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if is_save :
            idx_expand = str(idx).rjust(8, '0')
            file_path = os.path.join(path,idx_expand+".jpg")
            img = Image.open(file_path).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')
        img_ = img
        # label_loc = label

        if self.transform:
            img = self.transform(img)
            if self.mode == "fsl":
                lab = label
                if self.test_mosaic_list != None and is_save==False:
                    if idx % (self.mosaic_k*self.n_way) != 0:
                        img = self.test_mosaic(img,idx)

                if self.Location:
                    if self.mosaic:
                        # mosaic_img, label_loc = self.mosaicImg(img, idx, label)
                        mosaic_img,label_loc = img,torch.ones(self.grid**2)
                        lab = torch.zeros(self.n_way)
                        lab[label] = 1
                        # 8.8 取消 support阶段的mosaic
                        # # 无 原始图片
                        if idx % (self.mosaic_k/5) != 0:
                            if self.is_BaseMosaic:
                                # print("-----------")
                                img, lab = self.Base_Mosaic(img_, idx, label)
                                # lab = torch.ones(self.n_way)*(1-self.w)/(1-self.n_way)
                                # lab[label] = self.w

                            else:
                                img,_ = self.mosaicImg(img,idx,label,origin=True)


                    return img,mosaic_img,lab,label_loc

        if self.is_returnName:
            return img_path,img,label

        return img,label

    def __len__(self):
        return len(self.samples)

    def fsl_samples(self,image_lsit,label_lsit,sample_idx_list,mosaic_k=1):
        samples = []
        label_dict= {}
        ind = 0
        # print(len(sample_idx_list))
        for idx in sample_idx_list:
            img = image_lsit[idx]
            # print(img)
            label = label_lsit[idx]

            if label not in label_dict.keys():
                label_dict[label] = ind
                ind += 1
            item = img,label_dict[label]
            for i in range(mosaic_k):
                samples.append(item)
        return samples

    def getSamples(self,image_lsit,label_lsit,mosaic_k=1):
        # 确保随机后的结果相同
        random.seed(2022)
        samples = []
        cls_num = len(set(label_lsit))
        # nspc = int(len(image_lsit) / cls_num)
        Flag = True if self.mode == "test" else False
        # 对每个类随机采样
        self.train_num_dict = []
        for cls_id in range(len(set(label_lsit))):
            nspc = self.img_num_dict[cls_id]
            sample_idxs = random.sample(range(nspc),nspc)
            for i in range(len(sample_idxs)):
                if ((i%nspc)/nspc >= self.split and Flag) or ((i%nspc)/nspc < self.split and (Flag is False)):
                    sample_id = sum(self.img_num_dict[:cls_id]) +sample_idxs[i]
                    img = image_lsit[sample_id]
                    label = label_lsit[sample_id]
                    item = img,label
                    if cls_id >= len(self.train_num_dict):
                        self.train_num_dict.append(0)
                    self.train_num_dict[-1] += 1
                    for j in range(mosaic_k):
                        samples.append(item)

        random.seed()
        return samples

    def getCls(self,path):
        label_dict = {}
        label_list = []
        images_list = []
        ind = 0
        self.img_num_dict = []
        with open(path,'r') as f:
            reader = csv.reader(f)
            # 跳过表头
            next(reader)
            for r in reader:
                image = self.root+"/images/"+r[0]
                cls = r[1]
                if cls not in label_dict.keys():
                    self.img_num_dict.append(0)
                    label_dict[cls] = ind
                    ind += 1
                self.img_num_dict[-1] += 1
                label_list.append(label_dict[cls])
                images_list.append(image)
        return images_list,label_list,label_dict

    def mosaicImg(self,img,idx,label,origin=False):
        structure = True
        if self.random_mosaic:
            structure = False
        nspc = int(len(self.origin_samples) / self.n_way)
        cls_idx = int(idx / (self.mosaic_k * nspc))
        img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
        img_path, label_m = self.origin_samples[img_idx]
        img = Image.open(img_path).convert('RGB')
        assert label_m == label
        if self.transform:
            img_o = self.transform(img)
        h = img_o.size(1)
        w = img_o.size(2)

        # 修改此处，实现3*3以上的，但是随机的尺寸扰动下可能不好实现
        r_list ,c_list = [],[]
        for i in range(self.grid):
            r_list.append(i/self.grid)
            c_list.append(i/self.grid)
        r_list.append(1)
        c_list.append(1)
        r_list = np.trunc(np.array(r_list) * h)
        c_list = np.trunc(np.array(c_list) * w)
        r_list = r_list.astype(int)
        c_list = c_list.astype(int)

        # 1
        # img_mosaic = torch.zeros((img_o.size(0), h, w))
        img_mosaic = img_o.clone()

        label_loc = torch.zeros(self.grid**2)
        label_mosaic = label
        nov_rate = self.w
        if self.mode == "fsl":
            label_mosaic = torch.zeros(self.n_way)
            label_mosaic.data[label] = self.w
            if self.mosaic_origin_num == 0:
                self.mosaic_origin_num = 0
                # self.mosaic_origin_num = random.randint(1, self.bound_target)
        else:
            self.mosaic_origin_num = 4
        #     抽样样本
        # idxs = [0, 1, 2, 3]
        idxs = np.array(range(self.grid**2))
        random.shuffle(idxs)
        i = 0
        #     mosaic成图片
        if self.mosaic_origin_num == 0:
            nov_rate = 0

        if origin :
            self.mosaic_origin_num = self.grid**2

        while i<self.mosaic_origin_num:
            # label_mosaic.data[label] += self.w
            if self.mode != 'fsl':
                self.size_bias = 0
            if self.mode == "fsl":
                label_loc.data[idxs[i]] = 1
            r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)

            # 2  结构化的masaic
            if structure :
                img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_o[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]]
                if i % 5 == 0 and i!=0:
                    nspc = int(len(self.origin_samples) / self.n_way)
                    cls_idx = int(idx / (self.mosaic_k * nspc))
                    img_idx =idx
                    while img_idx == idx:
                        img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    img_path, label_m = self.origin_samples[img_idx]
                    img = Image.open(img_path).convert('RGB')
                    assert label_m == label
                    if self.transform:
                        img_o = self.transform(img)
                # img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_o[:, r_list[r]:r_list[r + 1],
                #                                                                   c_list[c]:c_list[c + 1]]
                i += 1
                continue

            img_m = img
            if self.transform_mosaic:
                self.transform_mosaic.__dict__['transforms'][1] = T.RandomCrop((int(r_list[r+1]-r_list[r]),int(c_list[c+1]-c_list[c])))
                img_m = self.transform_mosaic(img_m)
            img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m
            # img_mosaic[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = img
            if self.boundingbox:
                value_r,value_c = (r_list[r + 1] - r_list[r]) / h,(c_list[c+1]-c_list[c])/w
                label_loc.data[:] = torch.Tensor([r_list[r]/h+0.5*value_r, c_list[c]/w+0.5*value_c, value_r,value_c])

            if self.mode != 'fsl':
                start = sum(self.train_num_dict[:label])
                nspc = self.train_num_dict[label]
                img_idx = random.randint(start,start+nspc-1)
                img_path, label_m = self.origin_samples[img_idx]
                img = Image.open(img_path).convert('RGB')
                assert label_m == label
            elif self.mode=='fsl' and len(self.sample_idx_list)/self.n_way >1:
                nspc = int(len(self.origin_samples) / self.n_way)
                # print(nspc)
                # print(len(self.origin_samples))
                # print(len(self.label_dict.keys()))
                cls_idx = int(idx/(self.mosaic_k*nspc))
                # print(cls_idx)
                # num_sample_per_cls
                img_idx = random.randint(cls_idx*nspc,(cls_idx+1)*nspc - 1)
                img_path,label_m= self.origin_samples[img_idx]
                img = Image.open(img_path).convert('RGB')
                assert label_m==label

            # if self.transform_mosaic:
            #     img = self.transform_mosaic(img)
            # r2, c2 = int(idxs[i] / 2), int(idxs[i] % 2)
            # img_mosaic[:, r2 * h:(r2 + 1) * h, c2 * w:(c2 + 1) * w] = img_m
            i += 1
        if self.mode == "fsl":
            sample_list = np.where(np.arange(self.n_way)!=label)[0]
            # print(sample_list)
            if len(sample_list) >= self.grid**2-self.mosaic_origin_num:
                mosaic_idx = random.sample(list(sample_list), self.grid**2-self.mosaic_origin_num)
            else:
                mosaic_idx = np.random.choice(list(sample_list),self.grid**2-self.mosaic_origin_num)
            #     mosaic成图片
            if structure is False:
                for index in mosaic_idx:
                    c = index
                    nspc = int(len(self.origin_samples) / self.n_way)
                    cls_idx = c
                    img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    img_path, label_m = self.origin_samples[img_idx]
                    assert c==label_m
                    img_m = Image.open(img_path).convert('RGB')

                    r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)
                    if self.transform_mosaic:
                        self.transform_mosaic.__dict__['transforms'][1] = T.RandomCrop(
                            (int(r_list[r + 1] - r_list[r]), int(c_list[c + 1] - c_list[c])))
                        img_m = self.transform_mosaic(img_m)
                    img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m
                    # img_mosaic[:, r2 * h:(r2 + 1) * h, c2 * w:(c2 + 1) * w] = img_m
                    # print(c)
                    # print(pseudo_nk[0][0][:])
                    # label_mosaic.data[:] +=(1-nov_rate)/(self.grid**2-self.mosaic_origin_num) * pseudo_nk[label][index][:]
                    i += 1

            # 结构性的base img:
            else:
                cls_idx = label
                while cls_idx == label:
                    mosaic_ind = random.sample(list(sample_list), 1)
                # fsl_lab = self.fsl_label[label]
                    cls_idx = mosaic_ind[0]
                nspc = int(len(self.origin_samples) / self.n_way)
                img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                img_path, label_m = self.origin_samples[img_idx]

                assert cls_idx == label_m
                img_m = Image.open(img_path).convert('RGB')
                if self.transform:
                    img_m = self.transform(img_m)
                k_m = 0
                if self.grid == 2:
                    k_tol = 3
                else:
                    k_tol = 2
                for index in range(self.grid**2-self.mosaic_origin_num):
                    if k_m>= self.mosaic_origin_num:
                        mosaic_ind = random.sample(list(sample_list), 1)
                        cls_idx = mosaic_ind[0]
                        nspc = int(len(self.origin_samples) / self.n_way)
                        # cls_idx = int(idx / (self.mosaic_k * nspc))
                        img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                        img_mosaic_path, label_m = self.origin_samples[img_idx]
                        # img_mosaic_path, label_m = self.Base_samples[img_idx]

                        assert cls_idx == label_m
                        img_m = Image.open(img_mosaic_path).convert('RGB')
                        if self.transform:
                            img_m = self.transform(img_m)
                        k_m = 0
                    else:
                        k_m += 1
                    r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)
                    img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]]
                    i += 1

            if self.Location:
                return img_mosaic, label_loc
            return img_mosaic, label

        return img_mosaic,label

    def Base_Mosaic(self,img,idx,label,origin=False):
        structure = True
        if self.random_mosaic:
            structure = False
        # nspc = int(len(self.origin_samples) / self.n_way)
        # cls_idx = int(idx / (self.mosaic_k * nspc))
        # img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
        # img_path, label_m = self.origin_samples[img_idx]
        # # img = Image.open(img_path).convert('RGB')
        # assert label_m == label
        if self.transform:
            img_o = self.transform(img)
        h = img_o.size(1)
        w = img_o.size(2)

        # 修改此处，实现3*3以上的，但是随机的尺寸扰动下可能不好实现
        r_list ,c_list = [],[]
        for i in range(self.grid):
            r_list.append(i/self.grid)
            c_list.append(i/self.grid)
        r_list.append(1)
        c_list.append(1)
        r_list = np.trunc(np.array(r_list) * h)
        c_list = np.trunc(np.array(c_list) * w)
        r_list = r_list.astype(int)
        c_list = c_list.astype(int)

        img_mosaic = img_o.clone()

        label_loc = torch.zeros(self.grid**2)
        label_mosaic = label
        nov_rate = self.w
        if self.mode == "fsl":
            label_mosaic = torch.zeros(self.n_way)
            label_mosaic.data[label] = self.w
            if self.mosaic_origin_num == 0:
                self.mosaic_origin_num = 0
                # self.mosaic_origin_num = random.randint(1, self.bound_target)
        else:
            self.mosaic_origin_num = 4
        #     抽样样本
        # idxs = [0, 1, 2, 3]
        idxs = np.array(range(self.grid**2))
        random.shuffle(idxs)
        i = 0
        #     mosaic成图片
        if self.mosaic_origin_num == 0:
            nov_rate = 0

        # if origin :
        #     self.mosaic_origin_num = self.grid**2-6
        if self.mosaic_origin_num <=0:
            self.mosaic_origin_num = 1

        while i<self.mosaic_origin_num:
            # label_mosaic.data[label] += self.w
            if self.mode != 'fsl':
                self.size_bias = 0
            if self.mode == "fsl":
                label_loc.data[idxs[i]] = 1
            r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)

            # 2  结构化的masaic
            if structure :
                img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_o[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]]
                if i % 5 == 0 and i!=0:
                    nspc = int(len(self.origin_samples) / self.n_way)
                    cls_idx = int(idx / (self.mosaic_k * nspc))
                    img_idx =idx
                    while img_idx == idx:
                        img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    img_path, label_m = self.origin_samples[img_idx]
                    img = Image.open(img_path).convert('RGB')
                    assert label_m == label
                    if self.transform:
                        img_o = self.transform(img)
                # img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_o[:, r_list[r]:r_list[r + 1],
                #                                                                   c_list[c]:c_list[c + 1]]
                i += 1
                continue

            img_m = img
            if self.transform_mosaic:
                self.transform_mosaic.__dict__['transforms'][1] = T.RandomCrop((int(r_list[r+1]-r_list[r]),int(c_list[c+1]-c_list[c])))
                img_m = self.transform_mosaic(img_m)
            img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m
            # img_mosaic[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = img

            # if self.boundingbox:
            #     value_r,value_c = (r_list[r + 1] - r_list[r]) / h,(c_list[c+1]-c_list[c])/w
            #     label_loc.data[:] = torch.Tensor([r_list[r]/h+0.5*value_r, c_list[c]/w+0.5*value_c, value_r,value_c])

            if self.mode != 'fsl':
                start = sum(self.train_num_dict[:label])
                nspc = self.train_num_dict[label]
                img_idx = random.randint(start,start+nspc-1)
                img_path, label_m = self.origin_samples[img_idx]
                img = Image.open(img_path).convert('RGB')
                assert label_m == label
            elif self.mode=='fsl' and len(self.sample_idx_list)/self.n_way >1:
                nspc = int(len(self.origin_samples) / self.n_way)
                # print(nspc)
                # print(len(self.origin_samples))
                # print(len(self.label_dict.keys()))
                cls_idx = int(idx/(self.mosaic_k*nspc))
                # print(cls_idx)
                # num_sample_per_cls
                img_idx = random.randint(cls_idx*nspc,(cls_idx+1)*nspc - 1)
                img_path,label_m= self.origin_samples[img_idx]
                img = Image.open(img_path).convert('RGB')
                assert label_m==label

            # if self.transform_mosaic:
            #     img = self.transform_mosaic(img)
            # r2, c2 = int(idxs[i] / 2), int(idxs[i] % 2)
            # img_mosaic[:, r2 * h:(r2 + 1) * h, c2 * w:(c2 + 1) * w] = img_m
            i += 1

        if self.mode == "fsl":
            # sample_list = np.where(np.arange(self.n_way)!=label)[0]
            sample_list = self.mosaic_nk[label][:]
            # print(self.mosaic_nk)
            # print(sample_list)

            if len(sample_list) >= self.grid**2-self.mosaic_origin_num:
                mosaic_idx = random.sample(range(len(sample_list)), self.grid**2-self.mosaic_origin_num)
            else:
                mosaic_idx = np.random.choice(range(len((sample_list))),self.grid**2-self.mosaic_origin_num)

            dict_nk = np.array(self.pseudo_nk, copy=True)

            pseudo_nk = np.exp(dict_nk[:, :, :] * -1) / sum(np.exp(dict_nk * -1), 3)
            #     mosaic成图片
            if structure is True:
                c = random.sample(list(sample_list), 1)[0]
                nspc = int(len(self.Base_samples) / len(self.label_dict_base.keys()))
                cls_idx = c
                img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                img_path, label_m = self.Base_samples[img_idx]
                assert c == label_m
                if self.mosaic_center:
                    img_idx = np.random.choice(self.center_sample[cls_idx,:])
                    img_path = self.dict_path[img_idx]
                img_m = Image.open(img_path).convert('RGB')
                if self.transform_mosaic:
                    img_m = self.transform(img_m)

                while i < self.grid**2:
                    #  update： 可能存在问题，i可能是4，那么会在mosaic一个后变换
                    # if i % 5 == 0 :
                    #     c =  random.sample(list(sample_list), 1)[0]
                    #     nspc = int(len(self.Base_samples) / len(self.label_dict_base.keys()))
                    #     cls_idx = c
                    #     img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    #     img_path, label_m = self.Base_samples[img_idx]
                    #     # print(c)
                    #     # print(label_m)
                    #     assert c==label_m
                    #     img_m = Image.open(img_path).convert('RGB')
                    #     if self.transform_mosaic:
                    #         img_m = self.transform(img_m)

                    r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)

                    # print(img_mosaic.shape)
                    # print(img_m.shape)
                    img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m[:, r_list[r]:r_list[r + 1],
                                                                                      c_list[c]:c_list[c + 1]]

                    i += 1
            else:
                for index in mosaic_idx:
                    cls_idx = int(self.mosaic_nk[label][index])
                    nspc = int(len(self.Base_samples) / len(self.label_dict_base.keys()))
                    img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    img_path, label_m = self.Base_samples[img_idx]

                    assert cls_idx == label_m
                    if self.mosaic_center:
                        img_idx = np.random.choice(self.center_sample[cls_idx, :])
                        img_path = self.dict_path[img_idx]
                    img_m = Image.open(img_path).convert('RGB')
                    if self.transform_mosaic:
                        self.transform_mosaic.__dict__['transforms'][1] = T.RandomCrop(
                            (int(r_list[r + 1] - r_list[r]), int(c_list[c + 1] - c_list[c])))
                        img_m = self.transform_mosaic(img_m)
                    img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m
                    # print(pseudo_nk)
                    label_mosaic.data +=(1-nov_rate)/(self.grid**2-self.mosaic_origin_num) * pseudo_nk[label][int(index)][:]

                    # for index in range(self.grid**2-self.mosaic_origin_num):
                    #     if k_m>= self.mosaic_origin_num:
                    #         mosaic_ind = random.sample(list(self.mosaic_nk), 1)
                    #         cls_idx = mosaic_ind[0]
                    #         nspc = int(len(self.Base_samples) / self.n_way)
                    #         # cls_idx = int(idx / (self.mosaic_k * nspc))
                    #         img_idx = random.randint(cls_idx * nspc, (cls_idx + 1) * nspc - 1)
                    #         img_mosaic_path, label_m = self.Base_samples[img_idx]
                    #         # img_mosaic_path, label_m = self.Base_samples[img_idx]
                    #
                    #         assert cls_idx == label_m
                    #         img_m = Image.open(img_mosaic_path).convert('RGB')
                    #         if self.transform:
                    #             img_m = self.transform(img_m)
                    #         k_m = 0
                    #     else:
                    #         k_m += 1
                    # r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)
                    # img_mosaic[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = img_m[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]]

                    # i += 1

            if self.Location:
                return img_mosaic, label_mosaic
            return img_mosaic, label

        return img_mosaic,label

    def test_mosaic(self,img,idx,transform=None):
        if transform==None:
            transform = self.transform
        support_cls = int(idx/self.mosaic_k)%self.n_way
        k_shot = int(len(self.test_mosaic_list)/self.n_way)
        test_mosaic_idx = random.randint(support_cls * k_shot, (support_cls + 1) * k_shot - 1)
        img_path, label = self.test_mosaic_list[test_mosaic_idx]
        support_img = Image.open(img_path).convert('RGB')
        support_img = transform(support_img)
        h = img.size(1)
        w = img.size(2)

        # 行列位置
        r_list, c_list = [], []
        for i in range(self.grid):
            r_list.append(i / self.grid)
            c_list.append(i / self.grid)
        r_list.append(1)
        c_list.append(1)
        r_list = np.trunc(np.array(r_list) * h)
        c_list = np.trunc(np.array(c_list) * w)
        r_list = r_list.astype(int)
        c_list = c_list.astype(int)
        # 随机打乱 mosaic 的位置
        idxs = np.array(range(self.grid ** 2))
        random.shuffle(idxs)
        i = 0
        test_mosaic_num = self.grid**2-6
        while i<test_mosaic_num:
            r, c = int(idxs[i] / self.grid), int(idxs[i] % self.grid)
            img[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]] = support_img[:, r_list[r]:r_list[r + 1], c_list[c]:c_list[c + 1]]
            i += 1
        return img

    def save_temp(self):

        self.clearDir(path)
        # 抽取出图片，然后保存在字典（file:temp_file）里，图片保存到文件件里
        tf = T.Compose([
            T.Resize([img_size, img_size]),
            T.ToTensor()])
        convert_tf = T.ToPILImage()
        for idx,(img_path,label) in enumerate(self.samples):
            img = Image.open(img_path).convert('RGB')
            img_tensor = tf(img)
            if idx % (self.mosaic_k * self.n_way) != 0:
                img_tensor = self.test_mosaic(img_tensor,idx,tf)
            file = convert_tf(img_tensor)
            idx_expand = str(idx).rjust(8,'0')
            file_name = idx_expand+".jpg"
            file.save(os.path.join(path,file_name))


    def clearDir(self,path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transform_mosaic = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.RandomCrop(42,padding=1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset = MiniImageNetDataSet("C:\PyCode\mosaic_FSL\workship\mini-imageNet",mosaic=True,mosaic_k=2,mode="train_train",
                                  transform=transform,transform_mosaic=transform_mosaic)
    print(len(dataset))

