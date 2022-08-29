import os.path
import random
import time

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import torchvision.transforms as T
# from util.Mosaic_fsl import getMosaic_pseudo
import numpy as np

class MiniImageNetDataSet_mosaic(data.Dataset):
    def __init__(self,root, mode="train_fsl", mosaic_cls = 3,mosaic_novel_rate = 1,
                 cls_dict=None, n_way = 5,k_shot = 5,mosaic = True, Location = False,
                 mosaic_k=200, transform=None, transform_original=None,mosaic_origin_num =1,boundingbox=False):
        self.root = root
        self.transform = transform
        self.transform_original = transform_original
        # self.root = os.path.join(root,mode)
        train_mode,FSL_mode = mode.split("_")
        self.mode = mode
        self.train_mode = train_mode
        self.root = os.path.join(root,train_mode)
        self.FSL_mode = FSL_mode
        self.mosaic = mosaic
        self.mosaic_k = mosaic_k
        self.mosaic_cls = mosaic_cls
        self.n_way =n_way
        self.k_shot =k_shot
        self.cls_dict = cls_dict
        self.w = mosaic_novel_rate #fewshot图片的label占比
        self.Location = Location
        self.mosaic_nk = np.load("util/mosaic_nk.npy")
        self.pseudo_nk = np.load("util/pseudo_nk.npy")
        self.grid = 4
        self.bound_target = 2
        self.mosaic_origin_num =mosaic_origin_num
        self.boundingbox = boundingbox
        # print(self.mosaic_nk)
        if FSL_mode == "fsl":
            cls_dict = self.cls_dict
            # print(cls)
            if mosaic and train_mode== "train" :
                self.original_samples = self.load_image(self.root, cls_dict)
                self.samples =self.load_image(self.root,cls_dict,mosaic_k)
                cls_Base = self.getCls(os.path.join(root, "spilt", "BaseSet.txt"))
                self.cls_Base = cls_Base
                # self.samples_Base = self.load_image(self.root, cls_Base)
            else:
                self.samples = self.load_image(self.root, cls_dict)

        else:
            cls = self.getCls(os.path.join(root, "spilt", "BaseSet.txt"))
            self.samples = self.load_image(self.root,cls)


    def __getitem__(self, idx):
        img_path,label = self.samples[idx]
        img_origin = Image.open(img_path).convert('RGB')
        label_loc = torch.zeros(self.grid)

        if self.transform_original is not None and idx % self.mosaic_k == 0:
            label_original = torch.zeros(self.n_way)
            label_original.data[label] = 1
            label_loc.data[:]=1
            if self.boundingbox:
                label_loc.data[:] = torch.Tensor([0.5,0.5,1,1])
            img = self.transform_original(img_origin)
            return img, label_original, label_loc

        if self.transform:
            img = self.transform(img_origin)
        if self.mosaic and self.train_mode== "train":
            # print(img.shape)
            h = img.size(1)
            w = img.size(2)
            img_mosaic = torch.zeros((img.size(0),h*2,w*2))
            label_mosaic = torch.zeros(self.n_way)
            label_mosaic.data[label] = self.w
            # print(label_mosaic)
            # 采样nk_mosaic  相似的类有哪些
            mosaic_nk = self.mosaic_nk
            pseudo_nk = self.pseudo_nk
            #     抽样样本
            idxs = [0,1,2,3]
            random.shuffle(idxs)
            i = 0
            if self.mosaic_origin_num==0:
                self.mosaic_origin_num = random.randint(1,self.bound_target)
            for j in range(self.mosaic_origin_num):
                label_loc.data[idxs[i]] = 1
                r, c= int(idxs[i]/2),int(idxs[i]%2)
                img_mosaic[:,r*h:(r+1)*h,c*w:(c+1)*w] = img
                if self.boundingbox:
                    label_loc.data[:] = torch.Tensor([0.25+r*0.5, 0.25+c*0.5, 0.5, 0.5])
                if self.k_shot>1:
                    cls_ind = int(idx/(self.mosaic_k*self.k_shot))
                    img_ind = random.randint(0,self.k_shot-1)
                    # print(cls_ind*self.k_shot+img_ind)
                    # print(len(self.original_samples))
                    img_path,_ = self.original_samples[cls_ind*self.k_shot+img_ind]
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    # print(_)
                    # print(label)
                    assert _==label
                else:
                    img = self.transform(img_origin)

                i += 1
            # print(mosaic_nk[label][:])
            if self.mosaic_cls >= 3:
                mosaic_idx = random.sample(list(range(self.mosaic_cls)), 4-self.mosaic_origin_num)
            else:
                mosaic_idx = np.random.randint(0,self.mosaic_cls,4-self.mosaic_origin_num)
            # mosaic_c = random.sample(list(mosaic_nk[label][:]),3)
            # print(mosaic_idx)
            #     mosaic成图片
            for index in mosaic_idx:
                c = mosaic_nk[label][index]
                cls_name  = self.cls_Base[c]
                cls_path = os.path.join(self.root,cls_name)
                files_cls = os.listdir(cls_path)
                img_path = random.sample(files_cls,1)[0]
                img_mosaic_path = os.path.join(cls_path,img_path)
                img_m = Image.open(img_mosaic_path).convert('RGB')
                if self.transform:
                    img_m = self.transform(img_m)
                r2, c2 = int(idxs[i] / 2), int(idxs[i] % 2)
                img_mosaic[:, r2 * h:(r2 + 1) * h, c2 * w:(c2 + 1) * w] = img_m
                # print(c)
                # print(pseudo_nk[0][0][:])
                label_mosaic.data[:] +=(1-self.w)/3 * pseudo_nk[label][index][:]
                i += 1
            img = img_mosaic
            label =label_mosaic

            # if time.time()%1 > 0.99:
            #     print(label)
        # unloader = torchvision.transforms.ToPILImage()
        # image = img.cpu().clone()  # clone the tensor
        # image = image.squeeze(0)  # remove the fake batch dimension
        # image = unloader(image)
        # image.save('example.jpg')
        # print(img.shape)
        if self.Location:
            return img,label,label_loc
        return img,label

    def __len__(self):
        return len(self.samples)

    def getCls(self,path):
        cls = {}
        cla_name = open(path,"r").read().splitlines()
        # print(cla_name)
        i = 0
        for name in cla_name:
            if name != "":
                cls[i] = name
                i += 1
        # print(cls)
        return cls

    def load_image(self,root,cls,mosaic_k=1):
        k = self.k_shot
        samples = []
        idx_cls = 0
        cls_keys = cls.keys()
        # print(cls.values())
        for class_idx in cls_keys:
            i_sample = 0
            path = os.path.join(root,cls[class_idx])
            files = os.listdir(path)
            for file in files:
                if self.mode == "train_fsl":
                    i_sample += 1
                    if i_sample > k:
                        break
                img = os.path.join(path,file)
                label = class_idx
                if self.mode == "train_fsl":
                    label = idx_cls
                item = img,label
                for i in range(mosaic_k):
                    samples.append(item)
            idx_cls += 1
        # print(samples)
        return samples

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        T.Resize([84, 84]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset = MiniImageNetDataSet_mosaic("C:\PyCode\mosaic_FSL\workship\miniImageNet",mode="train_fsl",transform=transform)
    print(len(dataset))

