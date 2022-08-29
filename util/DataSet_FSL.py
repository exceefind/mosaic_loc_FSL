import os.path
import random

import torch
import torch.utils.data as data
from PIL import Image


class MiniImageNetDataSet(data.Dataset):
    def __init__(self,root, mode="train_fsl", mosaic_cls = 1, n_way = 5,k_shot = 5, mosaic = True,mosaic_k=2, transform=None, transform_mosaic=None):
        self.root = root
        self.transform = transform
        self.transform_mosaic = transform_mosaic
        # self.root = os.path.join(root,mode)
        train_mode,FSL_mode = mode.split("_")
        self.mode = mode
        self.root = os.path.join(root,train_mode)
        self.FSL_mode = FSL_mode
        self.mosaic = mosaic
        self.mosaic_k = mosaic_k
        self.mosaic_cls = mosaic_cls
        self.n_way =n_way
        self.k_shot =k_shot
        if FSL_mode == "fsl":
            cls = self.getCls(os.path.join(root,"spilt","Fewshot.txt"))
            if mosaic and train_mode== "train" :
                self.samples =self.load_image(self.root, cls)*mosaic_k
                cls_Base = self.getCls(os.path.join(root, "spilt", "BaseSet.txt"))
                self.cls_Base = cls_Base
                # self.samples_Base = self.load_image(self.root, cls_Base)
            else:
                self.samples = self.load_image(self.root, cls)
        else:
            cls = self.getCls(os.path.join("workship", "spilt", "BaseSet.txt"))
            if mosaic:
                self.origin_samples = self.load_image(self.root, cls)
                self.samples = self.load_image(self.root, cls,mosaic_k)
            else:
                self.samples = self.load_image(self.root,cls)
        self.cls = cls

    def __getitem__(self, idx):
        img_path,label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.mosaic and idx%self.mosaic_k!=0:
            img = self.mosaicImg(img,idx,label)
        elif self.transform :
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.samples)

    def getCls(self,path):
        cls = {}
        cla_name = open(path,"r").read().splitlines()
        i = 0
        for name in cla_name:
            if name != "":
                cls[i] = name
                i += 1

        return cls

    def load_image(self,root,cls,mosaic_k=1):
        n = self.n_way
        k = self.k_shot
        samples = []
        idx_cls = 0
        cls_keys = cls.keys()
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
                for j in range(mosaic_k):
                    samples.append(item)
            idx_cls += 1
        return samples

    def mosaicImg(self,img,idx,label):
        img = self.transform_mosaic(img)
        h = img.size(1)
        w = img.size(2)
        img_mosaic = torch.zeros((img.size(0), h * 2, w * 2))
        #     抽样样本
        idxs = [0, 1, 2, 3]
        random.shuffle(idxs)
        i = 0
        r, c = int(idxs[i] / 2), int(idxs[i] % 2)
        img_mosaic[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        i += 1
        #     mosaic成图片
        while i<4:
            nspc = len(self.origin_samples) / len(self.cls)
            # print(nspc)
            cls_idx = int(idx/(self.mosaic_k*nspc))
            # num_sample_per_cls
            # print(cls_idx)
            # print((cls_idx+1)*nspc)
            img_idx = random.randint(cls_idx*nspc,(cls_idx+1)*nspc-1)
            img_path,label_m= self.origin_samples[img_idx]
            img_m = Image.open(img_path).convert('RGB')
            # print(label)
            # print(label_m)
            assert label_m==label
            if self.transform:
                img_m = self.transform_mosaic(img_m)
            r2, c2 = int(idxs[i] / 2), int(idxs[i] % 2)
            img_mosaic[:, r2 * h:(r2 + 1) * h, c2 * w:(c2 + 1) * w] = img_m
            i += 1
        return img_mosaic
if __name__ == '__main__':
    dataset = MiniImageNetDataSet("C:\PyCode\mosaic_FSL\workship\miniImageNet",mode="train_fsl")
    print(len(dataset))

