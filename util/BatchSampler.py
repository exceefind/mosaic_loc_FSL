# coding=utf-8
import numpy as np
import torch

class BatchSampler(object):
    def __init__(self, labels, classes_per_it, num_samples, iterations):
        super(BatchSampler, self).__init__()




    def __iter__(self):
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            # randperm 随机打乱数字 并抽取出前cpi个
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            # print(len(self.classes))
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations