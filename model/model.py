import torch.nn
import torch.nn as nn
from conv4 import ConvNet
from resnet12 import resnet12
from resnet import *



class Net(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self,Location = False,net="conv64",num_classes=80, x_dim=3, hid_dim=64, z_dim=64):
        super(Net, self).__init__()
        if net == "conv64":
            z_dim = 64
            self.encoder = ConvNet()
        elif net == "resnet12":
            self.encoder = resnet12()
        elif net == "resnet18":
            self.encoder = resnet18()
        self.Location = Location
        dim_out = 5
        self.fc = nn.Linear(z_dim,num_classes)
        if self.Location:
            self.fc_loc = nn.Linear(z_dim,4)

    def forward(self, x ,is_feat=False ):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        feat = x
        x = self.fc(x)

        if self.Location:
            if is_feat:
                return feat, x,self.fc_loc(feat)
            else:
                return x,self.fc_loc(feat)
        # print(x.view(x.size(0), -1).shape)
        # print(self.fc(x.view(x.size(0), -1)))
        else:
            if is_feat:
                return feat, x
            else:
                return x
