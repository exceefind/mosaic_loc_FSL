import torch.nn
import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self,out="feature",Location = False,num_cls=80, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()
        z_dim = 64
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            torch.nn.AdaptiveAvgPool2d(2)
        )
        self.out = out
        self.Location = Location
        dim_out = 5
        self.fc = nn.Linear(z_dim*2*2,num_cls)
        if self.Location:
            self.fc_loc = nn.Linear(z_dim*2*2,5)

    def forward(self, x):
        x = self.encoder(x)
        if self.out=="feature":
            return x.view(x.size(0), -1)
        else:
            if self.Location:
                return self.fc(x.view(x.size(0), -1)),self.fc_loc(x.view(x.size(0), -1))
            # print(x.view(x.size(0), -1).shape)
            # print(self.fc(x.view(x.size(0), -1)))
            return self.fc(x.view(x.size(0), -1))
