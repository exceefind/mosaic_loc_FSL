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
    def __init__(self,Location = False,num_classes=80, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()
        z_dim = 64
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            torch.nn.AdaptiveAvgPool2d(1)
        )
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
