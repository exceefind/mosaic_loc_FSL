
import torch.nn as nn


class mlp_Net(nn.Module):
    # def __init__(self,in_dim,out_dim,hidden_dim=128,layer_num=1):
    #     super(mlp_Net, self).__init__()
    #     self.fc_layers = []
    #     if layer_num==1:
    #         hidden_dim = out_dim
    #     for i in range(layer_num):
    #         if i==0:
    #             fc = nn.Linear(in_dim,hidden_dim)
    #         elif i<layer_num-1:
    #             fc = nn.Linear(hidden_dim,hidden_dim)
    #         else:
    #             fc = nn.Linear(hidden_dim,out_dim)
    #         self.fc_layers.append(fc)
    #     self.relu = nn.ReLU(inplace=True)
    #
    # def forward(self,x):
    #     for i in range(len(self.fc_layers)):
    #         if i != 0:
    #             x = self.relu(x)
    #         x  = self.fc_layers[i](x)
    #
    #     return x

    def __init__(self,in_dim,out_dim,hidden_dim=128,layer_num=1):
        super(mlp_Net, self).__init__()
        self.fc = nn.Linear(in_dim,out_dim)

    def forward(self,x):

        return self.fc(x)
