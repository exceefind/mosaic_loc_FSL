
import torch.nn as nn
import torch


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

    def __init__(self,in_dim,out_dim,hidden_dim=2048,grid=2,layer_num=1,device='cpu'):
        super(mlp_Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, int(in_dim/4))
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim*2,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,int(hidden_dim/4))
        self.fc5 = nn.Linear(int(hidden_dim/4),out_dim)
        self.grid = 2
        self.device = device

    def forward(self,x,y):
        x = self.act(self.fc1(x)).reshape(x.shape[0],-1,1)
        # y = self.act(self.fc2(y))
        x = x.expand(x.shape[0],-1,self.grid**2 )
        y = y.reshape(x.shape[0],-1,self.grid**2 )
        xy = torch.sqrt(torch.mul(torch.sum(torch.pow(x,2),1),torch.sum(torch.pow(y,2),1)))
        res = torch.sum(torch.mul(x,y),dim=1)/xy
        return res
        # y_chunk = torch.chunk(y,self.grid**2,dim=1)
        # out =torch.zeros(y.shape[0],self.grid**2).to(self.device)
        # for i in range(self.grid**2):
        #     res = torch.cat([y_chunk[i],x],dim=1)
        #     # print(self.fc3(res)[:,0].shape)
        #     # print(out[:,1].shape)
        #     out[:,i] = self.fc3(res)[:,0]
        # tensor_cat = torch.cat([x,y],dim=1)
        # # out = tensor_cat
        # out = self.act(self.fc3(tensor_cat))
        # out = self.fc4(out)
        # return self.fc5(out)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)  # normal: mean=0, std=1

