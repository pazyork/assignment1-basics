import torch
from torch import nn

class Linear(torch.nn.Module):
    def __init__(self,out_features,in_features,device=None,dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(size=[out_features,in_features],device=device,dtype=dtype))
        std=2/(in_features+out_features)
        torch.nn.init.trunc_normal_(tensor=self.weight,mean=0,std=std,a=-3*std,b=3*std)
        
    def forward(self,x:torch.Tensor):
        y=torch.einsum('oi,...i->...o',self.weight,x)
        return y