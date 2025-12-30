import torch
from torch import nn

class UtilTool(nn.Module):
    
    @classmethod
    def softmax(cls, in_features:torch.Tensor,dim:int):
        ## x,1,z
        in_features_max=in_features.max(dim=dim,keepdim=True).values
        ## x,y,z
        features_base=torch.exp(in_features-in_features_max)
        ## x,y,z
        features_sum=features_base.sum(dim=dim,keepdim=True)
        return (features_base/features_sum)