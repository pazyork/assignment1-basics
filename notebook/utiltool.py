import torch
from torch import nn

class UtilTool(nn.Module):
    
    @classmethod
    def softmax(cls, in_features:torch.Tensor,dim:int):
        in_dtype=in_features.dtype
        in_features=in_features.to(torch.float64)
        print('in_features',in_features)
        ## x,y,z
        features_base=torch.exp(in_features)
        print('features_base',features_base)
        ## x,1,z
        features_sum=features_base.sum(dim=dim,keepdim=True)
        print('features_sum',features_sum)
        return (features_base/features_sum).to(in_dtype)