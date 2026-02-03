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
    
    @classmethod
    def scaled_dot_product_attention(cls,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask):
        # print('input shape',Q.shape,K.shape,V.shape,mask.shape)
        relate=torch.einsum('...qd,...kd->...qk',Q,K)*Q.shape[-1]**(-0.5)
        relate[~mask]=-torch.inf
        # print('relate',relate)
        # print('relate_masked',relate)
        relate_masked_softmax=cls.softmax(relate,-1)
        # print('relate_masked_softmax',relate_masked_softmax)
        result=torch.einsum('...qk,...kd->...qd',relate_masked_softmax,V)
        # print('result.shape',result.shape)
        return result
