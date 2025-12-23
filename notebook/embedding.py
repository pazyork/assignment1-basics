import torch
from torch import nn


# uv run pytest -k test_embedding
class Embedding(nn.Module):
    def __init__(self,num_embedding,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(size=[num_embedding,embedding_dim],device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(tensor=self.weight,mean=0,std=1,a=-3,b=3)
        
    def forward(self,token_ids:torch.Tensor):
        # weight：v * k
        # token_ids(可能的shape)：(b,s)、(s)
        ## self.weight ：v * k
        ##self.weight[token_ids] ：(b,s,k)
        return self.weight[token_ids]
    