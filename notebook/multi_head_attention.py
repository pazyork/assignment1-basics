from turtle import forward
import torch
from torch import nn
from einops import rearrange,repeat,reduce



# uv run pytest -k test_multihead_self_attention
class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int=None,theta:float=None):
        super().__init__()
        self.d_model=d_model
        # LLM场景内，qkv的tez
        self.num_heads=num_heads
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.d_in = self.d_model
        self.d_q=self.d_model//self.num_heads
        self.d_k=self.d_model//self.num_heads
        self.d_v=self.d_model//self.num_heads
        
        self.q_proj_weight = nn.Parameter(torch.empty([self.d_q*self.num_heads, self.d_in]))
        self.k_proj_weight = nn.Parameter(torch.empty([self.d_k*self.num_heads, self.d_in]))
        self.v_proj_weight = nn.Parameter(torch.empty([self.d_v*self.num_heads, self.d_in]))
        self.o_proj_weight = nn.Parameter(torch.empty([self.d_model,self.d_v*self.num_heads]))
        
        torch.nn.init.trunc_normal_(tensor=self.q_proj_weight,mean=0,std=1,a=-3,b=3)
        torch.nn.init.trunc_normal_(tensor=self.k_proj_weight,mean=0,std=1,a=-3,b=3)
        torch.nn.init.trunc_normal_(tensor=self.v_proj_weight,mean=0,std=1,a=-3,b=3)
        torch.nn.init.trunc_normal_(tensor=self.o_proj_weight,mean=0,std=1,a=-3,b=3)
        
        from notebook.rotary_positional_embedding import RotaryPositionalEmbedding
        
        if theta:
            self.rpe = RotaryPositionalEmbedding(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len)
        
        
        
        
    def forward(self,in_feature:torch.Tensor,is_rope:bool=False):
        q = torch.einsum('qi,...si->...sq',self.q_proj_weight,in_feature)
        q = rearrange(q, '... s (h q)->... h s q', h=self.num_heads)
        k = torch.einsum('ki,...si->...sk',self.k_proj_weight,in_feature)
        k = rearrange(k, '... s (h k)->... h s k', h=self.num_heads)
        v = torch.einsum('vi,...si->...sv',self.v_proj_weight,in_feature)
        v = rearrange(v, '... s (h v)->... h s v', h=self.num_heads)
        
        if is_rope:
            token_positions = torch.ones_like(q[...,0],dtype=torch.int16)
            token_positions = torch.cumsum(token_positions,dim=-1)
            q=self.rpe.forward(q,token_positions)
            k=self.rpe.forward(k,token_positions)
        
        len_seq = in_feature.shape[-2]
        mask = torch.full((len_seq,len_seq),True)
        mask = torch.tril(mask,diagonal=0)

        # 第一步：计算 Q @ K^T
        scale = torch.einsum('...hid,...hjd->...hij', q, k)
        scale = scale * torch.pow(torch.tensor(self.d_k), -0.5)

        # 第二步：应用 mask
        # 用 masked_fill，它支持广播
        scale = scale.masked_fill(~mask, float('-inf'))

        from .utiltool import UtilTool
        scale_softmax = UtilTool.softmax(scale,-1)
        o=torch.einsum('...hij,...hjv->...hiv',scale_softmax,v)
        print("o.shape=",o.shape)
        o=rearrange(o,'... h s v->... s (h v)')
        print("o.shape=",o.shape)
        print("self.o_proj_weight=",self.o_proj_weight.shape)
        o=torch.einsum('ov,... sv->...so',self.o_proj_weight,o)
        return o