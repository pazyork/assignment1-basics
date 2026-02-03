import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        # self.RotaryPositionMatrix=torch
        # self.register_buffer(persistent=False)
        # max_seq_len,d_k,d_k
        rotary_matrix=torch.zeros(max_seq_len,d_k,d_k,dtype=torch.float32,device=device)
        half_dim=d_k//2
        
        # max_seq_len,1
        positions=torch.arange(max_seq_len,device=device).unsqueeze(-1)
        # half_dim
        dims=torch.arange(half_dim,device=device)
        # max_seq_len,half_dim
        theta_iks=positions*(self.theta**(-2*dims/d_k))
        # max_seq_len,half_dim
        cos_thetas=torch.cos(theta_iks)
        sin_thetas=torch.sin(theta_iks)
        
        for k in range(half_dim):
            rotary_matrix[:,2*k,2*k]=cos_thetas[:,k]
            rotary_matrix[:,2*k+1,2*k+1]=cos_thetas[:,k]
            rotary_matrix[:,2*k,2*k+1]=-sin_thetas[:,k]
            rotary_matrix[:,2*k+1,2*k]=sin_thetas[:,k]
        
        if d_k%2==1:
            rotary_matrix[:,d_k-1,d_k-1]=1.0
        # 注册后，即可self.rotary_matrix这也适用
        self.register_buffer('rotary_matrix',rotary_matrix,False)
        
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        ## x (...,seq_len,d_k)
        ## token_positions  (...,seq_len)
        ## rotary_matrix (max_seq_len,d_k,d_k)
        ## vector_rotary_matrix (...,seq_len,d_k,d_k)
        # print('x.shape,token_positions.shape:',x.shape,token_positions.shape)
        self.vector_rotary_matrix=self.rotary_matrix[token_positions]
        result=torch.einsum('...qij,...qj->...qi',self.vector_rotary_matrix,x)
        # print('result.shape',result.shape)
        return result