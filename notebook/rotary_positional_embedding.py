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
        max_dim_idx=int(d_k/2)
        for i in range(max_seq_len):
            for k in range(max_dim_idx):
                theta_ik=torch.tensor(i*self.theta**((-2*k)/d_k))
                cos_theta=torch.cos(theta_ik)
                sin_theta=torch.sin(theta_ik)
                rotary_matrix[i,2*k,2*k]=cos_theta
                rotary_matrix[i,2*k,2*k+1]=-sin_theta
                rotary_matrix[i,2*k+1,2*k]=sin_theta
                rotary_matrix[i,2*k+1,2*k+1]=cos_theta
            if d_k%2==1:
                k=max_dim_idx
                theta_ik=torch.tensor(i*self.theta**((-2*k)/d_k))
                cos_theta=torch.cos(theta_ik)
                rotary_matrix[i,d_k-1,d_k-1]=cos_theta
                
        self.register_buffer('rotary_matrix',rotary_matrix,False)
        
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        ## x (...,seq_len,d_k)
        ## token_positions  (...,seq_len)
        ## rotary_matrix (max_seq_len,d_k,d_k)
        ## vector_rotary_matrix (...,seq_len,d_k,d_k)
        print('x.shape,token_positions.shape:',x.shape,token_positions.shape)
        self.vector_rotary_matrix=self.rotary_matrix[token_positions]
        result=torch.einsum('...qij,...qj->...qi',self.vector_rotary_matrix,x)
        print('result.shape',result.shape)
        return result