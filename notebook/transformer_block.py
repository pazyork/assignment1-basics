import torch
from torch import nn
from einops import rearrange,repeat,reduce
from jaxtyping import Bool, Float, Int

# uv run pytest -k test_multihead_self_attention
class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 max_seq_len,
                 theta:float
                 ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        from notebook.multi_head_attention import MultiHeadAttention
        self.attn=MultiHeadAttention(self.d_model,self.num_heads,self.max_seq_len,self.theta)
        from notebook.swiglu import SwiGLU
        self.ffn=SwiGLU(self.d_model,self.d_ff)
        from notebook.rmsnorm import RMSNorm
        self.ln1=RMSNorm(self.d_model)
        self.ln2=RMSNorm(self.d_model)
    
    def forward(self,
                in_features: Float[torch.Tensor, " batch sequence_length d_model"],
                ):
        ln1_output = self.ln1.forward(in_features)
        attn_output = self.attn.forward(ln1_output,True)
        
        ln2_input = in_features + attn_output
        ln2_output = self.ln2.forward(ln2_input)
        ffn_output = self.ffn.forward(ln2_output)
        result = ln2_input+ffn_output
        
        return result