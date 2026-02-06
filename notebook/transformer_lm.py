import torch
from torch import embedding, nn
from notebook.embedding import Embedding
from notebook.transformer_block import TransformerBlock
from notebook.rmsnorm import RMSNorm
from notebook.linear import Linear
from notebook.utiltool import UtilTool

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 d_ff:int,
                 rope_theta:float
                 ):
        super().__init__()
        self.vocab_size=vocab_size
        self.max_seq_len=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rope_theta=rope_theta
        
        self.embedding=Embedding(
            num_embedding=self.vocab_size,
            embedding_dim=self.d_model
        )
        self.transformer_blocks=torch.nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                theta=self.rope_theta
            )
            for _ in range(num_layers)
        ])
        self.norm=RMSNorm(d_model=self.d_model)
        self.linear=Linear(
            out_features=self.vocab_size,
            in_features=self.d_model
        )
    
    def forward(self,input_feature):
        hidden_state=self.embedding.forward(input_feature)
        for idx in range(len(self.transformer_blocks)):
            hidden_state=self.transformer_blocks[idx].forward(hidden_state)
        hidden_state=self.norm.forward(hidden_state)
        hidden_state=self.linear.forward(hidden_state)
        # output=UtilTool.softmax(hidden_state,-1)
        return hidden_state