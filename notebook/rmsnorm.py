from numpy import float32
import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int,eps:torch.float32=1e-5, device=None, dtype=torch.float32) -> None:
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.weight=torch.nn.Parameter(torch.ones(d_model,dtype=dtype,device=device))
        
    def _rms(self,x:torch.Tensor)->torch.Tensor:
        # return torch.sqrt(torch.pow(x,2).mean(-1)+self.eps).unsqueeze(-1)
        return torch.sqrt(torch.pow(x,2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x:torch.Tensor):
        # x：(b,l,d)
        # weigth：(d)
        x=x.to(torch.float32)
        in_dtype=x.dtype
        rms=self._rms(x)
        result=self.weight*(x/rms)
        return result.to(in_dtype)