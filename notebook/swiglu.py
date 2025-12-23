import torch

class SwiGLU(torch.nn.Module):
    def __init__(self,d_model:int,d_ff:int):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        ## 构造参数
        self.w1_weight=torch.nn.Parameter(torch.empty([d_ff,d_model],dtype=torch.float32))
        self.w2_weight=torch.nn.Parameter(torch.empty([d_model,d_ff],dtype=torch.float32))
        self.w3_weight=torch.nn.Parameter(torch.empty([d_ff,d_model],dtype=torch.float32))
        ## 怎么初始化还没敲定
        torch.nn.init.trunc_normal_(tensor=self.w1_weight,mean=0,std=1,a=-3,b=3)
        torch.nn.init.trunc_normal_(tensor=self.w2_weight,mean=0,std=1,a=-3,b=3)
        torch.nn.init.trunc_normal_(tensor=self.w3_weight,mean=0,std=1,a=-3,b=3)
    
    def _sigmod(self,x:torch.Tensor):
        # ...f->...f
        return (1+torch.exp(-1*x)).pow(-1)
    
    def _SiLU(self,x:torch.Tensor):
        # ...f*...f
        return x*self._sigmod(x)
    
    def forward(self,x:torch.Tensor):
        from torch import einsum
        in_type=x.dtype
        x=x.to(torch.float32)
        # einsum版本
        # x1=einsum('fd,...d->...f',self.w1_weight,x)
        # x3=(einsum('fd,...d->...f',self.w3_weight,x))
        # x13=self._SiLU(x1)*x3
        # result=einsum('df,...f->...d',self.w2_weight,x13)
        
        # 常规版本
        x1=x@self.w1_weight.T
        x3=x@self.w3_weight.T
        x13=self._SiLU(x1)*x3
        result=x13@self.w2_weight.T
        
        return result.to(in_type)