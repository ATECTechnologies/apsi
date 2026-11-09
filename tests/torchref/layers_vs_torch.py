import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from apsi import torch_compat as T
from apsi.stdlib import value_and_grad
from apsi import ops
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
bad=[]
def cmp(name, a, b, tol=1e-10):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape: bad.append(name); print("SHAPE", name, a.shape, b.shape); return
    e=np.max(np.abs(a-b)) if a.size else 0
    print(("ok  " if e<tol else "BAD ")+name, f"{e:.1e}")
    if not e<tol: bad.append(name)
def check(name, tm, am, inputs, train=False, out_index=None, tol=1e-10, kw=None):
    kw=kw or {}
    tm.train(train)
    tm.zero_grad(set_to_none=True)
    p=T.load_state_dict(am, am.init(0), tm.state_dict())
    if hasattr(am,'_walk'): pass
    xs=[x.clone().requires_grad_(x.is_floating_point()) for x in inputs]
    out=tm(*xs, **kw)
    ot=out[0] if isinstance(out, tuple) else out
    w=torch.randn_like(ot)
    (ot*w).sum().backward()
    xa=[x.detach().numpy() for x in inputs]
    def f(pp):
        o=am(pp,*xa,train=train,**kw)
        o=o[0] if isinstance(o, tuple) else o
        return ops.sum_(o*w.numpy())
    val, g = value_and_grad(f)(p)
    oa=am(p,*xa,train=train,**kw); oa=oa[0] if isinstance(oa,tuple) else oa
    cmp(name+" out", np.asarray(ops.raw(oa)), ot.detach().numpy(), tol)
    flat=T._flat_params(g)
    for k,v in tm.named_parameters():
        if v.grad is None: continue
        cmp(name+" grad "+k, np.asarray(ops.raw(flat[k])), v.grad.numpy(), tol*10)
    return p
check("Linear", nn.Linear(5,3), T.Linear(5,3), [torch.randn(4,5)])
check("Linear nobias 3d", nn.Linear(5,3,bias=False), T.Linear(5,3,bias=False), [torch.randn(2,4,5)])
check("Conv2d", nn.Conv2d(3,4,3), T.Conv2d(3,4,3), [torch.randn(2,3,8,8)])
check("Conv2d s2 p1 d2 g", nn.Conv2d(4,6,(3,2),stride=(2,1),padding=(1,2),dilation=(2,1),groups=2), T.Conv2d(4,6,(3,2),stride=(2,1),padding=(1,2),dilation=(2,1),groups=2), [torch.randn(2,4,9,7)])
check("Conv2d same", nn.Conv2d(2,3,5,padding="same"), T.Conv2d(2,3,5,padding="same"), [torch.randn(1,2,6,6)])
check("Conv1d", nn.Conv1d(3,5,4,stride=2,padding=1,dilation=2), T.Conv1d(3,5,4,stride=2,padding=1,dilation=2), [torch.randn(2,3,17)])
check("Conv1d depthwise", nn.Conv1d(4,4,3,groups=4,padding=1), T.Conv1d(4,4,3,groups=4,padding=1), [torch.randn(2,4,9)])
check("MaxPool2d", nn.MaxPool2d(3,2,1), T.MaxPool2d(3,2,1), [torch.randn(2,3,9,9)])
check("MaxPool2d ceil", nn.MaxPool2d(2,2,ceil_mode=True), T.MaxPool2d(2,2,ceil_mode=True), [torch.randn(1,2,7,7)])
check("AvgPool2d pad", nn.AvgPool2d(3,2,1), T.AvgPool2d(3,2,1), [torch.randn(2,3,9,8)])
check("AvgPool2d nocip", nn.AvgPool2d(3,2,1,count_include_pad=False), T.AvgPool2d(3,2,1,count_include_pad=False), [torch.randn(2,3,9,8)])
check("MaxPool1d", nn.MaxPool1d(3,2), T.MaxPool1d(3,2), [torch.randn(2,3,11)])
check("AvgPool1d", nn.AvgPool1d(2), T.AvgPool1d(2), [torch.randn(2,3,10)])
check("AdaptiveAvg2d 1", nn.AdaptiveAvgPool2d(1), T.AdaptiveAvgPool2d(1), [torch.randn(2,3,7,5)])
check("AdaptiveAvg2d 3x2", nn.AdaptiveAvgPool2d((3,2)), T.AdaptiveAvgPool2d((3,2)), [torch.randn(2,3,7,5)])
check("AdaptiveMax2d", nn.AdaptiveMaxPool2d((3,3)), T.AdaptiveMaxPool2d((3,3)), [torch.randn(2,3,7,8)])
check("AdaptiveAvg1d", nn.AdaptiveAvgPool1d(4), T.AdaptiveAvgPool1d(4), [torch.randn(2,3,10)])
for tr in (True, False):
    tm=nn.BatchNorm1d(5); tm.running_mean.normal_(); tm.running_var.uniform_(0.5,2)
    check(f"BatchNorm1d train={tr}", tm, T.BatchNorm1d(5), [torch.randn(8,5)], train=tr)
    tm=nn.BatchNorm2d(3,momentum=None); tm.running_mean.normal_()
    check(f"BatchNorm2d train={tr}", tm, T.BatchNorm2d(3,momentum=None), [torch.randn(4,3,5,5)], train=tr)
# running stats after several train steps
tm=nn.BatchNorm1d(4); am=T.BatchNorm1d(4); p=am.init(0); tm.train()
for _ in range(3):
    x=torch.randn(6,4); tm(x); am(p, x.numpy(), train=True)
cmp("BN running_mean", am.running_mean, tm.running_mean.numpy()); cmp("BN running_var", am.running_var, tm.running_var.numpy())
check("LayerNorm", nn.LayerNorm(6), T.LayerNorm(6), [torch.randn(3,4,6)])
check("LayerNorm 2d", nn.LayerNorm((4,6)), T.LayerNorm((4,6)), [torch.randn(3,4,6)])
check("GroupNorm", nn.GroupNorm(2,6), T.GroupNorm(2,6), [torch.randn(3,6,4,4)])
check("RMSNorm", nn.RMSNorm(6,eps=1e-6), T.RMSNorm(6,eps=1e-6), [torch.randn(3,4,6)])
check("Embedding pad", nn.Embedding(10,4,padding_idx=0), T.Embedding(10,4,padding_idx=0), [torch.tensor([[0,3,5],[2,0,9]])])
for bf in (True, False):
  for bi in (False, True):
    for nl in (1,2):
        check(f"LSTM bf={bf} bi={bi} L={nl}", nn.LSTM(5,7,num_layers=nl,batch_first=bf,bidirectional=bi), T.LSTM(5,7,num_layers=nl,batch_first=bf,bidirectional=bi), [torch.randn(3,4,5)])
        check(f"GRU bf={bf} bi={bi} L={nl}", nn.GRU(5,7,num_layers=nl,batch_first=bf,bidirectional=bi), T.GRU(5,7,num_layers=nl,batch_first=bf,bidirectional=bi), [torch.randn(3,4,5)])
check("RNN relu", nn.RNN(5,6,nonlinearity="relu",batch_first=True), T.RNN(5,6,nonlinearity="relu",batch_first=True), [torch.randn(2,4,5)])
# LSTM h_n, c_n
tm=nn.LSTM(4,5,num_layers=2,batch_first=True,bidirectional=True); am=T.LSTM(4,5,num_layers=2,batch_first=True,bidirectional=True); p=T.load_state_dict(am,am.init(0),tm.state_dict())
x=torch.randn(3,6,4); h0=torch.randn(4,3,5); c0=torch.randn(4,3,5)
o,(h,c)=tm(x,(h0,c0)); oa,(ha,ca)=am(p,x.numpy(),(h0.numpy(),c0.numpy()))
cmp("LSTM h_n",ha,h.detach().numpy()); cmp("LSTM c_n",ca,c.detach().numpy()); cmp("LSTM out h0",oa,o.detach().numpy())
for bf in (True, False):
    tm=nn.MultiheadAttention(8,2,batch_first=bf); am=T.MultiheadAttention(8,2,batch_first=bf)
    q=torch.randn(3,5,8); k=torch.randn(3,6,8) if bf else torch.randn(6,5,8)
    if not bf: q=torch.randn(5,3,8); k=torch.randn(6,3,8)
    check(f"MHA self bf={bf}", tm, am, [q,q,q])
    check(f"MHA cross bf={bf}", tm, am, [q,k,k])
    kpm=torch.zeros(3 if bf else 5,6,dtype=torch.bool) if False else torch.zeros(3,6,dtype=torch.bool); kpm[:,-2:]=True
    if not bf: kpm=torch.zeros(5,6,dtype=torch.bool); kpm[:,-2:]=True
    qq=torch.randn(3,5,8) if bf else torch.randn(5,3,8); kk=torch.randn(3,6,8) if bf else torch.randn(6,3,8)
    kpm=torch.zeros(3, 6, dtype=torch.bool); kpm[:,-2:]=True
    check(f"MHA kpm bf={bf}", tm, am, [qq,kk,kk], kw={"key_padding_mask":kpm})
    p=T.load_state_dict(am,am.init(0),tm.state_dict()); _,wt=tm(q,k,k); _,wa=am(p,q.numpy(),k.numpy(),k.numpy()); cmp("MHA weights",wa,wt.detach().numpy())
tm=nn.MultiheadAttention(8,2,batch_first=True,kdim=5,vdim=3); am=T.MultiheadAttention(8,2,batch_first=True,kdim=5,vdim=3)
check("MHA kdim vdim", tm, am, [torch.randn(2,4,8),torch.randn(2,6,5),torch.randn(2,6,3)])
mask=torch.triu(torch.ones(5,5,dtype=torch.bool),1)
check("MHA causal mask", nn.MultiheadAttention(8,2,batch_first=True), T.MultiheadAttention(8,2,batch_first=True), [torch.randn(2,5,8)]*3, kw={"attn_mask":mask})
for nf in (False, True):
    for act in ("relu","gelu"):
        tm=nn.TransformerEncoderLayer(8,2,16,dropout=0.0,activation=act,batch_first=True,norm_first=nf)
        check(f"EncoderLayer nf={nf} {act}", tm, T.TransformerEncoderLayer(8,2,16,dropout=0.0,activation=act,batch_first=True,norm_first=nf), [torch.randn(2,5,8)])
tm=nn.TransformerEncoder(nn.TransformerEncoderLayer(8,2,16,dropout=0.0),3,norm=nn.LayerNorm(8),enable_nested_tensor=False)
check("TransformerEncoder seqfirst", tm, T.TransformerEncoder(T.TransformerEncoderLayer(8,2,16,dropout=0.0),3,norm=T.LayerNorm(8)), [torch.randn(5,2,8)])
tm=nn.TransformerDecoderLayer(8,2,16,dropout=0.0,batch_first=True)
check("DecoderLayer", tm, T.TransformerDecoderLayer(8,2,16,dropout=0.0,batch_first=True), [torch.randn(2,4,8), torch.randn(2,6,8)])
tm=nn.Transformer(8,2,2,2,16,dropout=0.0,batch_first=True)
check("Transformer", tm, T.Transformer(8,2,2,2,16,dropout=0.0,batch_first=True), [torch.randn(2,6,8), torch.randn(2,4,8)], kw={"tgt_mask": nn.Transformer.generate_square_subsequent_mask(4)})
check("Sequential", nn.Sequential(nn.Linear(4,6),nn.GELU(),nn.Linear(6,2)), T.Sequential(T.Linear(4,6),T.module()["GELU"](),T.Linear(6,2)), [torch.randn(3,4)])
check("Bilinear", nn.Bilinear(3,4,2), T.Bilinear(3,4,2), [torch.randn(5,3), torch.randn(5,4)])
check("PReLU", nn.PReLU(3), T.PReLU(3), [torch.randn(2,3,4)])
M=T.module()
for name, kwa in [("ReLU",{}),("GELU",{}),("GELU",{"approximate":"tanh"}),("SiLU",{}),("Tanh",{}),("Sigmoid",{}),("Softmax",{"dim":-1}),("LogSoftmax",{"dim":1}),("LeakyReLU",{"negative_slope":0.2}),("ELU",{"alpha":0.7}),("SELU",{}),("CELU",{"alpha":1.3}),("Softplus",{"beta":2.0}),("Mish",{}),("Hardtanh",{}),("ReLU6",{}),("Hardswish",{}),("Hardsigmoid",{}),("Softsign",{}),("Tanhshrink",{}),("GLU",{})]:
    check("act "+name+str(kwa), getattr(nn,name)(**kwa), M[name](**kwa), [torch.randn(3,4)*3])
print("BAD:", bad)
