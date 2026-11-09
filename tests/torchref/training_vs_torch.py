import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from apsi import torch_compat as T, ops
from apsi.stdlib import value_and_grad
from apsi.values import Record
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
bad=[]
def cmp(name,a,b,tol=1e-10):
    a=np.asarray(a,float); b=np.asarray(b,float); e=np.max(np.abs(a-b)) if a.size else 0
    if not (a.shape==b.shape and e<tol): bad.append(name); print("BAD",name,a.shape,b.shape,e)
x=torch.randn(6,5,requires_grad=True); y=torch.randint(0,5,(6,)); y[2]=-100
w=torch.rand(5)+0.5
for kw in [{}, {"weight":w}, {"ignore_index":-100}, {"label_smoothing":0.1}, {"weight":w,"label_smoothing":0.2}, {"reduction":"sum"}, {"reduction":"none"}, {"reduction":"none","label_smoothing":0.1,"weight":w}]:
    yy=y if "ignore_index" in kw or "weight" in kw or "label_smoothing" in kw else y.clamp(min=0)
    kt={k:v for k,v in kw.items()}; ka={k:(v.numpy() if torch.is_tensor(v) else v) for k,v in kw.items()}
    t=F.cross_entropy(x,yy,**kt); a=T.cross_entropy(x.detach().numpy(), yy.numpy(), **ka)
    cmp(f"ce {list(kw)}", np.asarray(ops.raw(a)), t.detach().numpy())
    g=torch.autograd.grad(t.sum(),x)[0]; ga=value_and_grad(lambda z: ops.sum_(T.cross_entropy(z, yy.numpy(), **ka)))(x.detach().numpy())[1]
    cmp(f"ce grad {list(kw)}", ga, g.numpy())
pt=F.softmax(torch.randn(6,5),-1)
cmp("ce probs", ops.raw(T.cross_entropy(x.detach().numpy(), pt.numpy(), label_smoothing=0.1)), F.cross_entropy(x,pt,label_smoothing=0.1).detach().numpy())
x4=torch.randn(2,5,3,4); y4=torch.randint(0,5,(2,3,4))
cmp("ce 4d", ops.raw(T.cross_entropy(x4.numpy(), y4.numpy())), F.cross_entropy(x4,y4).numpy())
cmp("ce 4d none", ops.raw(T.cross_entropy(x4.numpy(), y4.numpy(), reduction="none")), F.cross_entropy(x4,y4,reduction="none").numpy())
lp=F.log_softmax(x,-1)
cmp("nll w", ops.raw(T.nll_loss(lp.detach().numpy(), y.numpy(), weight=w.numpy())), F.nll_loss(lp,y,weight=w).detach().numpy())
a=torch.randn(7,3); b=torch.randn(7,3); bt=(torch.rand(7,3)>.5).double(); pw=torch.rand(3)+.5
for nm,ft,fa in [("mse",F.mse_loss,T.mse_loss),("l1",F.l1_loss,T.l1_loss),("smoothl1",lambda u,v: F.smooth_l1_loss(u,v,beta=0.5),lambda u,v: T.smooth_l1_loss(u,v,beta=0.5)),("huber",lambda u,v:F.huber_loss(u,v,delta=0.7),lambda u,v:T.huber_loss(u,v,delta=0.7))]:
    cmp(nm, ops.raw(fa(a.numpy(),b.numpy())), ft(a,b).numpy())
cmp("bcel", ops.raw(T.binary_cross_entropy_with_logits(a.numpy(),bt.numpy(),pos_weight=pw.numpy())), F.binary_cross_entropy_with_logits(a,bt,pos_weight=pw).numpy())
cmp("bcel w sum", ops.raw(T.binary_cross_entropy_with_logits(a.numpy()*30,bt.numpy(),weight=pw.numpy(),reduction="sum")), F.binary_cross_entropy_with_logits(a*30,bt,weight=pw,reduction="sum").numpy())
cmp("bce", ops.raw(T.binary_cross_entropy(torch.sigmoid(a).numpy(),bt.numpy())), F.binary_cross_entropy(torch.sigmoid(a),bt).numpy())
cmp("kl", ops.raw(T.kl_div(F.log_softmax(a,-1).numpy(), F.softmax(b,-1).numpy(), reduction="batchmean")), F.kl_div(F.log_softmax(a,-1), F.softmax(b,-1), reduction="batchmean").numpy())
# optimizers: run 25 steps on a quadratic-ish objective with the same gradients
def run(make_t, make_a, steps=25, sched=None):
    torch.manual_seed(3)
    W=torch.randn(4,3,requires_grad=True); bvec=torch.randn(3,requires_grad=True)
    P=Record(W=W.detach().numpy().copy(), b=bvec.detach().numpy().copy())
    opt=make_t([W,bvec]); oa=make_a()
    st=sa=None
    if sched: st=sched[0](opt); sa=sched[1](oa)
    X=torch.randn(10,4); Y=torch.randn(10,3)
    for i in range(steps):
        opt.zero_grad(); loss=((torch.tanh(X@W+bvec)-Y)**2).mean(); loss.backward(); opt.step()
        g=value_and_grad(lambda p: ops.mean((ops.tanh(X.numpy()@p.W+p.b)-Y.numpy())**2))(P)[1]
        P=oa.step(P,g)
        if st: st.step(); sa.step()
    return np.r_[W.detach().numpy().ravel(), bvec.detach().numpy()], np.r_[np.asarray(P.W).ravel(), P.b], (opt.param_groups[0]["lr"], oa.lr)
O=torch.optim
cases=[("SGD",lambda p:O.SGD(p,lr=0.1),lambda:T.SGD(lr=0.1)),
 ("SGD mom nest wd",lambda p:O.SGD(p,lr=0.05,momentum=0.9,nesterov=True,weight_decay=1e-2),lambda:T.SGD(lr=0.05,momentum=0.9,nesterov=True,weight_decay=1e-2)),
 ("SGD damp",lambda p:O.SGD(p,lr=0.05,momentum=0.9,dampening=0.3),lambda:T.SGD(lr=0.05,momentum=0.9,dampening=0.3)),
 ("Adam",lambda p:O.Adam(p,lr=1e-2),lambda:T.Adam(lr=1e-2)),
 ("Adam wd ams",lambda p:O.Adam(p,lr=1e-2,weight_decay=0.1,amsgrad=True,betas=(0.8,0.99)),lambda:T.Adam(lr=1e-2,weight_decay=0.1,amsgrad=True,betas=(0.8,0.99))),
 ("AdamW",lambda p:O.AdamW(p,lr=1e-2),lambda:T.AdamW(lr=1e-2)),
 ("AdamW ams",lambda p:O.AdamW(p,lr=1e-2,weight_decay=0.3,amsgrad=True),lambda:T.AdamW(lr=1e-2,weight_decay=0.3,amsgrad=True)),
 ("RMSprop",lambda p:O.RMSprop(p,lr=1e-2),lambda:T.RMSprop(lr=1e-2)),
 ("RMSprop centered mom",lambda p:O.RMSprop(p,lr=1e-2,centered=True,momentum=0.9,weight_decay=0.01),lambda:T.RMSprop(lr=1e-2,centered=True,momentum=0.9,weight_decay=0.01)),
 ("Adagrad",lambda p:O.Adagrad(p,lr=0.1,lr_decay=0.01,initial_accumulator_value=0.1),lambda:T.Adagrad(lr=0.1,lr_decay=0.01,initial_accumulator_value=0.1)),
 ("Adamax",lambda p:O.Adamax(p,lr=0.01),lambda:T.Adamax(lr=0.01)),
 ("NAdam",lambda p:O.NAdam(p,lr=0.01),lambda:T.NAdam(lr=0.01)),
 ("RAdam",lambda p:O.RAdam(p,lr=0.01),lambda:T.RAdam(lr=0.01))]
for name,mt,ma in cases:
    a,b,_=run(mt,ma); cmp("opt "+name,b,a,1e-9)
S=O.lr_scheduler
scheds=[("StepLR",lambda o:S.StepLR(o,5,0.5),lambda o:T.StepLR(o,5,0.5)),
 ("MultiStepLR",lambda o:S.MultiStepLR(o,[3,7,7,12],0.3),lambda o:T.MultiStepLR(o,[3,7,7,12],0.3)),
 ("ExponentialLR",lambda o:S.ExponentialLR(o,0.9),lambda o:T.ExponentialLR(o,0.9)),
 ("CosineAnnealingLR",lambda o:S.CosineAnnealingLR(o,7,1e-4),lambda o:T.CosineAnnealingLR(o,7,1e-4)),
 ("WarmRestarts",lambda o:S.CosineAnnealingWarmRestarts(o,4,2,1e-4),lambda o:T.CosineAnnealingWarmRestarts(o,4,2,1e-4)),
 ("LinearLR",lambda o:S.LinearLR(o,0.2,1.0,8),lambda o:T.LinearLR(o,0.2,1.0,8)),
 ("ConstantLR",lambda o:S.ConstantLR(o,0.5,6),lambda o:T.ConstantLR(o,0.5,6)),
 ("LambdaLR",lambda o:S.LambdaLR(o,lambda e:0.95**e),lambda o:T.LambdaLR(o,lambda e:0.95**e)),
 ("MultiplicativeLR",lambda o:S.MultiplicativeLR(o,lambda e:0.97),lambda o:T.MultiplicativeLR(o,lambda e:0.97)),
 ("PolynomialLR",lambda o:S.PolynomialLR(o,10,2.0),lambda o:T.PolynomialLR(o,10,2.0)),
 ("OneCycleLR",lambda o:S.OneCycleLR(o,0.05,total_steps=30),lambda o:T.OneCycleLR(o,0.05,total_steps=30)),
 ("OneCycle 3ph lin",lambda o:S.OneCycleLR(o,0.05,total_steps=30,three_phase=True,anneal_strategy="linear"),lambda o:T.OneCycleLR(o,0.05,total_steps=30,three_phase=True,anneal_strategy="linear")),
 ("CyclicLR",lambda o:S.CyclicLR(o,0.001,0.05,step_size_up=4,cycle_momentum=False),lambda o:T.CyclicLR(o,0.001,0.05,step_size_up=4,cycle_momentum=False)),
]
for name, st, sa in scheds:
    mt=(lambda p:O.SGD(p,lr=0.05,momentum=0.9)) if name.startswith("Cyclic") else (lambda p:O.Adam(p,lr=1e-2))
    ma=(lambda:T.SGD(lr=0.05,momentum=0.9)) if name.startswith("Cyclic") else (lambda:T.Adam(lr=1e-2))
    a,b,(lt,la)=run(mt,ma,25 if "OneCycle" not in name else 29,(st,sa)); cmp("sched "+name,b,a,1e-9); cmp("sched lr "+name,[la],[lt],1e-12)
# ReduceLROnPlateau
W=torch.zeros(1,requires_grad=True); ot=O.SGD([W],lr=1.0); oa=T.SGD(lr=1.0); rt=S.ReduceLROnPlateau(ot,patience=2,factor=0.5); ra=T.ReduceLROnPlateau(oa,patience=2,factor=0.5)
for m in [5,4,4,4,4,4,3.9,3.9,3.9,3.9,3.9,2,2,2,2,2]:
    rt.step(m); ra.step(m)
cmp("plateau", [oa.lr], [ot.param_groups[0]["lr"]])
print("BAD:", bad)
