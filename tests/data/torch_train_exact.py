# A PyTorch training script used to check that its Apsi conversion trains identically (same weights in, same
# weights out). It uses batch norm, an LSTM, a scheduler, AdamW, gradient clipping and label smoothing.
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self, d_in, hidden, classes):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, classes))

    def forward(self, x):
        b, t, d = x.shape
        h = self.proj(x)
        h = self.bn(h.reshape(b * t, -1)).reshape(b, t, -1)
        h = F.gelu(h)
        seq, (h_n, c_n) = self.rnn(h)
        a, _ = self.attn(seq, seq, seq, need_weights=False)
        return self.head(a.mean(dim=1) + h_n[-1])


torch.manual_seed(0)
X = load_file("data.safetensors")["X"]
y = load_file("data.safetensors")["y"].long()
model = Model(5, 8, 3)
model.load_state_dict(load_file("init.safetensors"))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
losses = []
for epoch in range(6):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sched.step()
model.eval()
with torch.no_grad():
    final = model(X)
save_file(model.state_dict(), "final.safetensors")
