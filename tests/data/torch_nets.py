import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Block(nn.Module):
    """Pre-norm residual MLP block."""
    def __init__(self, d, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(self.norm(x))))
        return x + self.scale * h


class Net(nn.Module):
    def __init__(self, d_in, d, n_blocks, n_out):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([Block(d, 2 * d) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d, n_out))
        self.temperature = 2.0

    def features(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x):
        return self.head(self.features(x)) / self.temperature


class CNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 4 * 4, classes)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(self.drop(x))


class SeqClassifier(nn.Module):
    def __init__(self, vocab, d, hidden, classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lstm = nn.LSTM(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, classes)

    def forward(self, tokens):
        e = self.emb(tokens)
        seq, (h, c) = self.lstm(e)
        return self.out(seq[:, -1, :])


torch.manual_seed(0)
X = torch.randn(512, 6)
y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).long()
model = Net(6, 16, 2, 2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
for epoch in range(15):
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * xb.size(0)
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss {total / len(X):.4f}")
model.eval()
with torch.no_grad():
    pred = model(X).argmax(dim=1)
    acc = (pred == y).float().mean().item()
print(f"accuracy {acc:.3f}")
n_params = sum(p.numel() for p in model.parameters())
print("parameters", n_params)
torch.save(model.state_dict(), "net.pt")
