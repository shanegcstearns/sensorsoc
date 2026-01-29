# crudemodel.py
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HISTORY = 10             # window length (timesteps)
BASELINE_ALPHA = 1 / 256     # baseline adaptation speed
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 64

df = pd.read_csv("sd_out_clean.csv")  # columns: time, hr, ss

# Basic cleanup
df = df.dropna(subset=["time", "hr", "ss"]).copy()
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
df = df.dropna(subset=["time", "hr"]).copy()

# Map sleep states
label_map = {"W": 0, "N1": 1, "N2": 1, "N3": 1, "R": 2}
df = df[df["ss"].isin(label_map.keys())].copy()
df["label"] = df["ss"].map(label_map).astype(int)

time_arr = df["time"].to_numpy()
switch = np.zeros(len(df), dtype=bool)
switch[1:] = time_arr[1:] < time_arr[:-1]
df["subject_id"] = np.cumsum(switch).astype(int)

baseline = np.zeros(len(df), dtype=np.float32)
b = None
prev_sid = None
for i, (sid, hr) in enumerate(zip(df["subject_id"].to_numpy(), df["hr"].to_numpy())):
    if (prev_sid is None) or (sid != prev_sid):
        b = float(hr) 
    else:
        b = b + BASELINE_ALPHA * (float(hr) - b)
    baseline[i] = b
    prev_sid = sid

df["baseline"] = baseline
df["delta_hr"] = (df["hr"] - df["baseline"]) / (df["baseline"] + 1e-8)

delta = df["delta_hr"].to_numpy(dtype=np.float32)

X = []
y = []

sid_arr = df["subject_id"].to_numpy()

for i in range(HISTORY, len(df)):
    if sid_arr[i - HISTORY] != sid_arr[i]:
        continue

    w = delta[i - HISTORY : i] 

    slope = np.diff(w, prepend=w[0]).astype(np.float32)

    w_std = np.std(w).astype(np.float32)
    w_std_vec = np.full((HISTORY,), w_std, dtype=np.float32)

    feats = np.stack([w, slope, w_std_vec], axis=1)

    X.append(feats)
    y.append(int(df["label"].iloc[i]))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

if len(X) < 10:
    raise RuntimeError(f"Not enough samples after windowing. Got {len(X)} samples.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=SEED
)

train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

X_train = (X_train - train_mean) / train_std
X_test  = (X_test  - train_mean) / train_std

X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
X_test  = X_test.reshape(len(X_test), -1).astype(np.float32)

# ------------------------
# Convert to torch
# ------------------------
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# mlp model
class SleepMLP(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = SleepMLP(in_dim=X_train_t.shape[1], num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

def baseline_acc(always_class: int) -> float:
    preds = np.full_like(y_test, fill_value=always_class)
    return float((preds == y_test).mean())

print("y_train counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("y_test  counts:", dict(zip(*np.unique(y_test, return_counts=True))))
print(f"Baseline always W(0): {baseline_acc(0)*100:.2f}%")
print(f"Baseline always N(1): {baseline_acc(1)*100:.2f}%")
print(f"Baseline always R(2): {baseline_acc(2)*100:.2f}%")


model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    total_n = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/total_n:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc = float((preds == y_test).mean())
print(f"\nTest accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
print("\nConfusion matrix rows=true cols=pred (0=W, 1=N, 2=R):")
print(cm)

inv_label_map = {0: "W", 1: "N", 2: "R"}
print("\nSample predictions:")
for i in range(min(100, len(y_test))):
    true_label = inv_label_map[int(y_test[i])]
    pred_label = inv_label_map[int(preds[i])]
    status = "✓" if preds[i] == y_test[i] else "✗"
    print(f"{i}: true={true_label}, pred={pred_label} {status}")
