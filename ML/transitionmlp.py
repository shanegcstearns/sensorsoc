# transition_sleep_model.py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# Config
# ============================================================
SEED = 42
HISTORY = 10
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64
SMOOTH_K = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Load + clean data
# ============================================================
df = pd.read_csv("sd_out.csv")

need = ["time", "x", "y", "z", "hr", "ss"]
df = df.dropna(subset=need).copy()

# Sleep state mapping
label_map = {"W": 0, "N1": 1, "N2": 1, "N3": 2, "R": 2}
df = df[df["ss"].isin(label_map)]
df["label"] = df["ss"].map(label_map).astype(int)

# Subject split via time reset
time_arr = df["time"].to_numpy()
switch = np.zeros(len(df), dtype=bool)
switch[1:] = time_arr[1:] < time_arr[:-1]
df["subject_id"] = np.cumsum(switch)

# ============================================================
# Fixed-point helpers
# ============================================================
QHR = 14    # fractional bits for HR features
QACC = 6    # accel fractional bits

def q(x, qbits):
    return np.round(x * (1 << qbits)).astype(np.int16)

def qclip(x, qbits, bits=16):
    lim = (1 << (bits-1)) - 1
    return np.clip(q(x, qbits), -lim, lim).astype(np.int16)

# ============================================================
# HR baseline + delta
# ============================================================
BASELINE_ALPHA = 1 / 256
baseline = np.zeros(len(df), dtype=np.float32)

b = None
prev_sid = None
for i, (sid, hr) in enumerate(zip(df["subject_id"], df["hr"])):
    if prev_sid is None or sid != prev_sid:
        b = hr
    else:
        b = b + BASELINE_ALPHA * (hr - b)
    baseline[i] = b
    prev_sid = sid

df["delta_hr"] = (df["hr"] - baseline) / (baseline + 1e-8)

# Center accel per subject
for axis in ["x", "y", "z"]:
    df[f"{axis}_c"] = df[axis] - df.groupby("subject_id")[axis].transform("mean")

# ============================================================
# Transition labels
# ============================================================
# 0 = stay, 1 = lighten, 2 = deepen
def transition_label(prev, curr):
    if curr == prev:
        return 0
    elif curr < prev:
        return 1
    else:
        return 2

# ============================================================
# Windowing
# ============================================================
X = []
y_trans = []
y_state = []

labels = df["label"].to_numpy()
sid = df["subject_id"].to_numpy()

x = df["x_c"].to_numpy(np.float32)
y = df["y_c"].to_numpy(np.float32)
z = df["z_c"].to_numpy(np.float32)
dhr = df["delta_hr"].to_numpy(np.float32)

for i in range(HISTORY, len(df)):
    if sid[i - HISTORY] != sid[i]:
        continue

    prev_s = labels[i - 1]
    curr_s = labels[i]

    # transition target
    y_trans.append(transition_label(prev_s, curr_s))
    y_state.append(curr_s)

    # HR features
    w_hr = dhr[i - HISTORY:i]
    hr_slope = np.diff(w_hr, prepend=w_hr[0])
    hr_std = np.std(w_hr)

    w_hr_q = qclip(w_hr, QHR)
    hr_slope_q = qclip(hr_slope, QHR)
    hr_std_q = qclip(hr_std, QHR)
    hr_std_vec_q = np.full(HISTORY, hr_std_q, dtype=np.int16)

    hr_feats = np.stack([w_hr_q, hr_slope_q, hr_std_vec_q], axis=1)

    # Accel features
    ax = x[i - HISTORY:i]
    ay = y[i - HISTORY:i]
    az = z[i - HISTORY:i]

    # L1 norm instead of sqrt (much cheaper, often better)
    mag = np.abs(ax) + np.abs(ay) + np.abs(az)
    mag_slope = np.diff(mag, prepend=mag[0])

    acc_feats = np.stack([
        ax.astype(np.int16),
        ay.astype(np.int16),
        az.astype(np.int16),
        mag.astype(np.int16),
        mag_slope.astype(np.int16)
    ], axis=1)

    feats = np.concatenate([hr_feats, acc_feats], axis=1)
    X.append(feats)

X = np.array(X)
y_trans = np.array(y_trans)
y_state = np.array(y_state)

# ============================================================
# Train / test split (NO SHUFFLE)
# ============================================================
X_tr, X_te, ytr_t, yte_t, ytr_s, yte_s = train_test_split(
    X, y_trans, y_state,
    test_size=0.3,
    shuffle=False,
    random_state=SEED
)

# Normalize
# Replace normalization with shift-only scaling
X_tr = X_tr >> 1   # example
X_te = X_te >> 1


# Flatten
X_tr = X_tr.reshape(len(X_tr), -1)
X_te = X_te.reshape(len(X_te), -1)

# Torch
train_ds = TensorDataset(
    torch.tensor(X_tr, dtype=torch.int16).float(),
    torch.tensor(ytr_t)
)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# Model
# ============================================================
class TransitionMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)

model = TransitionMLP(X_tr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ============================================================
# Train
# ============================================================
for e in range(EPOCHS):
    tot = 0
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    print(f"Epoch {e+1:02d} | loss={tot/len(train_ds):.4f}")

# ============================================================
# Decode transitions → states
# ============================================================
def decode_transitions(trans, init_state):
    states = [int(init_state)]
    for t in trans:
        s = states[-1]
        if t == 1:
            s = max(0, s - 1)
        elif t == 2:
            s = min(2, s + 1)
        states.append(s)
    return np.array(states[1:])

def smooth_states(states, k):
    out = []
    for i in range(len(states)):
        w = states[max(0, i-k):i+1]
        out.append(np.bincount(w).argmax())
    return np.array(out)
# ============================================================
# Evaluate
# ============================================================
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_te, dtype=torch.int16).float())
    tpred = torch.argmax(logits, dim=1).numpy()

state_pred = decode_transitions(tpred, yte_s[0])
state_pred_s = smooth_states(state_pred, SMOOTH_K)

acc_raw = (state_pred == yte_s).mean()
acc_s   = (state_pred_s == yte_s).mean()

print("\nState accuracy:")
print(f"  raw:     {acc_raw*100:.2f}%")
print(f"  smooth:  {acc_s*100:.2f}%")

print("\nConfusion matrix (smoothed):")
print(confusion_matrix(yte_s, state_pred_s, labels=[0,1,2]))


# ============================================================
# Export model to ONNX
# ============================================================

# Pick a dummy input that matches the model input shape
dummy_input = torch.tensor(X_te[:1], dtype=torch.int16).float()

# Export the model
onnx_path = "transition_sleep_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,          # NNgen-friendly
    do_constant_folding=True,
    input_names=["x"],
    output_names=["logits"],
    dynamic_axes=None,
    # IMPORTANT: force legacy exporter so opset_version is respected
    dynamo=False,
)

print(f"\nModel exported to ONNX at {onnx_path}")
