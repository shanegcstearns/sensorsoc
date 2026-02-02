# crudemodel.py
import random
import numpy as np
import pandas as pd

import torch
import torch.onnx
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# TWEAKABLE KNOBS
# ============================================================
SEED = 42

HISTORY = 10                 # window length (timesteps)
BASELINE_ALPHA = 1 / 256     # HR baseline adaptation speed

EPOCHS = 10 # BACK TO 100
LR = 1e-3
BATCH_SIZE = 64

# How much each modality "matters" (feature scaling before training)
HR_WEIGHT = 1.0              # scale HR-derived features
ACC_WEIGHT = 1.0             # scale accel-derived features

# Feature choices
USE_ACC_MAG = True           # include accel magnitude features
USE_ACC_DXYZ = True          # include per-axis delta (motion) features
USE_ACC_JERK = True          # include jerk (diff of delta) features

# Normalize accel by subtracting per-subject mean (removes orientation / bias)
ACC_CENTER_PER_SUBJECT = True

# Optional: class weighting to fight imbalance
USE_CLASS_WEIGHTS = False

# ONNX export
EXPORT_ONNX = True
ONNX_PATH = "sleep_model.onnx"
NORM_PATH = "sleep_model_norm.npz"   # save mean/std for deployment
             # NNgen examples often use <= 10/11

# ============================================================
# Reproducibility
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("sd_out.csv")  # expected columns: time, x, y, z, hr, ss

need_cols = ["time", "x", "y", "z", "hr", "ss"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

df = df.dropna(subset=need_cols).copy()
for c in ["time", "x", "y", "z", "hr"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["time", "x", "y", "z", "hr"]).copy()

# ============================================================
# Map sleep states (your new mapping)
# W -> 0
# N1,N2 -> 1
# N3,R  -> 2
# ============================================================
label_map = {"W": 0, "N1": 1, "N2": 1, "N3": 2, "R": 2}
df = df[df["ss"].isin(label_map.keys())].copy()
df["label"] = df["ss"].map(label_map).astype(int)

# ============================================================
# Subject switch detection (time drops => new subject)
# ============================================================
time_arr = df["time"].to_numpy()
switch = np.zeros(len(df), dtype=bool)
switch[1:] = time_arr[1:] < time_arr[:-1]
df["subject_id"] = np.cumsum(switch).astype(int)

sid_arr = df["subject_id"].to_numpy(dtype=np.int64)

# ============================================================
# HR baseline per subject + delta_hr
# ============================================================
baseline = np.zeros(len(df), dtype=np.float32)
b = None
prev_sid = None
hr_arr = df["hr"].to_numpy(dtype=np.float32)

for i, (sid, hr) in enumerate(zip(sid_arr, hr_arr)):
    if (prev_sid is None) or (sid != prev_sid):
        b = float(hr)
    else:
        b = b + BASELINE_ALPHA * (float(hr) - b)
    baseline[i] = b
    prev_sid = sid

df["baseline"] = baseline
df["delta_hr"] = (df["hr"] - df["baseline"]) / (df["baseline"] + 1e-8)

# ============================================================
# Optional accel centering per subject
# ============================================================
if ACC_CENTER_PER_SUBJECT:
    df["x_c"] = df["x"] - df.groupby("subject_id")["x"].transform("mean")
    df["y_c"] = df["y"] - df.groupby("subject_id")["y"].transform("mean")
    df["z_c"] = df["z"] - df.groupby("subject_id")["z"].transform("mean")
else:
    df["x_c"] = df["x"].astype(float)
    df["y_c"] = df["y"].astype(float)
    df["z_c"] = df["z"].astype(float)

x_arr = df["x_c"].to_numpy(dtype=np.float32)
y_arr = df["y_c"].to_numpy(dtype=np.float32)
z_arr = df["z_c"].to_numpy(dtype=np.float32)

delta_hr = df["delta_hr"].to_numpy(dtype=np.float32)
labels = df["label"].to_numpy(dtype=np.int64)

# ============================================================
# Build history windows (HISTORY, F) then flatten for MLP.
# ============================================================
X = []
y = []

for i in range(HISTORY, len(df)):
    # don't let windows cross subject boundaries
    if sid_arr[i - HISTORY] != sid_arr[i]:
        continue

    # ---- HR features ----
    w_hr = delta_hr[i - HISTORY : i]
    hr_slope = np.diff(w_hr, prepend=w_hr[0]).astype(np.float32)
    hr_std = np.std(w_hr).astype(np.float32)
    hr_std_vec = np.full((HISTORY,), hr_std, dtype=np.float32)
    hr_feats = np.stack([w_hr, hr_slope, hr_std_vec], axis=1) * float(HR_WEIGHT)  # (H, 3)

    # ---- Accel features ----
    ax = x_arr[i - HISTORY : i]
    ay = y_arr[i - HISTORY : i]
    az = z_arr[i - HISTORY : i]

    acc_feat_list = []
    acc_feat_list.append(np.stack([ax, ay, az], axis=1))  # (H,3)

    if USE_ACC_MAG:
        mag = np.sqrt(ax * ax + ay * ay + az * az).astype(np.float32)
        acc_feat_list.append(mag.reshape(HISTORY, 1))
        mag_slope = np.diff(mag, prepend=mag[0]).astype(np.float32)
        acc_feat_list.append(mag_slope.reshape(HISTORY, 1))

    if USE_ACC_DXYZ:
        dax = np.diff(ax, prepend=ax[0]).astype(np.float32)
        day = np.diff(ay, prepend=ay[0]).astype(np.float32)
        daz = np.diff(az, prepend=az[0]).astype(np.float32)
        acc_feat_list.append(np.stack([dax, day, daz], axis=1))
        dmag = np.sqrt(dax * dax + day * day + daz * daz).astype(np.float32)
        acc_feat_list.append(dmag.reshape(HISTORY, 1))

    if USE_ACC_JERK:
        dax = np.diff(ax, prepend=ax[0]).astype(np.float32)
        day = np.diff(ay, prepend=ay[0]).astype(np.float32)
        daz = np.diff(az, prepend=az[0]).astype(np.float32)
        jx = np.diff(dax, prepend=dax[0]).astype(np.float32)
        jy = np.diff(day, prepend=day[0]).astype(np.float32)
        jz = np.diff(daz, prepend=daz[0]).astype(np.float32)
        acc_feat_list.append(np.stack([jx, jy, jz], axis=1))
        jmag = np.sqrt(jx * jx + jy * jy + jz * jz).astype(np.float32)
        acc_feat_list.append(jmag.reshape(HISTORY, 1))

    acc_feats = np.concatenate(acc_feat_list, axis=1).astype(np.float32) * float(ACC_WEIGHT)

    feats = np.concatenate([hr_feats, acc_feats], axis=1).astype(np.float32)  # (H, F_total)
    X.append(feats)
    y.append(int(labels[i]))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

if len(X) < 10:
    raise RuntimeError(f"Not enough samples after windowing. Got {len(X)} samples.")

# ============================================================
# Train / test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=SEED
)

# Normalize per-feature using train only
train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

X_train = (X_train - train_mean) / train_std
X_test  = (X_test  - train_mean) / train_std

# Flatten windows for MLP
X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
X_test  = X_test.reshape(len(X_test),  -1).astype(np.float32)

# Save normalization params for hardware / deployment
np.savez(NORM_PATH, mean=train_mean.astype(np.float32), std=train_std.astype(np.float32))
print(f"Saved normalization params to {NORM_PATH}")

# ============================================================
# Torch datasets
# ============================================================
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# Model
# ============================================================
class SleepMLP(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SleepMLP(in_dim=X_train_t.shape[1], num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Optional class weights
if USE_CLASS_WEIGHTS:
    unique, counts = np.unique(y_train, return_counts=True)
    freq = np.zeros(3, dtype=np.float32)
    for u, c in zip(unique, counts):
        freq[int(u)] = float(c)
    w = (freq.sum() / (freq + 1e-8)).astype(np.float32)
    w = w / w.sum() * 3.0
    class_w = torch.tensor(w, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_w)
else:
    criterion = nn.CrossEntropyLoss()

def baseline_acc(always_class: int) -> float:
    preds = np.full_like(y_test, fill_value=always_class)
    return float((preds == y_test).mean())

print("y_train counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("y_test  counts:", dict(zip(*np.unique(y_test, return_counts=True))))
print(f"Baseline always W(0): {baseline_acc(0)*100:.2f}%")
print(f"Baseline always N(1): {baseline_acc(1)*100:.2f}%")
print(f"Baseline always (N3/R)(2): {baseline_acc(2)*100:.2f}%")
print(f"Input dim (flattened): {X_train_t.shape[1]}")

# ============================================================
# Training
# ============================================================
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

# ============================================================
# Evaluation
# ============================================================
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc = float((preds == y_test).mean())
print(f"\nTest accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
print("\nConfusion matrix rows=true cols=pred (0=W, 1=N, 2=N3/R):")
print(cm)

inv_label_map = {0: "W", 1: "N1/N2", 2: "N3/R"}
print("\nSample predictions:")
for i in range(min(50, len(y_test))):
    true_label = inv_label_map[int(y_test[i])]
    pred_label = inv_label_map[int(preds[i])]
    status = "✓" if preds[i] == y_test[i] else "✗"
    print(f"{i}: true={true_label}, pred={pred_label} {status}")

# ============================================================
# Export ONNX (the correct way)
# ============================================================
# ============================================================
# Export ONNX (NNgen-friendly)
# ============================================================
if EXPORT_ONNX:
    model.eval()

    dummy = torch.zeros((1, X_train_t.shape[1]), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=11,          # NNgen-friendly
        do_constant_folding=True,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes=None,
        # IMPORTANT: force legacy exporter so opset_version is respected
        dynamo=False,
    )

    print(f"\nExported ONNX to {ONNX_PATH}")

    # sanity check right after export
    import onnx
    m = onnx.load(ONNX_PATH)
    print("ONNX opset:", m.opset_import[0].version)
    print("ONNX ir_version:", m.ir_version)
    print(f"ONNX input shape should be: (1, {X_train_t.shape[1]})")

