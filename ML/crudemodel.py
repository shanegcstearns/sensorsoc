# crudemodel.py
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix

# ============================================================
# TWEAKABLE KNOBS (play with these)
# ============================================================
SEED = 42

HISTORY = 10                 # window length (timesteps)
BASELINE_ALPHA = 1 / 256     # HR baseline adaptation speed
TEMP_BASELINE_ALPHA = 1 / 256  # skin-temp baseline adaptation speed

EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64

# How much each modality "matters" (feature scaling before training)
HR_WEIGHT = 1.0              # scale HR-derived features
TEMP_WEIGHT = 1.0            # scale temp-derived features
ACC_WEIGHT = 1.0             # scale accel-derived features

# Feature choices
USE_ACC_MAG = True           # include accel magnitude features
USE_ACC_DXYZ = True          # include per-axis delta (motion) features
USE_ACC_JERK = True          # include jerk (diff of delta) features

# Normalize accel by subtracting per-subject mean (removes orientation / bias)
ACC_CENTER_PER_SUBJECT = True

# Optional: class weighting to fight imbalance
USE_CLASS_WEIGHTS = False

# Which subjects to hold out for testing:
# - "last2"      : last two subjects in file
# - [a, b]       : explicit list/tuple of 2 subject_ids
# - integer      : one subject_id (still supported)
TEST_SUBJECTS = "last2"

# Output CSV for held-out subjects (both go into one file)
PRED_CSV = "heldout_subjects_predictions.csv"

# ============================================================
# Reproducibility
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Load data
# ============================================================
# NEW CSV FORMAT: time,x,y,z,temp,hr,ss
df = pd.read_csv("sd_out_clean.csv")  # expected columns: time, x, y, z, temp, hr, ss

need_cols = ["time", "x", "y", "z", "temp", "hr", "ss"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

df = df.dropna(subset=need_cols).copy()
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["z"] = pd.to_numeric(df["z"], errors="coerce")
df = df.dropna(subset=["time", "x", "y", "z", "temp", "hr"]).copy()

# ============================================================
# Map sleep states
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
all_subjects = np.unique(sid_arr)
if len(all_subjects) < 3:
    raise RuntimeError(
        f"Need at least 3 subjects to hold out 2 and still train. Found {len(all_subjects)}."
    )

# Resolve which subjects are test
if TEST_SUBJECTS == "last2":
    test_sids = [int(all_subjects[-2]), int(all_subjects[-1])]
elif isinstance(TEST_SUBJECTS, (list, tuple, np.ndarray)) and len(TEST_SUBJECTS) == 2:
    test_sids = [int(TEST_SUBJECTS[0]), int(TEST_SUBJECTS[1])]
    missing_test = [s for s in test_sids if s not in set(all_subjects.tolist())]
    if missing_test:
        raise RuntimeError(f"TEST_SUBJECTS contains missing IDs {missing_test}. Available: {all_subjects.tolist()}")
elif isinstance(TEST_SUBJECTS, (int, np.integer)):
    test_sids = [int(TEST_SUBJECTS)]
else:
    raise RuntimeError('TEST_SUBJECTS must be "last2", a list/tuple of 2 ints, or a single int.')

train_sids = [int(s) for s in all_subjects.tolist() if int(s) not in set(test_sids)]
if len(train_sids) < 1:
    raise RuntimeError("No training subjects left after selecting test subjects.")

print(f"Detected {len(all_subjects)} subjects via time drops.")
print(f"Held-out TEST_SUBJECTS={test_sids}; training on {len(train_sids)} subjects: {train_sids}")

# ============================================================
# HR baseline per subject + delta_hr
# Temp baseline per subject + delta_temp
# (computed across full file but resets per subject)
# ============================================================
baseline_hr = np.zeros(len(df), dtype=np.float32)
baseline_temp = np.zeros(len(df), dtype=np.float32)

b_hr = None
b_t = None
prev_sid = None

hr_arr = df["hr"].to_numpy(dtype=np.float32)
temp_arr = df["temp"].to_numpy(dtype=np.float32)

for i, (sid, hr, temp) in enumerate(zip(sid_arr, hr_arr, temp_arr)):
    if (prev_sid is None) or (sid != prev_sid):
        b_hr = float(hr)
        b_t = float(temp)
    else:
        b_hr = b_hr + BASELINE_ALPHA * (float(hr) - b_hr)
        b_t  = b_t  + TEMP_BASELINE_ALPHA * (float(temp) - b_t)

    baseline_hr[i] = b_hr
    baseline_temp[i] = b_t
    prev_sid = sid

df["hr_baseline"] = baseline_hr
df["delta_hr"] = (df["hr"] - df["hr_baseline"]) / (df["hr_baseline"] + 1e-8)

df["temp_baseline"] = baseline_temp
df["delta_temp"] = (df["temp"] - df["temp_baseline"]) / (df["temp_baseline"] + 1e-8)

# ============================================================
# Optional accel centering per subject
# NOTE: whole-subject mean (not streaming-safe), kept as-is per your request.
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
delta_temp = df["delta_temp"].to_numpy(dtype=np.float32)

labels = df["label"].to_numpy(dtype=np.int64)
times = df["time"].to_numpy(dtype=np.float64)
ss_raw = df["ss"].astype(str).to_numpy()

# ============================================================
# Build history windows
# Keep window end indices so we can write a CSV across BOTH test subjects.
# ============================================================
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []
test_end_idx = []   # df index i where window ends
test_sid_vec = []   # subject id per test sample (for CSV + per-subject metrics)

for i in range(HISTORY, len(df)):
    # don't let windows cross subject boundaries
    if sid_arr[i - HISTORY] != sid_arr[i]:
        continue

    sid = int(sid_arr[i])

    # ---- HR features (per timestep) ----
    w_hr = delta_hr[i - HISTORY : i]  # (H,)
    hr_slope = np.diff(w_hr, prepend=w_hr[0]).astype(np.float32)
    hr_std = np.std(w_hr).astype(np.float32)
    hr_std_vec = np.full((HISTORY,), hr_std, dtype=np.float32)
    hr_feats = np.stack([w_hr, hr_slope, hr_std_vec], axis=1) * float(HR_WEIGHT)

    # ---- Temp features (per timestep) ----
    w_t = delta_temp[i - HISTORY : i]  # (H,)
    t_slope = np.diff(w_t, prepend=w_t[0]).astype(np.float32)
    t_std = np.std(w_t).astype(np.float32)
    t_std_vec = np.full((HISTORY,), t_std, dtype=np.float32)

    # stack => (H, 3)
    temp_feats = np.stack([w_t, t_slope, t_std_vec], axis=1) * float(TEMP_WEIGHT)

    # ---- Accel features (per timestep) ----
    ax = x_arr[i - HISTORY : i]
    ay = y_arr[i - HISTORY : i]
    az = z_arr[i - HISTORY : i]

    acc_feat_list = []
    acc_feat_list.append(np.stack([ax, ay, az], axis=1))

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

    # ---- Combine per-timestep features ----
    feats = np.concatenate([hr_feats, temp_feats, acc_feats], axis=1).astype(np.float32)
    lab = int(labels[i])

    if sid in set(test_sids):
        X_test_list.append(feats)
        y_test_list.append(lab)
        test_end_idx.append(i)
        test_sid_vec.append(sid)
    else:
        X_train_list.append(feats)
        y_train_list.append(lab)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.int64)
X_test  = np.array(X_test_list, dtype=np.float32)
y_test  = np.array(y_test_list, dtype=np.int64)
test_end_idx = np.array(test_end_idx, dtype=np.int64)
test_sid_vec = np.array(test_sid_vec, dtype=np.int64)

if len(X_train) < 10:
    raise RuntimeError(f"Not enough TRAIN samples after windowing. Got {len(X_train)} samples.")
if len(X_test) < 10:
    raise RuntimeError(f"Not enough TEST samples after windowing for subjects {test_sids}. Got {len(X_test)} samples.")

# ============================================================
# Normalize per-feature using TRAIN ONLY
# ============================================================
train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

X_train_n = (X_train - train_mean) / train_std
X_test_n  = (X_test  - train_mean) / train_std

# Flatten windows for MLP
X_train_n = X_train_n.reshape(len(X_train_n), -1).astype(np.float32)
X_test_n  = X_test_n.reshape(len(X_test_n),  -1).astype(np.float32)

# ============================================================
# Torch datasets
# ============================================================
X_train_t = torch.tensor(X_train_n)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test_n)
y_test_t  = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# Model
# ============================================================
class SleepMLP(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = SleepMLP(in_dim=X_train_t.shape[1], num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Optional class weights (computed on TRAIN only)
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

# ============================================================
# Helpers / baselines on the held-out subjects (combined)
# ============================================================
def baseline_acc(always_class: int) -> float:
    preds_b = np.full_like(y_test, fill_value=always_class)
    return float((preds_b == y_test).mean())

print("y_train counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("y_test  counts (combined):", dict(zip(*np.unique(y_test, return_counts=True))))
print(f"Baseline always W(0): {baseline_acc(0)*100:.2f}%")
print(f"Baseline always N(1): {baseline_acc(1)*100:.2f}%")
print(f"Baseline always (N3/R)(2): {baseline_acc(2)*100:.2f}%")
print(f"Feature dims per timestep: {X_train.shape[-1]}  | flattened: {X_train_n.shape[1]}")
print(f"HR_WEIGHT={HR_WEIGHT}  TEMP_WEIGHT={TEMP_WEIGHT}  ACC_WEIGHT={ACC_WEIGHT}  ACC_CENTER_PER_SUBJECT={ACC_CENTER_PER_SUBJECT}")
print(f"USE_ACC_MAG={USE_ACC_MAG}  USE_ACC_DXYZ={USE_ACC_DXYZ}  USE_ACC_JERK={USE_ACC_JERK}")
print(f"USE_CLASS_WEIGHTS={USE_CLASS_WEIGHTS}")

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
# Evaluation on held-out subjects (combined)
# ============================================================
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc = float((preds == y_test).mean())
print(f"\nHeld-out subjects {test_sids} combined accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
print("\nConfusion matrix (combined) rows=true cols=pred (0=W, 1=N, 2=N3/R):")
print(cm)

for sid in test_sids:
    m = (test_sid_vec == int(sid))
    if m.sum() == 0:
        continue
    acc_sid = float((preds[m] == y_test[m]).mean())
    print(f"Held-out subject {sid} accuracy: {acc_sid * 100:.2f}%  (n={int(m.sum())})")

inv_label_map = {0: "W", 1: "N1/N2", 2: "N3/R"}

# ============================================================
# Write per-timestep CSV for BOTH held-out subjects
# ============================================================
out = pd.DataFrame({
    "subject_id": test_sid_vec,
    "time": times[test_end_idx],
    "actual_label": y_test,
    "pred_label": preds,
    "actual_state": [inv_label_map[int(v)] for v in y_test],
    "pred_state":   [inv_label_map[int(v)] for v in preds],
    "ss_raw": ss_raw[test_end_idx],
})

out["_end_idx"] = test_end_idx
out = out.sort_values(["subject_id", "_end_idx"]).drop(columns=["_end_idx"])

out.to_csv(PRED_CSV, index=False)
print(f"\nWrote held-out subjects predictions to: {PRED_CSV}")
print(out.head(10).to_string(index=False))
