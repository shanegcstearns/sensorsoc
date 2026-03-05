# transition_sleep_model_newcsv.py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# Data Features (NEW CSV):
# subject_id, subject_orig, time, hr, movement, cosine, ss
#
# Notes:
# - You said we can ignore subject_orig.
# - movement is already a (likely) accel magnitude / activity measure, so we use it as accel_mag.
# - cosine is an extra feature (e.g., time-of-day phase); we include it directly.
# - temp/x/y/z no longer exist, so all accel baseline code is removed.
# ============================================================

SETUP_MAX_SAMPLES = 10 * 60 * 1   # first 10 minutes (adjust if needed)
MIN_SETUP_SAMPLES = 0            # minimum Wake samples to trust
EPS = 1e-8

# History window size, for model input and RMSSD
WINDOW_SEC = 60   # 1-minute window (in samples if fs=1Hz)
STEP_SEC = 300    # every 5 minutes
WINDOW_SAMPLES = WINDOW_SEC
STEP_SAMPLES = STEP_SEC

SEED = 16
EPOCHS = 50
LR = 1e-4

CSV_PATH = "compiled_sleep_dataset.csv"  # <- change if needed

# ============================================================
# Load + basic cleaning
# ============================================================
df = pd.read_csv(CSV_PATH)

# Required columns for the new format
need = ["subject_id", "time", "hr", "movement", "cosine", "ss"]
df = df.dropna(subset=need).copy()

# Ensure expected dtypes
df["subject_id"] = df["subject_id"].astype(int)
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
df["movement"] = pd.to_numeric(df["movement"], errors="coerce")
df["cosine"] = pd.to_numeric(df["cosine"], errors="coerce")
df = df.dropna(subset=["time", "hr", "movement", "cosine"]).copy()

# Sort within each subject by time (important for diff / rmssd windows)
df = df.sort_values(["subject_id", "time"], kind="mergesort").reset_index(drop=True)

# ============================================================
# HR baseline + delta (per-subject)
# ============================================================
resting_hr = {}

for sid, g in df.groupby("subject_id", sort=False):
    early = g.iloc[:SETUP_MAX_SAMPLES]
    wake_hr = early.loc[early["ss"] == "W", "hr"].values
    if len(wake_hr) >= MIN_SETUP_SAMPLES and len(wake_hr) > 0:
        resting_hr[sid] = float(np.median(wake_hr))
    else:
        resting_hr[sid] = float(early["hr"].mean())

df["hr_baseline"] = df["subject_id"].map(resting_hr)
df["delta_hr"] = (df["hr"] - df["hr_baseline"]) / (df["hr_baseline"] + EPS)

# ============================================================
# Label mapping + Clearing out Wake
# ============================================================
# Keep only sleep stages you care about.
# If your dataset includes other labels, they will get dropped here unless you add them.
df = df[~df["ss"].isin(["W"])].copy()

# 0 = okay to wake, 1 = bad to wake
label_map = {"N1": 0, "N2": 0, "R": 0, "N3": 1}
df = df[df["ss"].isin(label_map)].copy()
df["label"] = df["ss"].map(label_map).astype(int)

# ============================================================
# Feature Creation (NEW CSV)
# ============================================================
# movement is treated as accel magnitude
df["accel_mag"] = df["movement"].fillna(0.0)

# jerk = abs(diff) per subject (diff should not cross subject boundaries)
df["accel_jerk"] = 0.0
for sid, g in df.groupby("subject_id", sort=False):
    idx = g.index
    mag = g["accel_mag"].to_numpy()
    jerk = np.abs(np.diff(mag, prepend=mag[0]))
    df.loc[idx, "accel_jerk"] = jerk

# HR RMSSD per subject (windowed). If your sampling rate is not 1Hz,
# change WINDOW_SAMPLES/STEP_SAMPLES accordingly.
df["hr_rmssd"] = 0.0
for sid, g in df.groupby("subject_id", sort=False):
    idx = g.index
    hr = g["hr"].to_numpy()
    rmssd = np.zeros(len(hr), dtype=float)

    if len(hr) >= WINDOW_SAMPLES:
        for start in range(0, len(hr) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            end = start + WINDOW_SAMPLES
            hr_window = hr[start:end]
            hr_diff = np.diff(hr_window)
            rmssd_value = float(np.sqrt(np.mean(hr_diff**2))) if len(hr_diff) > 0 else 0.0
            rmssd[start:end] = rmssd_value
    else:
        # If too short, just compute a single RMSSD-like value over the whole subject
        hr_diff = np.diff(hr)
        rmssd_value = float(np.sqrt(np.mean(hr_diff**2))) if len(hr_diff) > 0 else 0.0
        rmssd[:] = rmssd_value

    df.loc[idx, "hr_rmssd"] = rmssd

# Composite feature
df["hrv_mov"] = df["hr_rmssd"] * df["accel_mag"]

# cosine is already a column; include directly
# (If cosine is periodic time-of-day, keeping it raw is fine; scaler could help later.)
features = [
    "accel_mag",
    "accel_jerk",
    "delta_hr",
    "hr_rmssd",
    #"temp",
    "hrv_mov"
]

X = df[features].fillna(0.0).to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int64)

# ============================================================
# Train / test split (subject-wise)
# ============================================================
np.random.seed(SEED)
subject_ids = df["subject_id"].unique()

# Pick a reasonable split even if you have fewer subjects than before
n_subjects = len(subject_ids)
if n_subjects < 2:
    raise ValueError("Need at least 2 subjects for a subject-wise train/test split.")

# ~80/20 split, but at least 1 subject in test
n_train = max(1, int(round(0.8 * n_subjects)))
n_train = min(n_train, n_subjects - 1)

train_subs = np.random.choice(subject_ids, size=n_train, replace=False)
test_subs = np.array([sid for sid in subject_ids if sid not in train_subs])

print("Subjects:", subject_ids)
print("Train subjects:", train_subs)
print("Test subjects:", test_subs)

train_df = df[df["subject_id"].isin(train_subs)].copy()
test_df = df[df["subject_id"].isin(test_subs)].copy()

X_train = train_df[features].fillna(0.0).to_numpy(dtype=np.float32)
y_train = train_df["label"].to_numpy(dtype=np.int64)

X_test = test_df[features].fillna(0.0).to_numpy(dtype=np.float32)
y_test = test_df["label"].to_numpy(dtype=np.int64)

# ============================================================
# Sampler for imbalance (safe against missing class)
# ============================================================
class_counts = np.bincount(y_train, minlength=2).astype(float)
if np.any(class_counts == 0):
    # If one class is missing in training, WeightedRandomSampler can't fix that.
    # Train will be degenerate; still run but warn.
    print("WARNING: One of the classes is missing in y_train:", class_counts)

# Inverse frequency weights (avoid divide-by-zero)
inv = np.zeros_like(class_counts)
inv[class_counts > 0] = 1.0 / class_counts[class_counts > 0]
sample_weights = inv[y_train] if len(y_train) > 0 else np.array([], dtype=float)

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(
    train_ds,
    batch_size=WINDOW_SEC,  # same as before
    sampler=sampler
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

test_loader = DataLoader(
    test_ds,
    batch_size=WINDOW_SEC,
    shuffle=False
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
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),  # raw logits
        )

    def forward(self, x):
        return self.net(x)

model = TransitionMLP(X_train.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=LR)
pos_weight = 10.0  # tweak between 3-10
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
#loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ============================================================
# Train
# ============================================================
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item() * X_batch.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# ============================================================
# Evaluate
# ============================================================
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_output = model(X_test_tensor)
    test_pred = torch.argmax(test_output, dim=1).cpu().numpy()

accuracy = (test_pred == y_test).mean() if len(y_test) > 0 else 0.0
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
print(classification_report(y_test, test_pred, labels=[0, 1], zero_division=0))
