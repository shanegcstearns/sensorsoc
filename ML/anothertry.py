import torch
# import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

import pandas as pd
import numpy as np
# ============================================================
# Data Features:
# 1. Accel magnitude
# 2. Accel magnitude slope (jerk)
# 3. Heart rate delta
# 4. HR RMSSD (if possible)
# 5. Skin temperature?
# 6. HRV / movement (possibly)
#
# Want to identify: after X amount of time has elapsed, can we reasonably predict when is a good time to wake up user
# ============================================================

SETUP_MAX_SAMPLES = 10 * 60 * 1   # first 10 minutes (adjust FS if needed)
MIN_SETUP_SAMPLES = 0           # minimum Wake samples to trust
EPS = 1e-8
ACCEL_SHIFT = 6  # for accel baseline calc, bits to shift by.

#History window size, for model input and RMSSD
WINDOW_SEC = 60  # 1-minute window
STEP_SEC = 300   # every 5 minutes
WINDOW_SAMPLES = WINDOW_SEC 
STEP_SAMPLES = STEP_SEC

SEED = 16 #42 org
EPOCHS = 50
LR = 1e-4

df = pd.read_csv("sd_out_clean.csv")

need = ["time", "x", "y", "z", "temp", "hr", "ss"]
df = df.dropna(subset=need).copy()

#Intuition: the model will be slept for X amount of hours,
# we (probably) won't be in wake, even if we are it gets miscategorized as light sleep


# Assign subject IDs based on time resets
time_arr = df["time"].to_numpy()
switch = np.zeros(len(df), dtype=bool)
switch[1:] = time_arr[1:] < time_arr[:-1]
df["subject_id"] = np.cumsum(switch)


import numpy as np
import pandas as pd

def compute_resting_hr_per_subject(df):
    

    return resting_hr

# ============================================================
# HR baseline + delta
# ============================================================
resting_hr = {}
baseline_x = np.zeros(len(df))
baseline_y = np.zeros(len(df))
baseline_z = np.zeros(len(df))

for sid, g in df.groupby("subject_id", sort=False):
    idx = g.index  # indices in the full df
    g = g.reset_index(drop=True)  # optional, for clean local indexing

    # HR baseline
    early = g.iloc[:SETUP_MAX_SAMPLES]
    wake_hr = early.loc[early["ss"] == "W", "hr"].values
    if len(wake_hr) >= MIN_SETUP_SAMPLES:
        resting_hr[sid] = np.median(wake_hr)
    else:
        resting_hr[sid] = early["hr"].mean()

    # Accel EMA baseline
    b_x = df.loc[idx[0], "x"]
    b_y = df.loc[idx[0], "y"]
    b_z = df.loc[idx[0], "z"]

    for j, i in enumerate(idx):
        b_x += (df.loc[i, "x"] - b_x) / (2**ACCEL_SHIFT)
        b_y += (df.loc[i, "y"] - b_y) / (2**ACCEL_SHIFT)
        b_z += (df.loc[i, "z"] - b_z) / (2**ACCEL_SHIFT)

        baseline_x[i] = b_x
        baseline_y[i] = b_y
        baseline_z[i] = b_z

# Subtract the baselines
df["x_c"] = df["x"] - baseline_x
df["y_c"] = df["y"] - baseline_y
df["z_c"] = df["z"] - baseline_z

# HR baseline & delta
df["hr_baseline"] = df["subject_id"].map(resting_hr)
df["delta_hr"] = (df["hr"] - df["hr_baseline"]) / (df["hr_baseline"] + EPS) #what does EPS do here?

# ============================================================
# Label mapping + Clearing out Wake
# ============================================================
df = df[~df['ss'].isin(['W'])]
label_map = {"N1": 0, "N2": 0, "R": 0,"N3": 1} # 0 = Okay to wake, 1 = bad to wake
df = df[df["ss"].isin(label_map)]
df["label"] = df["ss"].map(label_map).astype(int)

# ============================================================
# Feature Creation
# ============================================================
# Magnitude from centered axes
df["accel_mag"] = np.sqrt(df["x_c"]**2 + df["y_c"]**2 + df["z_c"]**2) 
# Jerk
df["accel_jerk"] = np.abs(df["accel_mag"].diff().fillna(0)) #trying abs, having negatives makes it more complex

# HR RMSSD
hr = df["hr"].to_numpy()
rmssd = np.zeros(len(hr))

for start in range(0, len(hr) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
    end = start + WINDOW_SAMPLES
    hr_window = hr[start:end]
    hr_diff = np.diff(hr_window)
    rmssd_value = np.sqrt(np.mean(hr_diff**2))
    
    # assign this value to all samples in the window
    rmssd[start:end] = rmssd_value
df["hr_rmssd"] = rmssd

#Composite HRV + movement feature
df["hrv_mov"] = df["hr_rmssd"] * df["accel_mag"] #try out composite movement + HRV feature 


features = [
    "accel_mag",
    "accel_jerk",
    "delta_hr",
    "hr_rmssd",
    "temp",
    "hrv_mov"
]

X = df[features].fillna(0).to_numpy()
y = df["label"].to_numpy()

# ============================================================
# Train / test split
# ============================================================

np.random.seed(SEED)  # reproducible
subject_ids = df["subject_id"].unique()
print(subject_ids)
train_subs = np.random.choice(subject_ids, size=7, replace=False)
test_subs = np.array([sid for sid in subject_ids if sid not in train_subs])

print("Train subjects:", train_subs)
print("Test subjects:", test_subs)

train_df = df[df["subject_id"].isin(train_subs)]
test_df = df[df["subject_id"].isin(test_subs)]

X_train = train_df[features].to_numpy()
y_train = train_df["label"].to_numpy()

X_test = test_df[features].to_numpy()
y_test = test_df["label"].to_numpy()

class_sample_counts = np.bincount(y_train)
weights = 1.0 / class_sample_counts
sample_weights = weights[y_train]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(
    train_ds,
    batch_size=WINDOW_SEC,
    #shuffle=True,
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
# Compute class counts
classes, counts = np.unique(y_train, return_counts=True)
# Inverse frequency weighting
weights = counts.sum() / (len(classes) * counts)
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Class weights:", class_weights)

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
pos_weight = 5.0  # tweak between 3-10
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
#loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ============================================================
# Train
# ============================================================

train_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        # 1️⃣ Forward pass
        outputs = model(X_batch)          # raw logits
        loss = loss_fn(outputs, y_batch) # compute loss

        # 2️⃣ Backward pass
        opt.zero_grad()   # clear previous gradients
        loss.backward()   # compute gradients w.r.t. weights

        # 3️⃣ Update weights
        opt.step()

        epoch_loss += loss.item() * X_batch.size(0)  # multiply by batch size

    # Average loss over all samples in epoch
    avg_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_output = model(X_test_tensor)
    test_pred = torch.argmax(test_output, dim=1)

    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    accuracy = (test_pred == y_test_tensor).float().mean()

print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
print(confusion_matrix(y_test, test_pred, labels=[0,1]))
print(classification_report(y_test, test_pred, labels=[0,1]))
