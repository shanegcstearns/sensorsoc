# import torch
# from torch import nn
# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader
# from torchvision import transforms


# '''
#     Multilayer Perceptron
# '''
# class MLP(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * 32 * 3, 64), 
#             nn.ReLU(),
#             nn.Linear(64, 32), 
#             nn.ReLU(), 
#             nn.Linear(32, 10)
#         )

#     def forward(self, x):
#         '''Forward pass'''
#         return self.layers(x)

# if __name__ == '__main__':

#     # Hyperparameters
#     batch_size = 10
#     learning_rate = 1e-4
#     epochs = 5

#     # set fixed random seed
#     torch.manual_seed(42)

#     # data transformations
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
#     )

#     # CIFAR10 dataset (save to local)
#     dataset = CIFAR10(root='./data/cifar10',train=True,download=True,transform=transform) # CHANGE TO SLEEP DATASET
#     trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

#     # Initializing Net
#     mlp = MLP()

#     # defining loss function
#     loss_function = nn.CrossEntropyLoss()
#     # defining optimizer
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)


#     # training loop
#     for epoch in range(epochs):
        
#         print('Epoch %s' % (epoch+1))

#         current_loss = 0.0

#         # iterate over the data
#         for i, data in enumerate(trainloader, 0):
            
#             # get data and ground truth
#             inputs, targets = data

#             # set gradients of all optimized tensors to zero
#             optimizer.zero_grad()

#             # forward pass of data through net
#             outputs = mlp(inputs)

#             # compute loss
#             loss = loss_function(outputs, targets)

#             # backward pass
#             loss.backward()

#             # optimizing parameters
#             optimizer.step()

#             # show stats
#             current_loss += loss.item()
#             if i % 500 == 499:
#                 print('Loss after mini-batch %5d: %.3f'% (i+1, current_loss/500))
#                 current_loss = 0.0

#     # end
#     print('Training finished!')
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------
# Hyperparameters
# ------------------------
HISTORY = 5
BASELINE_ALPHA = 1 / 1024
EPOCHS = 20
LR = 1e-3

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("sd_out_clean.csv")  # columns: time, hr, ss

# ------------------------
# Map sleep stages to ints
# ------------------------
label_map = {
    "W": 0,
    "N1": 1,
    "N2": 1,
    "N3": 1,
    "R": 2,
}

df = df[df["ss"].isin(label_map.keys())]
df["label"] = df["ss"].map(label_map)

# ------------------------
# Adaptive HR baseline
# ------------------------
baseline = []
b = df["hr"].iloc[0]

for hr in df["hr"]:
    b = b + BASELINE_ALPHA * (hr - b)
    baseline.append(b)

df["baseline"] = baseline
df["delta_hr"] = (df["hr"] - df["baseline"]) / df["baseline"]

# ------------------------
# Build history windows
# ------------------------
X = []
y = []

for i in range(HISTORY, len(df)):
    X.append(df["delta_hr"].iloc[i-HISTORY:i].values)
    y.append(df["label"].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# ------------------------
# Train / test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# ------------------------
# Model
# ------------------------
class HROnlySleepModel(nn.Module):
    def __init__(self, history_len, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(history_len, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = HROnlySleepModel(HISTORY, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ------------------------
# Training
# ------------------------
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {loss.item():.4f}")

# ------------------------
# Evaluation
# ------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y_test).float().mean().item()

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# Show a few example predictions
inv_label_map = {v: k for k, v in label_map.items()}
print("\nSample predictions:")
for i in range(10):
    true_label = inv_label_map[y_test[i].item()]
    pred_label = inv_label_map[preds[i].item()]
    status = "✓" if preds[i] == y_test[i] else "✗"
    print(f"{i}: true={true_label}, pred={pred_label} {status}")
