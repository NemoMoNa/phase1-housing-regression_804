# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # ← 追加！
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# NumPy ndarray → Pandas DataFrameに変換（列名も設定すると便利）
df_X = pd.DataFrame(X, columns=data.feature_names)

# 各列の欠損値数を確認
print(df_X.isnull().sum())

# 平均で欠損値を埋める
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Convert to PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x) #model(x) → 実は forward() が呼ばれる

model = MLP()

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

for epoch in range(30):
    model.train()
    total_loss = 0 #added 
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() #そのミニバッチ全体の平均ロス（スカラー値）
        
    avg_loss = total_loss / len(train_loader) #added    
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    mse = criterion(preds, y_test_tensor)
    print(f"Test MSE: {mse.item():.4f}")
    
import matplotlib.pyplot as plt

plt.scatter(y_test_tensor.numpy(), preds.numpy(), alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Median House Values")
plt.grid(True)
plt.show()
