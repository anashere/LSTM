#!/usr/bin/env python
# coding: utf-8

# In[24]:


# ==========================================
# Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader


# ==========================================
# 1.Load the data set
# ==========================================
df = pd.read_csv("LT.csv")

df.info()

#fix date , sorting and cleaning
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates().dropna()

# selecting needed columns
cols = ['Prev Close','Open','High','Low','Last','Close',
        'VWAP', 'Volume', 'Turnover', 'Trades','Deliverable Volume']
data =df[cols].values
dates =df['Date'].values


# ==========================================
# 2. Train/Test Split & Normalization
# ==========================================
split=int(len(data) * 0.8)
train_data=data[:split]
test_data=data[split:]

scaler=StandardScaler()
train_scaled=scaler.fit_transform(train_data)
test_scaled=scaler.transform(test_data)
scaled_data =np.vstack((train_scaled,test_scaled))


# ==========================================
# 3.Time-Series Chunking
# ==========================================
X,y,target_dates = [],[],[]
for i in range(len(scaled_data)-20-5+1):
        X.append(scaled_data[i:i+20])
        y.append(scaled_data[i+20:i+20+5])
        target_dates.append(dates[i+20:i+20+5])

X=np.array(X)
y=np.array(y)

# split again
seq_split=int(len(X)*0.8)
X_train,X_test =X[:seq_split],X[seq_split:]
y_train,y_test =y[:seq_split],y[seq_split:]
dates_test = target_dates[seq_split:]

#loaders
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)) , batch_size=32 , shuffle=False)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)) , batch_size=32 , shuffle=False)


# ==========================================
# 4. Defining LSTM Model
# ==========================================
class TimeSeriesLSTM(nn.Module):
        def __init__(self):
                super().__init__()
                # 11 features in , 128 hidden size, 2 layers
                self.lstm=nn.LSTM(11,128,2,batch_first=True,dropout=0.3)
                #Output gives 5 days x 11 features
                self.fc=nn.Linear(128,55)

        def forward(self,x):
                out, _ = self.lstm(x)
                # Take the last time step output
                out = self.fc(out[:,-1,:]) 
                # Reshape output to 5 days and 11 features
                return out.view(-1,5,11)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=TimeSeriesLSTM().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn=nn.MSELoss()


# ==========================================
# 5. Training Loop
# ==========================================
for epoch in range(60):
        model.train()
        for batch_X, batch_y in train_loader:
                batch_X, batch_y=batch_X.to(device),batch_y.to(device)
        
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
        
        if epoch % 10==0:
                print(f"Epoch {epoch} finished. Loss: {loss.item():.4f} ")

# ==========================================
# 6.Evaluation and Metrics
# ==========================================
model.eval()
test_preds=[]
test_actuals=[]
with torch.no_grad():
        for batch_X,batch_y in test_loader:
                preds = model(batch_X.to(device)).cpu().numpy()
                test_preds.append(preds)
                test_actuals.append(batch_y.numpy())

# combine all batches
test_preds =np.concatenate(test_preds,axis=0)
test_actuals =np.concatenate(test_actuals,axis=0)

# reshape to (-1, 11) for inverse scaling
test_preds =test_preds.reshape(-1,11)
test_actuals =test_actuals.reshape(-1,11)

# inverse transform (back to original values)
inv_preds =scaler.inverse_transform(test_preds)
inv_actuals =scaler.inverse_transform(test_actuals)

feature_names=['Prev Close','Open','High','Low','Last','Close',
               'VWAP','Volume','Turnover','Trades','Deliverable Volume']

print("\n--- Overall Model Performance ---")

overall_mse=mean_squared_error(inv_actuals,inv_preds)
overall_rmse=np.sqrt(overall_mse)
overall_mae=mean_absolute_error(inv_actuals,inv_preds)
overall_mape=np.mean(np.abs((inv_actuals - inv_preds)/(inv_actuals+1e-8)))*100
overall_accuracy = 100 - overall_mape

print(f"Overall MSE: {overall_mse:.4f}")
print(f"Overall RMSE: {overall_rmse:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Overall MAPE: {overall_mape:.2f}%")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

print("\n" + "="*40)
print("Performance Metrics (All Features)")
print("="*40)
for i in range(11):
        true_vals = inv_actuals[:, i]
        pred_vals = inv_preds[:, i]
        mse=mean_squared_error(true_vals, pred_vals)
        rmse=np.sqrt(mse)
        mae=mean_absolute_error(true_vals, pred_vals)
        mape=np.mean(np.abs((true_vals - pred_vals)/(true_vals + 1e-8))) * 100
        accuracy=100 - mape

        print(f"\nFeature: {feature_names[i]}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

# ==========================================
# 7. Plotting True vs Predicted
# ==========================================
# reshape back to sequence form
inv_preds_seq=inv_preds.reshape(-1,5,11)
inv_actuals_seq=inv_actuals.reshape(-1,5,11)

# create 11 subplots (4 rows x 3 columns = 12 slots)
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i in range(11):
        true_vals = inv_actuals_seq[:, 0, i]
        pred_vals = inv_preds_seq[:, 0, i]

        axes[i].plot(true_vals[-150:], label="True", marker='.')
        axes[i].plot(pred_vals[-150:], label="Predicted", linestyle='--')
        axes[i].set_title(feature_names[i])
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend()

fig.delaxes(axes[11])

fig.suptitle("All Feature Predictions (True vs Predicted)", fontsize=16)

plt.tight_layout()
plt.show()

# %%
