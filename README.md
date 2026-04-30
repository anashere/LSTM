# Time Series Forecasting using LSTM (PyTorch)

## Student Details
- Name: Muhammed Anas Ali P Y  
- Roll No: 25MAM030  

---

## Project Overview

This project implements a **multi-output time series forecasting model** using LSTM in PyTorch.
The model uses past **20 days of stock data** to predict the **next 5 days of all 11 features**.

---
## Dataset
- Dataset: NIFTY-50 Stock Market Data  
- File used: **LT.csv (Larsen & Toubro)**  
- Selected based on roll number  

The dataset contains historical stock price and trading data from 2000 to 2021.

---

##  Features Used
- Prev Close  
- Open  
- High  
- Low  
- Last  
- Close  
- VWAP  
- Volume  
- Turnover  
- Trades  
- Deliverable Volume  

---

##  Methodology

### 1. Data Preprocessing
- Converted date column to datetime format  
- Sorted data chronologically  
- Removed missing values and duplicates  

### 2. Train-Test Split
- 80% training data  
- 20% testing data  

### 3. Normalization
- Used StandardScaler  
- Applied on training data and then test data  

### 4. Sequence Creation
- Input: last **20 days (11 features)**  
- Output: next **5 days (11 features)**  
- Overlapping sequences used  

---

## Model
- LSTM model implemented using PyTorch  
- 2 LSTM layers (hidden size = 128)  
- Fully connected layer  
- Output reshaped to (5 days × 11 features)  

---

##  Evaluation Metrics

### Overall Performance
- MSE  
- RMSE  
- MAE  
- MAPE  
- Accuracy (100 - MAPE)  

### Feature-wise Performance
Metrics are calculated for all 11 features individually.

---

## Results
- Good accuracy (~93–94%) for price-related features  
- Lower accuracy for volume-related features due to high variability  

---

## Visualization
- All 11 features plotted using subplots  
- Comparison between actual and predicted values  

---

## Conclusion
The model successfully performs multi-output prediction.  
It captures price trends well but performs lower on highly volatile features like volume and trades.

---

## How to Run

Install required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn torch
