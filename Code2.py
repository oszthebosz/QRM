import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEIGHT_1 = 0.4
WEIGHT_2 = 0.6

VAR_PERCENTAGE = 99

data = pd.read_excel(r"579004_579091.xlsx").dropna(axis=1)
data

data_weighted = data.mul([WEIGHT_1, WEIGHT_2], axis=1).sum(axis=1).mul(-1)
data_weighted

# 2) Historical Simulation

var_99 = np.percentile(data_weighted, VAR_PERCENTAGE)  
print(f"VaR at {VAR_PERCENTAGE}% is: {var_99}")

data_weighted_sorted = data_weighted.copy()
data_weighted_sorted.iloc[:] = np.sort(data_weighted, axis=0)
idx = int(np.ceil(len(data_weighted_sorted) * VAR_PERCENTAGE / 100)) - 1
var_stocks = data_weighted_sorted.iloc[idx] 

print(var_stocks) # They dont agree, the percentile function uses interpolation, so use this one in the final answer (non-interpolated)
