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

# 4) EVT Approach
k = 100
n = len(data)

data_weighted_sorted = data_weighted.copy()
data_weighted_sorted.iloc[:] = np.sort(data_weighted, axis=0)

log_sum = 0
for i in range(1, k + 1):
    log_sum += np.log(data_weighted_sorted.iloc[(n - i)])

tail_index_estimate = ((1 / k) * log_sum - np.log(data_weighted_sorted.iloc[(n - k - 1)]))**-1 # Hill estimator
print(f"Alpha hat = {tail_index_estimate}")

var_evt = data_weighted_sorted.iloc[(n - k - 1)] * ((k / (n * (1 - VAR_PERCENTAGE/100)))**(1 / tail_index_estimate))
print(f"VaRs at {VAR_PERCENTAGE}% using the EVT approach with k = {k}: {var_evt}")