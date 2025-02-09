import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(0)

WEIGHT_1 = 0.4
WEIGHT_2 = 0.6

VAR_PERCENTAGE = 99

data = pd.read_excel(r"579004_579091.xlsx").dropna(axis=1)
data

# 6) Reverse Clayton copula

df_loss = data.copy().mul(-1)

# Fit Marginals to t-distributions
def fit_student_t(data):
    df, loc, scale = stats.t.fit(data, method="MLE")
    return df, loc, scale

df1, loc1, scale1 = fit_student_t(df_loss.iloc[:,0])
df2, loc2, scale2 = fit_student_t(df_loss.iloc[:,1])

# Transform data to uniform [0,1] using CDF of fitted t-distributions
U1 = stats.t.cdf(df_loss.iloc[:,0], df=df1, loc=loc1, scale=scale1)
U2 = stats.t.cdf(df_loss.iloc[:,1], df=df2, loc=loc2, scale=scale2)

# Reverse Clayton copula transformation
V1, V2 = 1 - U1, 1 - U2

# Estimate Kendall’s Tau and fit reverse Clayton copula using method of moments
tau = stats.kendalltau(V1, V2)[0]  
theta = 2 * tau / (1 - tau)  

# Validate using Upper Tail Dependence
data_sorted = data.copy()
data_sorted.iloc[:] = np.sort(data, axis=0)
n = len(data)

lambdas = []
for k in range(1, 400):
    tail_dependence_sum = 0
    for i in range(1, n + 1):
        if data.iloc[(i - 1), 0] > data_sorted.iloc[(n - k - 1), 0] and data.iloc[(i - 1), 1] > data_sorted.iloc[(n - k - 1), 1]:
            tail_dependence_sum += 1

    lambda_U_empirical = (1 / k) * tail_dependence_sum
    lambdas.append(lambda_U_empirical)

plt.figure(figsize=(16,9))
plt.plot(lambdas)
plt.xlabel("k")
plt.ylabel("Empirical Lambda")
plt.show()

lambda_U_theoretical = 2 ** (-1 / theta)

print(f"Fitted Clayton parameter theta using method of moments: {theta:.4f}")
print(f"Empirical upper tail dependence: {lambda_U_empirical:.4f}")
print(f"Theoretical upper tail dependence: {lambda_U_theoretical:.4f}")

def clayton_conditional_inverse(u, t, theta):
    v = (u**(-theta) * t**(-theta / (1 + theta)) - u**(-theta) + 1)**(-1 / theta)
    return v

# Generate uniform samples from Clayton copula
n_sim = 1000000
S = np.random.uniform(size=(n_sim, 2))  # Independent uniform variable

# Compute dependent uniforms
U1 = S[:,0]
U2 = np.array([clayton_conditional_inverse(u, t, theta) for u, t in S])

# Reverse Clayton transformations
V1 = 1 - U1
V2 = 1 - U2

print(f"Min(1 - V1): {np.min(1 - V1)}, Max(1 - V1): {np.max(1 - V1)}")
print(f"Min(1 - V2): {np.min(1 - V2)}, Max(1 - V2): {np.max(1 - V2)}")

# Transform to t-distributions
X1_sim = stats.t.ppf(V1, df=df1, loc=loc1, scale=scale1)  # Reverse CDF
X2_sim = stats.t.ppf(V2, df=df2, loc=loc2, scale=scale2)  # Reverse CDF

# Compute portfolio losses
L_portfolio = WEIGHT_1 * X1_sim + WEIGHT_2 * X2_sim
print(L_portfolio)

# Compute 99% VaR 
#idx = int(np.ceil(len(L_portfolio) * VAR_PERCENTAGE / 100)) - 1
#var = np.sort(L_portfolio)[idx]

var = np.percentile(L_portfolio, VAR_PERCENTAGE)

print(f"Portfolio 99% VaR: {var:.4f}")
