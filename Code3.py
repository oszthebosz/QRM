import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from sklearn.mixture import GaussianMixture

# Load the data
file_path = '579004_579091.xlsx'
data = pd.read_excel(file_path, sheet_name='Data')

# Extract the loss returns for the two stocks
stock1_returns = data['Stock1'].values
stock2_returns = data['Stock2'].values

# Conduct the JB tests
JB_stat_1, p_value_stock1 = jarque_bera(stock1_returns)
JB_stat_2, p_value_stock2 = jarque_bera(stock2_returns)

print(f"JB Test statistic for Stock 1: {JB_stat_1}")
print(f"JB Test statistic for Stock 2: {JB_stat_2}")
print(f"JB Test p-value for Stock 1: {p_value_stock1}")
print(f"JB Test p-value for Stock 2: {p_value_stock2}")

# Check normality of the portfolio return
weights = np.array([0.4, 0.6])
portfolio_returns = weights[0] * stock1_returns + weights[1] * stock2_returns
# print(portfolio_returns[:10]) # is a check
JB_stat_port, p_value_portfolio = jarque_bera(portfolio_returns)
print(f"JB Test statistic for Portfolio: {JB_stat_port}")
print(f"JB Test p-value for Portfolio: {p_value_portfolio}")

# Fit a GMM
gmm = GaussianMixture(n_components=2, tol = 1e-10, max_iter=100000) # adjuted tol and max_iter to avoid local minima
gmm.fit(portfolio_returns.reshape(-1, 1)) # Uses EM-algorithm to fit the GMM

# Extract the means, variances, and weights of the two normal distributions in the mixture
means = gmm.means_.flatten()
covariances = gmm.covariances_.flatten()
st_devs = np.sqrt(covariances)
weights = gmm.weights_
print(f'Means: {means}, St. Deviations: {st_devs}, Weights: {weights}')

# Get 99% quantile
quantile_99 = np.percentile(portfolio_returns, 99)  # using np.percentile
print(f"99% VaR from historical data: {quantile_99}")
quantile_by_hand = np.sort(portfolio_returns)[1979]  # using indexing
print(f"99% VaR from historical data by hand: {np.sort(portfolio_returns)[1977:1982]}") # Middle element

# Given parameters for the normal mixture
pi = weights[0]
mu1, sigma1 = means[0], st_devs[0]
mu2, sigma2 = means[1], st_devs[1]
N = 10000000 # To get better estimate

# Simulate losses
rand_vals = np.random.rand(N)  # uniform random numbers
losses = np.where(rand_vals < pi, 
                  np.random.normal(mu1, sigma1, N), 
                  np.random.normal(mu2, sigma2, N))

# Compute VaR at 99% confidence level
VaR_99 = np.percentile(losses, 99)
print(f"Estimated 99% VaR: {VaR_99:.5f}")
