import numpy as np
import pandas as pd
from scipy.stats import norm

# Step 1: Load the data
file_path = '579004_579091.xlsx'
data = pd.read_excel(file_path, sheet_name='Data')
portfolio_returns = data['Portfolio'].values

# Calculate the mean and standard deviation of the portfolio returns (as a check)
mean_portfolio = np.mean(portfolio_returns)
std_portfolio = np.std(portfolio_returns)

# Calculate means and variance of the two stocks
mean_stock1 = np.mean(data['Stock1'].values)
var_stock1 = np.var(data['Stock1'].values)
print(f"Mean Stock1: {mean_stock1}")
print(f"Standard Deviation Stock1: {var_stock1}")

mean_stock2 = np.mean(data['Stock2'].values)
var_stock2 = np.var(data['Stock2'].values)
print(f"Mean Stock2: {mean_stock2}")
print(f"Standard Deviation Stock2: {var_stock2}")

# Calculate the correlation between the two stocks (unusee)
correlation = np.corrcoef(data['Stock1'].values, data['Stock2'].values)[0, 1]
# print(f"Correlation: {correlation}")

# Calculate the covariance between the two stocks
covariance = np.cov(data['Stock1'].values, data['Stock2'].values)[0, 1]
print(f"Covariance: {covariance}")

# Calculate the portfolio variance with 40% and 60% weights for Stock1 and Stock2, respectively
w1 = 0.4
w2 = 0.6
portfolio_mean = w1 * mean_stock1 + w2 * mean_stock2
print(f"Portfolio Mean: {portfolio_mean}")
portfolio_var = w1**2 * var_stock1 + w2**2 * var_stock2 + 2 * w1 * w2 * covariance
print(f"Portfolio Variance: {portfolio_var}")


# Calculate the 99% VaR using exact z-score
z_score_99 = norm.ppf(0.99)
print(f"Z-Score for 99% Confidence Level: {z_score_99}")
VaR_99 = portfolio_mean + z_score_99 * np.sqrt(portfolio_var)
print(f"99% VaR using variance-covariance method: {VaR_99}")