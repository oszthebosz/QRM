import numpy as np
import pandas as pd
from scipy.stats import norm

# Step 1: Load the data
file_path = '579004_579091.xlsx'
data = pd.read_excel(file_path, sheet_name='Data')

# Calculate VaR based on varinance-covariance method
# Extract the loss returns for the portfolio
portfolio_returns = data['Portfolio'].values  # Assuming 'Portfolio' column in

# Calculate the mean and standard deviation of the portfolio returns
mean_portfolio = np.mean(portfolio_returns)
std_portfolio = np.std(portfolio_returns)

# Calculate means and standard deviations of the two stocks
mean_stock1 = np.mean(data['Stock1'].values)
std_stock1 = np.std(data['Stock1'].values)

mean_stock2 = np.mean(data['Stock2'].values)
std_stock2 = np.std(data['Stock2'].values)

# Calculate the correlation between the two stocks
correlation = np.corrcoef(data['Stock1'].values, data['Stock2'].values)[0, 1]

# Calculate the portfolio standard deviation using the formula
# std_portfolio = sqrt(w1^2 * std1^2 + w2^2 * std2^2 + 2 * w1 * w2 * std1 * std2 * correlation)
# With 40 and 60% weights for Stock1 and Stock2 respectively
w1 = 0.4
w2 = 0.6
portfolio_std = np.sqrt(w1**2 * std_stock1**2 + w2**2 * std_stock2**2 + 2 * w1 * w2 * std_stock1 * std_stock2 * correlation)
portfolio_mean = w1 * mean_stock1 + w2 * mean_stock2
print(f"Portfolio Mean: {portfolio_mean}")

# Calculate the 99% VaR using the variance-covariance method
# Use exact z-score for 99% confidence level
z_score_99 = norm.ppf(0.99)
VaR_99 = portfolio_mean - z_score_99 * portfolio_std

# Calculate the 99% VaR using the variance-covariance method
# Use exact z-score for 99% confidence level

print(f"99% VaR using variance-covariance method: {VaR_99}")