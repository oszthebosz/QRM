import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Step 1: Load the data (assuming the Excel file is named 'portfolio_data.xlsx')
file_path = '579004_579091.xlsx'
data = pd.read_excel(file_path, sheet_name='Data')

# Extract the loss returns for the two stocks
stock1_returns = data['Stock1'].values  # Assuming 'Stock1' column in the Data sheet
stock2_returns = data['Stock2'].values  # Assuming 'Stock2' column in the Data sheet

# Step 2: Conduct the JB test for normality
JB_stat_1, p_value_stock1 = jarque_bera(stock1_returns)
JB_stat_2, p_value_stock2 = jarque_bera(stock2_returns)

print(f"JB Test statistic for Stock 1: {JB_stat_1}")
print(f"JB Test statistic for Stock 2: {JB_stat_2}")
print(f"JB Test p-value for Stock 1: {p_value_stock1}")
print(f"JB Test p-value for Stock 2: {p_value_stock2}")

# Check normality of the portfolio return
weights = np.array([0.4, 0.6])
portfolio_returns = weights[0] * stock1_returns + weights[1] * stock2_returns
print(portfolio_returns[:10])

JB_stat_port, p_value_portfolio = jarque_bera(portfolio_returns)

print(f"JB Test statistic for Portfolio: {JB_stat_port}")
print(f"JB Test p-value for Portfolio: {p_value_portfolio}")

# Step 3: Fit a Normal Mixture Model
# Fit a 2-component Gaussian Mixture model to the portfolio returns
gmm = GaussianMixture(n_components=2)
gmm.fit(portfolio_returns.reshape(-1, 1))

# Step 4: Calculate the 99% VaR
# We need the 1% quantile of the fitted mixture model (which corresponds to VaR at 99% confidence level)
# GMM gives us the probability density, so we need to calculate the quantile

# Extract the means, variances, and weights of the two normal distributions in the mixture
means = gmm.means_.flatten()
covariances = gmm.covariances_.flatten()
weights = gmm.weights_
print(f'Means: {means}, Variances: {covariances}, Weights: {weights}')

# Print the probabilities of normalities for the two components
probabilities = gmm.predict_proba(portfolio_returns.reshape(-1, 1))
print(f'Probabilities: {probabilities}')

# To get the 1% quantile, we first calculate the weighted cumulative distribution
quantile_99 = np.percentile(portfolio_returns, 1)  # The 1% quantile directly from the data

# Step 5: Display results
print(f"99% VaR from normal mixture model: {quantile_99}")

# Optional: Plot the portfolio loss distribution and GMM density
x = np.linspace(min(portfolio_returns), max(portfolio_returns), 1000)
gmm_density = np.exp(gmm.score_samples(x.reshape(-1, 1)))

plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns, bins=50, density=True, alpha=0.6, color='g', label='Portfolio Returns')
plt.plot(x, gmm_density, label='Normal Mixture Model', color='r', linewidth=2)
plt.title('Portfolio Loss Distribution and Normal Mixture Model')
plt.legend()
plt.show()
