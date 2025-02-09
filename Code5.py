import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

# Load data
file_path = "579004_579091.xlsx"
data = pd.read_excel(file_path, sheet_name="Data")
portfolio_losses = data['Portfolio']

# Fit a GARCH(1,1) model with normal innovations
garch_model = arch_model(portfolio_losses, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp="off")

# Extract model parameters
mu = garch_fit.params['mu']
omega = garch_fit.params['omega']
alpha1 = garch_fit.params['alpha[1]']
beta1 = garch_fit.params['beta[1]']

# Forecast next-day volatility
garch_forecast = garch_fit.forecast(horizon=1) # next day volatility 
predicted_volatility = np.sqrt(garch_forecast.variance.iloc[-1, 0]) # as standard deviation

# Historical Simulation (HS) VaR for residuals
residuals = garch_fit.resid / garch_fit.conditional_volatility  # Standardized residuals
hs_var_error = np.percentile(residuals, 99)  # 99% quantile of standardized residuals

# Predicted VaR(99%) for portfolio losses
predicted_var_99 = mu + predicted_volatility * hs_var_error # using formula from the slides

# Print results
print("GARCH(1,1) Model Parameters:")
print(f"mu: {mu:.6f}")
print(f"omega: {omega:.6f}")
print(f"alpha1: {alpha1:.6f}")
print(f"beta1: {beta1:.6f}")

print("\nVaR Estimates:")
print(f"Predicted Volatility: {predicted_volatility:.6f}")
print(f"HS VaR for Error Term (99%): {hs_var_error:.6f}")
print(f"Predicted VaR(99%): {predicted_var_99:.6f}")