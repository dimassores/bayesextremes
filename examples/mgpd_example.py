"""
Example script demonstrating the use of the MGPD model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bayesextremes.models import MGPD

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
# Generate bulk data from a mixture of Gamma distributions
alpha1, beta1 = 2.0, 1.0  # First Gamma component
alpha2, beta2 = 5.0, 0.5  # Second Gamma component
weights = [0.6, 0.4]      # Mixture weights

data1 = np.random.gamma(alpha1, 1/beta1, int(n_samples * weights[0]))
data2 = np.random.gamma(alpha2, 1/beta2, int(n_samples * weights[1]))
bulk_data = np.concatenate([data1, data2])

# Generate tail data from a Generalized Pareto Distribution
xi = 0.5  # Shape parameter
sigma = 1.0  # Scale parameter
u = np.percentile(bulk_data, 90)  # Threshold at 90th percentile
tail_data = u + np.random.pareto(xi, 200) * sigma

# Combine bulk and tail data
data = np.concatenate([bulk_data, tail_data])

# Plot the data
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=50, stat='density')
plt.axvline(u, color='r', linestyle='--', label='Threshold')
plt.title('Synthetic Data with Extreme Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Initialize and fit the MGPD model
model = MGPD(data, n_iterations=10000, burn_in=1000)
model.fit()

# Print parameter estimates
print("Parameter Estimates:")
print(f"Shape parameter (ξ): {model.xi:.3f}")
print(f"Scale parameter (σ): {model.sigma:.3f}")
print(f"Threshold (u): {model.u:.3f}")

# Plot the fitted model
x = np.linspace(0, 20, 1000)
density = model.predict_density(x)

plt.figure(figsize=(10, 6))
sns.histplot(data, bins=50, stat='density', alpha=0.5)
plt.plot(x, density, 'r-', linewidth=2)
plt.axvline(model.u, color='g', linestyle='--', label='Estimated Threshold')
plt.title('Fitted MGPD Model')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend(['Fitted Density', 'Data', 'Estimated Threshold'])
plt.show()

# Model diagnostics
plt.figure(figsize=(15, 10))
for i, param in enumerate(['xi', 'sigma', 'u']):
    plt.subplot(3, 1, i+1)
    plt.plot(getattr(model, f'{param}_trace'))
    plt.title(f'{param} Trace')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Print acceptance rates
print("\nAcceptance Rates:")
for param in ['xi', 'sigma', 'u']:
    print(f"{param}: {getattr(model, f'{param}_acceptance_rate'):.3f}")

# Compute and plot return levels
return_periods = np.array([10, 50, 100, 500])
return_levels = model.predict_return_levels(return_periods)

plt.figure(figsize=(10, 6))
plt.plot(return_periods, return_levels, 'o-')
plt.title('Return Level Plot')
plt.xlabel('Return Period (years)')
plt.ylabel('Return Level')
plt.grid(True)
plt.show() 