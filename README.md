# BayesExtremes

A Python package for Bayesian analysis of extreme values.

## Overview

BayesExtremes is a comprehensive package for Bayesian analysis of extreme values, implementing various models for extreme value analysis including:

- Generalized Extreme Value (GEV) distribution
- Mixed Gamma-Generalized Pareto Distribution (MGPD)
- Gamma Mixture Model
- Poisson Mixture Model
- Poisson Mixture Regression Model

## Features

- Bayesian inference using MCMC methods
- Automatic threshold selection
- Return level prediction
- Parameter estimation with credible intervals
- Comprehensive test coverage
- Type hints and documentation
- Code quality checks with pre-commit hooks
- Continuous integration with GitHub Actions

## Installation

```bash
pip install bayesextremes
```

## Usage

### GEV Model

```python
from bayesextremes.models import GEV
import numpy as np

# Generate sample data
data = np.random.genextreme(0.5, loc=0, scale=1, size=100)

# Initialize and fit the model
model = GEV(data)
model.fit()

# Get parameter estimates
estimates = model.parameter_estimates
print(f"Shape parameter (xi): {estimates['xi']:.3f}")
print(f"Scale parameter (sigma): {estimates['sigma']:.3f}")
print(f"Location parameter (mu): {estimates['mu']:.3f}")

# Predict return levels
return_level = model.predict_return_level(100)  # 100-year return level
print(f"100-year return level: {return_level:.3f}")
```

### MGPD Model

```python
from bayesextremes.models import MGPD
import numpy as np
from scipy.stats import gamma, genpareto

# Generate sample data from mixture of Gamma and GPD
bulk_data = np.concatenate([
    gamma.rvs(a=2, scale=1, size=450),
    gamma.rvs(a=5, scale=2, size=450)
])
tail_data = genpareto.rvs(c=0.5, loc=0, scale=1, size=100)
data = np.concatenate([bulk_data, tail_data])

# Initialize and fit the model
model = MGPD(data)
model.fit()

# Get parameter estimates
estimates = model.parameter_estimates
print(f"Shape parameter (xi): {estimates['xi']:.3f}")
print(f"Scale parameter (sigma): {estimates['sigma']:.3f}")
print(f"Threshold (u): {estimates['u']:.3f}")

# Predict return levels
return_level = model.predict_return_level(100)  # 100-year return level
print(f"100-year return level: {return_level:.3f}")
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dimassores/bayesextremes.git
cd bayesextremes
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

### Testing

Run the test suite:
```bash
pytest
```

### Documentation

Build the documentation:
```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{bayesextremes,
  author = {Dimas Lima},
  title = {BayesExtremes: A Python package for Bayesian analysis of extreme values},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dimassores/bayesextremes}
}
```

## References

1. do Nascimento, F.F., Gamerman, D. and Lopes, H.F., 2012. A semiparametric Bayesian approach to extreme value estimation. Statistics and Computing, 22(2), pp.661-675.

2. De Haan, L., Ferreira, A. and Ferreira, A., 2006. Extreme value theory: an introduction (Vol. 21). New York: Springer.

3. Coles, S., 2001. An introduction to statistical modeling of extreme values. London: Springer.
