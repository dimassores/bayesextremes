# BayesExtremes

A Python package for Bayesian analysis of extreme values.

## Overview

BayesExtremes is a comprehensive package for Bayesian analysis of extreme values, implementing various models for extreme value analysis including:

- Mixed Gamma-Pareto Distribution Model
- Generalized Extreme Value (GEV) distribution
- Poisson Mixture Model
- Poisson Mixture Regression Model

All models are built on a common base class that provides shared functionality for Bayesian inference, including MCMC sampling and parameter estimation.

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

### Mixed Gamma-Pareto Model

```python
import numpy as np
from bayesextremes.models import MixedGammaParetoModel

# Generate some example data
data = np.random.gamma(shape=2, scale=1, size=1000)

# Initialize model with prior values
prior_values = {
    'a_mu': np.array([2.0, 3.0]),
    'b_mu': np.array([1.0, 1.0]),
    'c_eta': np.array([2.0, 2.0]),
    'd_eta': np.array([1.0, 1.0]),
    'alpha_p': np.array([1.0, 1.0]),
    'mu_u': np.percentile(data, 90),
    'sigma_u': 1.0
}

# Initialize and fit the model
model = MixedGammaParetoModel(
    data=data,
    k=2,
    prior_values=prior_values,
    n_iterations=1000,
    burn_in=100,
    thin=10
)
model.fit()

# Get parameter estimates and summary statistics
summary = model.get_summary()
print(f"Shape parameters: {summary['etas']['mean']}")
print(f"Scale parameters: {summary['mus']['mean']}")
print(f"Mixing weights: {summary['ps']['mean']}")
```

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

### Poisson Mixture Model

```python
from bayesextremes.models import PoissonMixture
import numpy as np

# Generate sample data
data = np.random.poisson(lam=[5, 10], size=(100, 2))

# Initialize and fit the model
model = PoissonMixture(data, k=2)
model.fit()

# Get parameter estimates
summary = model.get_summary()
print(f"Mixture weights: {summary['ps']['mean']}")
print(f"Rate parameters: {summary['lambdas']['mean']}")
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
pytest tests/
```

### Documentation

Build the documentation:
```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
