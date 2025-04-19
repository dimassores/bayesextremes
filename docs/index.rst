Welcome to BayesExtremes's documentation!
====================================

BayesExtremes is a Python package for Bayesian analysis of extreme values. It provides implementations of various Bayesian models for extreme value analysis, including:

- Mixed Gamma-Pareto Distribution Model
- Generalized Extreme Value Distribution
- Poisson Mixture Model
- Poisson Mixture Regression Model

All models are built on a common base class that provides shared functionality for Bayesian inference, including MCMC sampling and parameter estimation.

Installation
-----------

.. code-block:: bash

    pip install bayesextremes

Quick Start
----------

Here's a quick example of how to use the Mixed Gamma-Pareto Model:

.. code-block:: python

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

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   examples
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 