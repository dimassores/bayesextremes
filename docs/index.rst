Welcome to BayesExtremes's documentation!
====================================

BayesExtremes is a Python package for Bayesian analysis of extreme values. It provides implementations of various Bayesian models for extreme value analysis, including:

- Gamma Mixture Model
- Generalized Extreme Value Distribution
- Gamma Extreme Value Estimation
- Poisson Mixture Model
- Poisson Mixture Regression Model

Installation
-----------

.. code-block:: bash

    pip install bayesextremes

Quick Start
----------

Here's a quick example of how to use the package:

.. code-block:: python

    import numpy as np
    from bayesextremes import GammaMixtureModel

    # Generate some example data
    data = np.random.gamma(shape=2, scale=1, size=1000)

    # Initialize and fit the model
    model = GammaMixtureModel(n_iterations=1000, data=data, k=2)
    model.fit()

    # Get parameter estimates
    chains = model.get_parameter_chains()
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