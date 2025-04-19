"""
Poisson Mixture Models for Extreme Value Analysis.

This module implements Poisson mixture models for count data with extreme values,
including both simple mixture and regression approaches.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
from scipy.stats import poisson
from .base import BaseModel

class PoissonMixture(BaseModel):
    """
    Poisson Mixture Model for count data with extreme values.
    
    This class implements a mixture of Poisson distributions to model count data
    with potential extreme values. The model assumes that the data comes from
    a mixture of two Poisson distributions with different rates.
    
    Parameters
    ----------
    data : np.ndarray
        Array of count observations
    n_iterations : int, optional
        Number of MCMC iterations, by default 10000
    burn_in : int, optional
        Number of burn-in iterations, by default 1000
    rate_prior : Tuple[float, float], optional
        Prior parameters for rate parameters (alpha, beta), by default (1, 1)
    weight_prior : Tuple[float, float], optional
        Prior parameters for mixture weight (alpha, beta), by default (1, 1)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        n_iterations: int = 10000,
        burn_in: int = 1000,
        rate_prior: Tuple[float, float] = (1, 1),
        weight_prior: Tuple[float, float] = (1, 1),
    ):
        super().__init__(data, n_iterations, burn_in)
        self.rate_prior = rate_prior
        self.weight_prior = weight_prior
        
        # Initialize parameters
        self.rate1 = np.mean(data)  # Rate for first component
        self.rate2 = np.max(data)   # Rate for second component
        self.weight = 0.5           # Mixture weight
        
        # Initialize traces
        self.rate1_trace = []
        self.rate2_trace = []
        self.weight_trace = []
        
    def log_likelihood(self) -> float:
        """Compute the log-likelihood of the Poisson mixture model."""
        log_lik = 0.0
        for x in self.data:
            log_lik += np.log(
                self.weight * poisson.pmf(x, self.rate1) +
                (1 - self.weight) * poisson.pmf(x, self.rate2)
            )
        return log_lik
    
    def log_prior(self) -> float:
        """Compute the log-prior of the Poisson mixture model."""
        log_prior = 0.0
        # Rate parameters (gamma prior)
        log_prior += (self.rate_prior[0] - 1) * np.log(self.rate1) - self.rate_prior[1] * self.rate1
        log_prior += (self.rate_prior[0] - 1) * np.log(self.rate2) - self.rate_prior[1] * self.rate2
        # Mixture weight (beta prior)
        log_prior += (self.weight_prior[0] - 1) * np.log(self.weight)
        log_prior += (self.weight_prior[1] - 1) * np.log(1 - self.weight)
        return log_prior
    
    def metropolis_step(self, param: str) -> bool:
        """
        Perform a Metropolis-Hastings step for a given parameter.
        
        Parameters
        ----------
        param : str
            Parameter to update ('rate1', 'rate2', or 'weight')
            
        Returns
        -------
        bool
            Whether the proposal was accepted
        """
        current_value = getattr(self, param)
        if param == 'weight':
            proposal = np.random.beta(2, 2)  # Beta proposal for weight
        else:
            proposal = np.random.gamma(current_value, 1)  # Gamma proposal for rates
        
        # Store current value and set proposal
        setattr(self, param, proposal)
        proposal_log_posterior = self.log_likelihood() + self.log_prior()
        
        # Restore current value
        setattr(self, param, current_value)
        current_log_posterior = self.log_likelihood() + self.log_prior()
        
        # Accept or reject
        log_ratio = proposal_log_posterior - current_log_posterior
        if np.log(np.random.random()) < log_ratio:
            setattr(self, param, proposal)
            return True
        return False
    
    def fit(self) -> None:
        """Fit the Poisson mixture model using MCMC."""
        for _ in range(self.n_iterations):
            # Update parameters
            self.metropolis_step('rate1')
            self.metropolis_step('rate2')
            self.metropolis_step('weight')
            
            # Store traces after burn-in
            if _ >= self.burn_in:
                self.rate1_trace.append(self.rate1)
                self.rate2_trace.append(self.rate2)
                self.weight_trace.append(self.weight)
    
    def predict_probability(self, x: int) -> float:
        """
        Predict the probability of observing a count x.
        
        Parameters
        ----------
        x : int
            Count value to predict probability for
            
        Returns
        -------
        float
            Predicted probability
        """
        return (
            np.mean(self.weight_trace) * poisson.pmf(x, np.mean(self.rate1_trace)) +
            (1 - np.mean(self.weight_trace)) * poisson.pmf(x, np.mean(self.rate2_trace))
        )
    
    @property
    def parameter_estimates(self) -> dict:
        """Get the parameter estimates from the MCMC chains."""
        return {
            'rate1': np.mean(self.rate1_trace),
            'rate2': np.mean(self.rate2_trace),
            'weight': np.mean(self.weight_trace)
        }
    
    @property
    def parameter_credible_intervals(self) -> dict:
        """Get the 95% credible intervals for the parameters."""
        return {
            'rate1': np.percentile(self.rate1_trace, [2.5, 97.5]),
            'rate2': np.percentile(self.rate2_trace, [2.5, 97.5]),
            'weight': np.percentile(self.weight_trace, [2.5, 97.5])
        }

class PoissonMixtureRegression(BaseModel):
    """
    Poisson Mixture Regression Model for count data with covariates.
    
    This class implements a regression model where the rate parameters of a
    Poisson mixture are modeled as functions of covariates.
    
    Parameters
    ----------
    data : np.ndarray
        Array of count observations
    covariates : np.ndarray
        Array of covariate values
    n_iterations : int, optional
        Number of MCMC iterations, by default 10000
    burn_in : int, optional
        Number of burn-in iterations, by default 1000
    rate_prior : Tuple[float, float], optional
        Prior parameters for rate parameters (alpha, beta), by default (1, 1)
    weight_prior : Tuple[float, float], optional
        Prior parameters for mixture weight (alpha, beta), by default (1, 1)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        covariates: np.ndarray,
        n_iterations: int = 10000,
        burn_in: int = 1000,
        rate_prior: Tuple[float, float] = (1, 1),
        weight_prior: Tuple[float, float] = (1, 1),
    ):
        super().__init__(data, n_iterations, burn_in)
        self.covariates = covariates
        self.rate_prior = rate_prior
        self.weight_prior = weight_prior
        
        # Initialize parameters
        self.beta1 = np.zeros(covariates.shape[1])  # Coefficients for first component
        self.beta2 = np.zeros(covariates.shape[1])  # Coefficients for second component
        self.weight = 0.5                          # Mixture weight
        
        # Initialize traces
        self.beta1_trace = []
        self.beta2_trace = []
        self.weight_trace = []
        
    def compute_rates(self, beta: np.ndarray) -> np.ndarray:
        """Compute rates from covariates and coefficients."""
        return np.exp(self.covariates @ beta)
    
    def log_likelihood(self) -> float:
        """Compute the log-likelihood of the Poisson mixture regression model."""
        rates1 = self.compute_rates(self.beta1)
        rates2 = self.compute_rates(self.beta2)
        
        log_lik = 0.0
        for i, x in enumerate(self.data):
            log_lik += np.log(
                self.weight * poisson.pmf(x, rates1[i]) +
                (1 - self.weight) * poisson.pmf(x, rates2[i])
            )
        return log_lik
    
    def log_prior(self) -> float:
        """Compute the log-prior of the Poisson mixture regression model."""
        log_prior = 0.0
        # Coefficient parameters (normal prior)
        log_prior += -0.5 * np.sum(self.beta1**2)
        log_prior += -0.5 * np.sum(self.beta2**2)
        # Mixture weight (beta prior)
        log_prior += (self.weight_prior[0] - 1) * np.log(self.weight)
        log_prior += (self.weight_prior[1] - 1) * np.log(1 - self.weight)
        return log_prior
    
    def metropolis_step(self, param: str) -> bool:
        """
        Perform a Metropolis-Hastings step for a given parameter.
        
        Parameters
        ----------
        param : str
            Parameter to update ('beta1', 'beta2', or 'weight')
            
        Returns
        -------
        bool
            Whether the proposal was accepted
        """
        if param == 'weight':
            current_value = self.weight
            proposal = np.random.beta(2, 2)
            self.weight = proposal
        else:
            current_value = getattr(self, param)
            proposal = current_value + np.random.normal(0, 0.1, size=current_value.shape)
            setattr(self, param, proposal)
        
        proposal_log_posterior = self.log_likelihood() + self.log_prior()
        
        if param == 'weight':
            self.weight = current_value
        else:
            setattr(self, param, current_value)
        current_log_posterior = self.log_likelihood() + self.log_prior()
        
        log_ratio = proposal_log_posterior - current_log_posterior
        if np.log(np.random.random()) < log_ratio:
            if param == 'weight':
                self.weight = proposal
            else:
                setattr(self, param, proposal)
            return True
        return False
    
    def fit(self) -> None:
        """Fit the Poisson mixture regression model using MCMC."""
        for _ in range(self.n_iterations):
            # Update parameters
            self.metropolis_step('beta1')
            self.metropolis_step('beta2')
            self.metropolis_step('weight')
            
            # Store traces after burn-in
            if _ >= self.burn_in:
                self.beta1_trace.append(self.beta1.copy())
                self.beta2_trace.append(self.beta2.copy())
                self.weight_trace.append(self.weight)
    
    def predict(self, covariates: np.ndarray) -> np.ndarray:
        """
        Predict expected counts for new covariates.
        
        Parameters
        ----------
        covariates : np.ndarray
            New covariate values
            
        Returns
        -------
        np.ndarray
            Predicted expected counts
        """
        beta1_mean = np.mean(self.beta1_trace, axis=0)
        beta2_mean = np.mean(self.beta2_trace, axis=0)
        weight_mean = np.mean(self.weight_trace)
        
        rates1 = np.exp(covariates @ beta1_mean)
        rates2 = np.exp(covariates @ beta2_mean)
        
        return weight_mean * rates1 + (1 - weight_mean) * rates2
    
    @property
    def parameter_estimates(self) -> dict:
        """Get the parameter estimates from the MCMC chains."""
        return {
            'beta1': np.mean(self.beta1_trace, axis=0),
            'beta2': np.mean(self.beta2_trace, axis=0),
            'weight': np.mean(self.weight_trace)
        }
    
    @property
    def parameter_credible_intervals(self) -> dict:
        """Get the 95% credible intervals for the parameters."""
        return {
            'beta1': np.percentile(self.beta1_trace, [2.5, 97.5], axis=0),
            'beta2': np.percentile(self.beta2_trace, [2.5, 97.5], axis=0),
            'weight': np.percentile(self.weight_trace, [2.5, 97.5])
        } 