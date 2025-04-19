"""
Poisson Mixture Models for Extreme Value Analysis.

This module implements Poisson mixture models for count data with extreme values,
including both simple mixture and regression approaches.
"""

from typing import Optional, Tuple, Union, List, Dict, Any
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
    
    def _metropolis_step(self, param: str) -> bool:
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
        for i in range(self.n_iterations):
            # Update parameters
            self._metropolis_step('rate1')
            self._metropolis_step('rate2')
            self._metropolis_step('weight')
            
            # Update traces
            self._update_trace('rate1', self.rate1, i)
            self._update_trace('rate2', self.rate2, i)
            self._update_trace('weight', self.weight, i)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the probability density for new data.
        
        Parameters
        ----------
        x : np.ndarray
            New data points
            
        Returns
        -------
        np.ndarray
            Predicted probability densities
        """
        return np.array([
            self.weight * poisson.pmf(xi, self.rate1) +
            (1 - self.weight) * poisson.pmf(xi, self.rate2)
            for xi in x
        ])
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model fit including component probabilities.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics including parameter estimates and component probabilities
        """
        summary = super().get_summary()
        
        # Add component probabilities for common values
        common_values = [0, 1, 2, 5, 10]
        summary['component_probabilities'] = {
            f'x={x}': {
                'component1': poisson.pmf(x, self.rate1),
                'component2': poisson.pmf(x, self.rate2)
            }
            for x in common_values
        }
        
        return summary

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
    
    def _metropolis_step(self, param: str) -> bool:
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
        current_value = getattr(self, param)
        if param == 'weight':
            proposal = np.random.beta(2, 2)  # Beta proposal for weight
        else:
            proposal = np.random.normal(current_value, 0.1)  # Normal proposal for coefficients
        
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
        """Fit the Poisson mixture regression model using MCMC."""
        for i in range(self.n_iterations):
            # Update parameters
            self._metropolis_step('beta1')
            self._metropolis_step('beta2')
            self._metropolis_step('weight')
            
            # Update traces
            self._update_trace('beta1', self.beta1, i)
            self._update_trace('beta2', self.beta2, i)
            self._update_trace('weight', self.weight, i)
    
    def predict(self, covariates: np.ndarray) -> np.ndarray:
        """
        Predict the probability density for new data.
        
        Parameters
        ----------
        covariates : np.ndarray
            New covariate values
            
        Returns
        -------
        np.ndarray
            Predicted probability densities
        """
        rates1 = np.exp(covariates @ self.beta1)
        rates2 = np.exp(covariates @ self.beta2)
        
        return np.array([
            self.weight * poisson.pmf(x, r1) + (1 - self.weight) * poisson.pmf(x, r2)
            for x, r1, r2 in zip(self.data, rates1, rates2)
        ])
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model fit including coefficient estimates.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics including coefficient estimates and mixture weight
        """
        summary = super().get_summary()
        
        # Add coefficient interpretations
        summary['coefficient_interpretations'] = {
            'beta1': {
                'mean': np.mean(self.beta1),
                'effect': f"1 unit increase in covariate multiplies rate by {np.exp(np.mean(self.beta1)):.2f}"
            },
            'beta2': {
                'mean': np.mean(self.beta2),
                'effect': f"1 unit increase in covariate multiplies rate by {np.exp(np.mean(self.beta2)):.2f}"
            }
        }
        
        return summary 