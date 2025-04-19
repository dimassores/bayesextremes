"""
Generalized Extreme Value (GEV) Distribution Model.

This module implements the GEV distribution for extreme value analysis using both
maximum likelihood and Bayesian approaches.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import genextreme
from .base import BaseModel

class GEV(BaseModel):
    """
    Generalized Extreme Value Distribution Model.
    
    This class implements the GEV distribution for extreme value analysis.
    The GEV distribution combines three types of extreme value distributions:
    Gumbel, Fréchet, and Weibull, through a shape parameter ξ.
    
    Parameters
    ----------
    data : np.ndarray
        Array of block maxima observations
    n_iterations : int, optional
        Number of MCMC iterations, by default 10000
    burn_in : int, optional
        Number of burn-in iterations, by default 1000
    shape_prior : Tuple[float, float], optional
        Prior parameters for shape parameter (mean, std), by default (0, 1)
    scale_prior : Tuple[float, float], optional
        Prior parameters for scale parameter (mean, std), by default (1, 1)
    loc_prior : Tuple[float, float], optional
        Prior parameters for location parameter (mean, std), by default (0, 1)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        n_iterations: int = 10000,
        burn_in: int = 1000,
        shape_prior: Tuple[float, float] = (0, 1),
        scale_prior: Tuple[float, float] = (1, 1),
        loc_prior: Tuple[float, float] = (0, 1),
    ):
        super().__init__(data, n_iterations, burn_in)
        self.shape_prior = shape_prior
        self.scale_prior = scale_prior
        self.loc_prior = loc_prior
        
        # Initialize parameters
        self.xi = 0.0  # Shape parameter
        self.sigma = 1.0  # Scale parameter
        self.mu = 0.0  # Location parameter
        
        # Initialize traces
        self.xi_trace = []
        self.sigma_trace = []
        self.mu_trace = []
        
    def log_likelihood(self) -> float:
        """Compute the log-likelihood of the GEV model."""
        return np.sum(genextreme.logpdf(
            self.data,
            c=self.xi,
            loc=self.mu,
            scale=self.sigma
        ))
    
    def log_prior(self) -> float:
        """Compute the log-prior of the GEV model."""
        log_prior = 0.0
        # Shape parameter prior (normal)
        log_prior += -0.5 * ((self.xi - self.shape_prior[0]) / self.shape_prior[1])**2
        # Scale parameter prior (log-normal)
        log_prior += -0.5 * ((np.log(self.sigma) - self.scale_prior[0]) / self.scale_prior[1])**2
        # Location parameter prior (normal)
        log_prior += -0.5 * ((self.mu - self.loc_prior[0]) / self.loc_prior[1])**2
        return log_prior
    
    def metropolis_step(self, param: str) -> bool:
        """
        Perform a Metropolis-Hastings step for a given parameter.
        
        Parameters
        ----------
        param : str
            Parameter to update ('xi', 'sigma', or 'mu')
            
        Returns
        -------
        bool
            Whether the proposal was accepted
        """
        current_value = getattr(self, param)
        current_log_posterior = self.log_likelihood() + self.log_prior()

        # Propose new value
        if param == 'sigma':
            proposal = np.abs(np.random.normal(current_value, 0.1))
        else:
            proposal = np.random.normal(current_value, 0.1)

        # Apply proposal
        setattr(self, param, proposal)
        proposal_log_posterior = self.log_likelihood() + self.log_prior()

        # Decide to accept/reject
        if np.log(np.random.random()) < proposal_log_posterior - current_log_posterior:
            return True
        else:
            setattr(self, param, current_value)
            return False
    
    def fit(self) -> None:
        """Fit the GEV model using MCMC."""
        for _ in range(self.n_iterations):
            # Update parameters
            self.metropolis_step('xi')
            self.metropolis_step('sigma')
            self.metropolis_step('mu')
            
            # Store traces after burn-in
            if _ >= self.burn_in:
                self.xi_trace.append(self.xi)
                self.sigma_trace.append(self.sigma)
                self.mu_trace.append(self.mu)
    
    def predict_return_level(self, return_period: float) -> float:
        """
        Predict the return level for a given return period.
        
        Parameters
        ----------
        return_period : float
            Return period in years
            
        Returns
        -------
        float
            Predicted return level
        """
        if self.xi == 0:
            return self.mu - self.sigma * np.log(-np.log(1 - 1/return_period))
        return self.mu + self.sigma/self.xi * ((-np.log(1 - 1/return_period))**(-self.xi) - 1)
    
    @property
    def parameter_estimates(self) -> dict:
        """Get the parameter estimates from the MCMC chains."""
        return {
            'xi': np.mean(self.xi_trace),
            'sigma': np.mean(self.sigma_trace),
            'mu': np.mean(self.mu_trace)
        }
    
    @property
    def parameter_credible_intervals(self) -> dict:
        """Get the 95% credible intervals for the parameters."""
        return {
            'xi': np.percentile(self.xi_trace, [2.5, 97.5]),
            'sigma': np.percentile(self.sigma_trace, [2.5, 97.5]),
            'mu': np.percentile(self.mu_trace, [2.5, 97.5])
        } 