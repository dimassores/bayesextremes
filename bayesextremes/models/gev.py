"""
Generalized Extreme Value (GEV) Distribution Model.

This module implements the GEV distribution for extreme value analysis using both
maximum likelihood and Bayesian approaches.
"""

from typing import Optional, Tuple, Union, Dict, Any
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
        # Check for invalid parameters
        if self.sigma <= 0:
            return -np.inf
            
        log_prior = 0.0
        try:
            # Shape parameter prior (normal)
            log_prior += -0.5 * ((self.xi - self.shape_prior[0]) / self.shape_prior[1])**2
            # Scale parameter prior (log-normal)
            log_prior += -0.5 * ((np.log(self.sigma) - self.scale_prior[0]) / self.scale_prior[1])**2
            # Location parameter prior (normal)
            log_prior += -0.5 * ((self.mu - self.loc_prior[0]) / self.loc_prior[1])**2
        except (ValueError, RuntimeWarning):
            return -np.inf
            
        return log_prior
    
    def _metropolis_step(self, param: str) -> bool:
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
        for i in range(self.n_iterations):
            # Update parameters
            self._metropolis_step('xi')
            self._metropolis_step('sigma')
            self._metropolis_step('mu')
            
            # Update traces
            self._update_trace('xi', self.xi, i)
            self._update_trace('sigma', self.sigma, i)
            self._update_trace('mu', self.mu, i)
    
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
        return genextreme.pdf(x, c=self.xi, loc=self.mu, scale=self.sigma)
    
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
            
        Raises
        ------
        ValueError
            If return_period is less than or equal to 1
        """
        if return_period <= 1:
            raise ValueError("return_period must be greater than 1")
            
        if self.xi == 0:
            return self.mu - self.sigma * np.log(-np.log(1 - 1/return_period))
        return self.mu + self.sigma/self.xi * ((-np.log(1 - 1/return_period))**(-self.xi) - 1)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model fit including return levels.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics including parameter estimates and return levels
        """
        summary = super().get_summary()
        
        # Add return levels for common return periods
        return_periods = [10, 50, 100]
        summary['return_levels'] = {
            f'{period}-year': self.predict_return_level(period)
            for period in return_periods
        }
        
        return summary 