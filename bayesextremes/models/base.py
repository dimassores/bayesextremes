"""
Base model class for all Bayesian extreme value models.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import logging

class BaseModel:
    """Base class for all Bayesian extreme value models."""
    
    def __init__(self, n_iterations: int, data: np.ndarray, **kwargs):
        """
        Initialize the base model.
        
        Args:
            n_iterations: Number of MCMC iterations
            data: Input data array
            **kwargs: Additional model-specific parameters
        """
        self.n_iterations = n_iterations
        self.data = np.asarray(data)
        self._validate_inputs()
        self._setup_logging()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if len(self.data) == 0:
            raise ValueError("data cannot be empty")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("data contains non-finite values")
            
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def fit(self) -> None:
        """Fit the model to the data."""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def get_parameter_chains(self) -> Dict[str, np.ndarray]:
        """Get the MCMC chains for all parameters."""
        raise NotImplementedError("Subclasses must implement get_parameter_chains()")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model fit."""
        raise NotImplementedError("Subclasses must implement get_summary()")
        
    def _metropolis_step(
        self,
        current: float,
        proposal: float,
        log_posterior: callable,
        proposal_dist: callable,
        **kwargs
    ) -> Tuple[float, bool]:
        """
        Perform a Metropolis-Hastings step.
        
        Args:
            current: Current parameter value
            proposal: Proposed parameter value
            log_posterior: Function to compute log posterior
            proposal_dist: Function to compute proposal distribution
            **kwargs: Additional arguments for log_posterior
            
        Returns:
            Tuple of (new value, whether accepted)
        """
        log_ratio = (
            log_posterior(proposal, **kwargs) - log_posterior(current, **kwargs) +
            proposal_dist(current, proposal) - proposal_dist(proposal, current)
        )
        prob = min(1, np.exp(log_ratio))
        accepted = np.random.random() < prob
        return (proposal if accepted else current, accepted) 