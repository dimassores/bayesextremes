"""
Base model class for all Bayesian extreme value models.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import logging

class BaseModel:
    """Base class for all Bayesian extreme value models."""
    
    def __init__(
        self,
        data: np.ndarray,
        n_iterations: int = 10000,
        burn_in: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the base model.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        n_iterations : int, optional
            Number of MCMC iterations, by default 10000
        burn_in : int, optional
            Number of burn-in iterations. If None, set to 10% of n_iterations
        **kwargs : dict
            Additional model-specific parameters
        """
        self.n_iterations = n_iterations
        self.burn_in = burn_in if burn_in is not None else int(0.1 * n_iterations)
        self.data = np.asarray(data)
        self._validate_inputs()
        self._setup_logging()
        self._initialize_traces()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.burn_in >= self.n_iterations:
            raise ValueError("burn_in must be less than n_iterations")
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
            
    def _initialize_traces(self) -> None:
        """Initialize trace arrays for MCMC parameters."""
        self.traces = {}
            
    def fit(self) -> None:
        """Fit the model to the data."""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def get_parameter_chains(self) -> Dict[str, np.ndarray]:
        """Get the MCMC chains for all parameters."""
        return {k: np.array(v) for k, v in self.traces.items()}
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model fit."""
        chains = self.get_parameter_chains()
        summary = {}
        for param, chain in chains.items():
            summary[param] = {
                'mean': np.mean(chain),
                'std': np.std(chain),
                '2.5%': np.percentile(chain, 2.5),
                '97.5%': np.percentile(chain, 97.5)
            }
        return summary
        
    def log_likelihood(self) -> float:
        """Compute the log-likelihood of the model."""
        raise NotImplementedError("Subclasses must implement log_likelihood()")
        
    def log_prior(self) -> float:
        """Compute the log-prior of the model."""
        raise NotImplementedError("Subclasses must implement log_prior()")
        
    def _metropolis_step(self, param: str) -> bool:
        """
        Perform a Metropolis-Hastings step for a given parameter.
        
        Parameters
        ----------
        param : str
            Parameter to update
            
        Returns
        -------
        bool
            Whether the proposal was accepted
        """
        current_value = getattr(self, param)
        current_log_posterior = self.log_likelihood() + self.log_prior()
        
        # Propose new value
        if param == 'weight' or param == 'p':
            proposal = np.random.beta(2, 2)  # Beta proposal for weights
        elif param in ['sigma', 'rate1', 'rate2']:
            proposal = np.abs(np.random.normal(current_value, 0.1))  # Positive parameters
        else:
            proposal = np.random.normal(current_value, 0.1)  # General case
            
        # Apply proposal
        setattr(self, param, proposal)
        proposal_log_posterior = self.log_likelihood() + self.log_prior()
        
        # Decide to accept/reject
        if np.log(np.random.random()) < proposal_log_posterior - current_log_posterior:
            return True
        else:
            setattr(self, param, current_value)
            return False
        
    def _update_trace(self, param: str, value: float, iteration: int) -> None:
        """
        Update the trace for a parameter.
        
        Parameters
        ----------
        param : str
            Parameter name
        value : float
            Parameter value
        iteration : int
            Current iteration number
        """
        if iteration >= self.burn_in:
            if param not in self.traces:
                self.traces[param] = []
            self.traces[param].append(value) 