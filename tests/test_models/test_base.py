"""
Tests for the base model class.
"""
import numpy as np
import pytest
from bayesextremes.models.base import BaseModel

class TestBaseModel:
    """Test cases for the BaseModel class."""
    
    def test_initialization(self):
        """Test model initialization with valid inputs."""
        data = np.array([1, 2, 3, 4, 5])
        model = BaseModel(n_iterations=1000, data=data)
        assert model.n_iterations == 1000
        np.testing.assert_array_equal(model.data, data)
        
    def test_invalid_iterations(self):
        """Test initialization with invalid number of iterations."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="n_iterations must be positive"):
            BaseModel(n_iterations=0, data=data)
            
    def test_empty_data(self):
        """Test initialization with empty data."""
        with pytest.raises(ValueError, match="data cannot be empty"):
            BaseModel(n_iterations=1000, data=np.array([]))
            
    def test_non_finite_data(self):
        """Test initialization with non-finite data."""
        data = np.array([1, 2, np.inf, 4])
        with pytest.raises(ValueError, match="data contains non-finite values"):
            BaseModel(n_iterations=1000, data=data)
            
    def test_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = BaseModel(n_iterations=1000, data=np.array([1, 2, 3]))
        with pytest.raises(NotImplementedError):
            model.fit()
        with pytest.raises(NotImplementedError):
            model.predict(np.array([1, 2]))
        with pytest.raises(NotImplementedError):
            model.get_parameter_chains()
        with pytest.raises(NotImplementedError):
            model.get_summary()
            
    def test_metropolis_step(self):
        """Test the Metropolis-Hastings step."""
        model = BaseModel(n_iterations=1000, data=np.array([1, 2, 3]))
        
        def log_posterior(x):
            return -x**2
            
        def proposal_dist(x, y):
            return 0
            
        current = 1.0
        proposal = 2.0
        
        new_value, accepted = model._metropolis_step(
            current=current,
            proposal=proposal,
            log_posterior=log_posterior,
            proposal_dist=proposal_dist
        )
        
        assert isinstance(new_value, float)
        assert isinstance(accepted, bool) 