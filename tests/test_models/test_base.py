"""
Tests for the BaseModel class.
"""

import numpy as np
import pytest
from bayesextremes.models.base import BaseModel

class TestBaseModel:
    """Test suite for the BaseModel class."""
    
    def test_initialization(self):
        """Test model initialization with valid data."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data, n_iterations=1000, burn_in=100)
        assert model.data.shape == (100,)
        assert model.n_iterations == 1000
        assert model.burn_in == 100
        
    def test_invalid_iterations(self):
        """Test initialization with invalid number of iterations."""
        data = np.random.normal(0, 1, 100)
        with pytest.raises(ValueError):
            BaseModel(data, n_iterations=0, burn_in=100)
        with pytest.raises(ValueError):
            BaseModel(data, n_iterations=1000, burn_in=2000)
            
    def test_empty_data(self):
        """Test initialization with empty data array."""
        with pytest.raises(ValueError):
            BaseModel(np.array([]), n_iterations=1000, burn_in=100)
            
    def test_non_finite_data(self):
        """Test initialization with non-finite values in data."""
        data = np.array([1, 2, np.inf, 3])
        with pytest.raises(ValueError):
            BaseModel(data, n_iterations=1000, burn_in=100)
            
    def test_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data, n_iterations=1000, burn_in=100)
        
        with pytest.raises(NotImplementedError):
            model.log_likelihood()
            
        with pytest.raises(NotImplementedError):
            model.log_prior()
            
        with pytest.raises(NotImplementedError):
            model.fit()
            
    def test_metropolis_step(self):
        """Test the Metropolis-Hastings step."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data, n_iterations=1000, burn_in=100)
        
        # Create a simple test class that implements the required methods
        class TestModel(BaseModel):
            def log_likelihood(self):
                return -0.5 * np.sum(self.data**2)
                
            def log_prior(self):
                return 0.0
                
            def fit(self):
                pass
                
        test_model = TestModel(data, n_iterations=1000, burn_in=100)
        current_value = 1.0
        proposal = 1.1
        
        # Test acceptance
        np.random.seed(42)
        result = test_model.metropolis_step(current_value, proposal)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], bool) 