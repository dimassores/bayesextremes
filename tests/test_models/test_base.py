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
        model = BaseModel(data=data, n_iterations=1000, burn_in=100)
        assert model.data.shape == (100,)
        assert model.n_iterations == 1000
        assert model.burn_in == 100
        assert isinstance(model.traces, dict)
        
    def test_invalid_iterations(self):
        """Test initialization with invalid number of iterations."""
        data = np.random.normal(0, 1, 100)
        with pytest.raises(ValueError):
            BaseModel(data=data, n_iterations=0)
        with pytest.raises(ValueError):
            BaseModel(data=data, n_iterations=1000, burn_in=1000)
        with pytest.raises(ValueError):
            BaseModel(data=data, n_iterations=1000, burn_in=-1)
            
    def test_empty_data(self):
        """Test initialization with empty data array."""
        with pytest.raises(ValueError):
            BaseModel(data=np.array([]), n_iterations=1000)
            
    def test_non_finite_data(self):
        """Test initialization with non-finite values in data."""
        data = np.array([1, 2, np.inf, 3])
        with pytest.raises(ValueError):
            BaseModel(data=data, n_iterations=1000)
            
    def test_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data=data, n_iterations=1000)
        
        with pytest.raises(NotImplementedError):
            model.predict(np.array([1.0]))
            
        with pytest.raises(NotImplementedError):
            model.log_likelihood()
            
        with pytest.raises(NotImplementedError):
            model.log_prior()
            
        with pytest.raises(NotImplementedError):
            model.fit()
            
    def test_metropolis_step(self):
        """Test the Metropolis-Hastings step."""
        data = np.random.normal(0, 1, 100)
        
        # Create a simple test class that implements the required methods
        class TestModel(BaseModel):
            def log_likelihood(self) -> float:
                return -0.5 * np.sum(self.data**2)  # Simple Gaussian likelihood
                
            def log_prior(self) -> float:
                return 0.0  # Flat prior
                
            def fit(self):
                pass
                
            def predict(self, x):
                return np.zeros_like(x)
                
        test_model = TestModel(data=data, n_iterations=1000)
        test_model.test_param = 1.0
        
        # Test acceptance
        np.random.seed(42)
        accepted = test_model._metropolis_step('test_param')
        assert isinstance(accepted, bool)
        
    def test_update_trace(self):
        """Test the trace update functionality."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data=data, n_iterations=1000, burn_in=100)
        
        # Test updating trace before burn-in
        model._update_trace('test_param', 1.0, 50)
        assert 'test_param' not in model.traces
        
        # Test updating trace after burn-in
        model._update_trace('test_param', 1.0, 150)
        assert 'test_param' in model.traces
        assert len(model.traces['test_param']) == 1
        assert model.traces['test_param'][0] == 1.0
        
    def test_get_summary(self):
        """Test the summary statistics calculation."""
        data = np.random.normal(0, 1, 100)
        model = BaseModel(data=data, n_iterations=1000, burn_in=100)
        
        # Add some test traces
        model.traces = {
            'param1': [1.0, 2.0, 3.0],
            'param2': [4.0, 5.0, 6.0]
        }
        
        summary = model.get_summary()
        assert 'param1' in summary
        assert 'param2' in summary
        
        for param in ['param1', 'param2']:
            assert 'mean' in summary[param]
            assert 'std' in summary[param]
            assert '2.5%' in summary[param]
            assert '97.5%' in summary[param] 