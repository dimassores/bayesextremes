"""
Tests for the GEV model.
"""

import numpy as np
import pytest
from scipy.stats import genextreme
from bayesextremes.models import GEV

class TestGEV:
    """Test suite for the GEV model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data from a GEV distribution."""
        np.random.seed(42)
        return genextreme.rvs(c=0.5, loc=0, scale=1, size=100)
    
    def test_initialization(self, sample_data):
        """Test model initialization with valid data."""
        model = GEV(sample_data)
        assert model.data.shape == (100,)
        assert model.n_iterations == 10000
        assert model.burn_in == 1000
        
    def test_log_likelihood(self, sample_data):
        """Test log-likelihood computation."""
        model = GEV(sample_data)
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        assert not np.isinf(log_lik)
        
    def test_log_prior(self, sample_data):
        """Test log-prior computation."""
        model = GEV(sample_data)
        log_prior = model.log_prior()
        assert isinstance(log_prior, float)
        assert not np.isnan(log_prior)
        assert not np.isinf(log_prior)
        
    def test_metropolis_step(self, sample_data):
        """Test Metropolis-Hastings step."""
        model = GEV(sample_data)
        for param in ['xi', 'sigma', 'mu']:
            accepted = model.metropolis_step(param)
            assert isinstance(accepted, bool)
            
    def test_fit(self, sample_data):
        """Test model fitting."""
        model = GEV(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        # Check that traces are populated
        assert len(model.xi_trace) > 0
        assert len(model.sigma_trace) > 0
        assert len(model.mu_trace) > 0
        
        # Check that traces have the correct length
        expected_length = model.n_iterations - model.burn_in
        assert len(model.xi_trace) == expected_length
        assert len(model.sigma_trace) == expected_length
        assert len(model.mu_trace) == expected_length
        
    def test_predict_return_level(self, sample_data):
        """Test return level prediction."""
        model = GEV(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        return_periods = [10, 50, 100]
        for period in return_periods:
            level = model.predict_return_level(period)
            assert isinstance(level, float)
            assert not np.isnan(level)
            assert not np.isinf(level)
            
    def test_parameter_estimates(self, sample_data):
        """Test parameter estimates property."""
        model = GEV(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        estimates = model.parameter_estimates
        assert isinstance(estimates, dict)
        assert 'xi' in estimates
        assert 'sigma' in estimates
        assert 'mu' in estimates
        
        for param in estimates.values():
            assert isinstance(param, float)
            assert not np.isnan(param)
            assert not np.isinf(param)
            
    def test_parameter_credible_intervals(self, sample_data):
        """Test parameter credible intervals property."""
        model = GEV(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        intervals = model.parameter_credible_intervals
        assert isinstance(intervals, dict)
        assert 'xi' in intervals
        assert 'sigma' in intervals
        assert 'mu' in intervals
        
        for param in intervals.values():
            assert isinstance(param, np.ndarray)
            assert param.shape == (2,)
            assert not np.any(np.isnan(param))
            assert not np.any(np.isinf(param))
            assert param[0] <= param[1]  # Lower bound <= upper bound 