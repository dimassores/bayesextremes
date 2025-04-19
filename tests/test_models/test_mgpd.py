"""
Tests for the MGPD model.
"""

import numpy as np
import pytest
from scipy.stats import gamma, genpareto
from bayesextremes.models import MGPD

class TestMGPD:
    """Test suite for the MGPD model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data from a mixture of Gamma and GPD distributions."""
        np.random.seed(42)
        
        # Generate bulk data from mixture of two Gamma distributions
        n_bulk = 900
        bulk_data = np.concatenate([
            gamma.rvs(a=2, scale=1, size=n_bulk//2),
            gamma.rvs(a=5, scale=2, size=n_bulk//2)
        ])
        
        # Generate tail data from GPD
        n_tail = 100
        tail_data = genpareto.rvs(c=0.5, loc=0, scale=1, size=n_tail)
        
        return np.concatenate([bulk_data, tail_data])
    
    def test_initialization(self, sample_data):
        """Test model initialization with valid data."""
        model = MGPD(sample_data)
        assert model.data.shape == (1000,)
        assert model.n_iterations == 10000
        assert model.burn_in == 1000
        
    def test_log_likelihood(self, sample_data):
        """Test log-likelihood computation."""
        model = MGPD(sample_data)
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        assert not np.isinf(log_lik)
        
    def test_log_prior(self, sample_data):
        """Test log-prior computation."""
        model = MGPD(sample_data)
        log_prior = model.log_prior()
        assert isinstance(log_prior, float)
        assert not np.isnan(log_prior)
        assert not np.isinf(log_prior)
        
    def test_metropolis_step(self, sample_data):
        """Test Metropolis-Hastings step."""
        model = MGPD(sample_data)
        for param in ['xi', 'sigma', 'u']:
            accepted = model.metropolis_step(param)
            assert isinstance(accepted, bool)
            
    def test_fit(self, sample_data):
        """Test model fitting."""
        model = MGPD(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        # Check that traces are populated
        assert len(model.xi_trace) > 0
        assert len(model.sigma_trace) > 0
        assert len(model.u_trace) > 0
        
        # Check that traces have the correct length
        expected_length = model.n_iterations - model.burn_in
        assert len(model.xi_trace) == expected_length
        assert len(model.sigma_trace) == expected_length
        assert len(model.u_trace) == expected_length
        
    def test_predict_return_level(self, sample_data):
        """Test return level prediction."""
        model = MGPD(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        return_periods = [10, 50, 100]
        for period in return_periods:
            level = model.predict_return_level(period)
            assert isinstance(level, float)
            assert not np.isnan(level)
            assert not np.isinf(level)
            
    def test_parameter_estimates(self, sample_data):
        """Test parameter estimates property."""
        model = MGPD(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        estimates = model.parameter_estimates
        assert isinstance(estimates, dict)
        assert 'xi' in estimates
        assert 'sigma' in estimates
        assert 'u' in estimates
        
        for param in estimates.values():
            assert isinstance(param, float)
            assert not np.isnan(param)
            assert not np.isinf(param)
            
    def test_parameter_credible_intervals(self, sample_data):
        """Test parameter credible intervals property."""
        model = MGPD(sample_data, n_iterations=100, burn_in=10)
        model.fit()
        
        intervals = model.parameter_credible_intervals
        assert isinstance(intervals, dict)
        assert 'xi' in intervals
        assert 'sigma' in intervals
        assert 'u' in intervals
        
        for param in intervals.values():
            assert isinstance(param, np.ndarray)
            assert param.shape == (2,)
            assert not np.any(np.isnan(param))
            assert not np.any(np.isinf(param))
            assert param[0] <= param[1]  # Lower bound <= upper bound
            
    def test_threshold_selection(self, sample_data):
        """Test threshold selection method."""
        model = MGPD(sample_data)
        thresholds = np.linspace(np.quantile(sample_data, 0.5), 
                               np.quantile(sample_data, 0.95), 10)
        selected_threshold = model.select_threshold(thresholds)
        
        assert isinstance(selected_threshold, float)
        assert not np.isnan(selected_threshold)
        assert not np.isinf(selected_threshold)
        assert selected_threshold >= min(thresholds)
        assert selected_threshold <= max(thresholds) 