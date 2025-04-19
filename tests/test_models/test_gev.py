"""
Tests for the GEV model.
"""

import numpy as np
import pytest
from scipy.stats import genextreme
from bayesextremes.models import GEV

class TestGEV:
    """Test suite for the GEV model."""
    
    def test_initialization(self):
        """Test GEV model initialization."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize model
        model = GEV(
            data=data,
            n_iterations=1000,
            burn_in=100,
            shape_prior=(0, 1),
            scale_prior=(1, 1),
            loc_prior=(0, 1)
        )
        
        # Check default parameters
        assert model.n_iterations == 1000
        assert model.burn_in == 100
        assert model.xi == 0.0
        assert model.sigma == 1.0
        assert model.mu == 0.0
        assert model.shape_prior == (0, 1)
        assert model.scale_prior == (1, 1)
        assert model.loc_prior == (0, 1)
        
    def test_log_likelihood(self):
        """Test GEV log-likelihood calculation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize model
        model = GEV(data=data)
        
        # Test log-likelihood calculation
        log_lik = model.log_likelihood()
        assert np.isfinite(log_lik)
        
        # Compare with scipy implementation
        scipy_log_lik = np.sum(genextreme.logpdf(
            data, c=model.xi, loc=model.mu, scale=model.sigma
        ))
        assert np.isclose(log_lik, scipy_log_lik)
        
    def test_log_prior(self):
        """Test GEV log-prior calculation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize model
        model = GEV(
            data=data,
            shape_prior=(0, 1),
            scale_prior=(1, 1),
            loc_prior=(0, 1)
        )
        
        # Test log-prior calculation
        log_prior = model.log_prior()
        assert np.isfinite(log_prior)
        
        # Test with invalid parameters
        model.sigma = -1.0
        assert model.log_prior() == -np.inf
        
    def test_fit(self):
        """Test GEV model fitting."""
        # Generate some test data
        np.random.seed(42)
        true_xi = 0.1
        true_sigma = 2.0
        true_mu = 1.0
        data = genextreme.rvs(
            c=true_xi,
            loc=true_mu,
            scale=true_sigma,
            size=100
        )
        
        # Initialize and fit model
        model = GEV(
            data=data,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Check that traces were updated
        assert 'xi' in model.traces
        assert 'sigma' in model.traces
        assert 'mu' in model.traces
        
        # Check trace lengths
        expected_length = (model.n_iterations - model.burn_in)
        assert len(model.traces['xi']) == expected_length
        assert len(model.traces['sigma']) == expected_length
        assert len(model.traces['mu']) == expected_length
        
    def test_predict(self):
        """Test GEV model prediction."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize and fit model
        model = GEV(
            data=data,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Test prediction
        x = np.linspace(-5, 10, 100)
        predictions = model.predict(x)
        
        # Basic checks
        assert len(predictions) == len(x)
        assert np.all(predictions >= 0)  # PDF should be non-negative
        assert np.all(np.isfinite(predictions))  # No infinities or NaNs
        
        # Compare with scipy implementation
        scipy_predictions = genextreme.pdf(
            x, c=model.xi, loc=model.mu, scale=model.sigma
        )
        assert np.allclose(predictions, scipy_predictions)
        
    def test_predict_return_level(self):
        """Test GEV return level calculation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize and fit model
        model = GEV(
            data=data,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Test return level calculation
        return_period = 100
        return_level = model.predict_return_level(return_period)
        
        # Basic checks
        assert np.isfinite(return_level)
        
        # Test invalid return period
        with pytest.raises(ValueError):
            model.predict_return_level(0.5)
            
    def test_get_summary(self):
        """Test GEV model summary statistics."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gumbel(loc=0, scale=1, size=100)
        
        # Initialize and fit model
        model = GEV(
            data=data,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Get summary
        summary = model.get_summary()
        
        # Check summary contents
        assert 'xi' in summary
        assert 'sigma' in summary
        assert 'mu' in summary
        assert 'return_levels' in summary
        
        # Check summary statistics
        for param in ['xi', 'sigma', 'mu']:
            assert 'mean' in summary[param]
            assert 'std' in summary[param]
            assert '2.5%' in summary[param]
            assert '97.5%' in summary[param]
            
        # Check return levels
        assert '10-year' in summary['return_levels']
        assert '50-year' in summary['return_levels']
        assert '100-year' in summary['return_levels'] 