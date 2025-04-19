"""
Tests for the GEV model.
"""
import numpy as np
import pytest
from bayesextremes.models import GEV

def test_gev_initialization():
    """Test GEV model initialization."""
    # Generate some test data
    np.random.seed(42)
    data = np.random.gumbel(loc=0, scale=1, size=100)
    
    # Initialize model
    model = GEV(data=data)
    
    # Check default parameters
    assert model.n_iterations == 10000
    assert model.burn_in == 1000
    assert model.xi == 0.0
    assert model.sigma == 1.0
    assert model.mu == 0.0
    
def test_gev_fit_and_predict():
    """Test GEV model fitting and prediction."""
    # Generate some test data
    np.random.seed(42)
    true_xi = 0.1
    true_sigma = 2.0
    true_mu = 1.0
    data = np.random.gumbel(loc=true_mu, scale=true_sigma, size=100)
    
    # Initialize and fit model with fewer iterations for testing
    model = GEV(
        data=data,
        n_iterations=1000,
        burn_in=100,
        shape_prior=(0, 1),
        scale_prior=(1, 1),
        loc_prior=(0, 1)
    )
    model.fit()
    
    # Test prediction
    x = np.linspace(-5, 10, 100)
    predictions = model.predict(x)
    
    # Basic checks
    assert len(predictions) == len(x)
    assert np.all(predictions >= 0)  # PDF should be non-negative
    assert np.all(np.isfinite(predictions))  # No infinities or NaNs
    
def test_gev_return_level():
    """Test GEV return level calculation."""
    # Generate some test data
    np.random.seed(42)
    data = np.random.gumbel(loc=0, scale=1, size=100)
    
    # Initialize model
    model = GEV(data=data)
    
    # Test return level calculation
    return_period = 100
    return_level = model.predict_return_level(return_period)
    
    # Basic checks
    assert np.isfinite(return_level)
    
    # Test invalid return period
    with pytest.raises(ValueError):
        model.predict_return_level(0.5)
        
def test_gev_summary():
    """Test GEV model summary statistics."""
    # Generate some test data
    np.random.seed(42)
    data = np.random.gumbel(loc=0, scale=1, size=100)
    
    # Initialize and fit model with fewer iterations for testing
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