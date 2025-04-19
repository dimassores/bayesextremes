"""
Tests for the Mixed Gamma-Pareto Distribution model.
"""
import numpy as np
import pytest
from scipy.stats import gamma, truncnorm, norm, dirichlet, invgamma
from bayesextremes.models import MixedGammaParetoModel

class TestMixedGammaParetoModel:
    """Test suite for the Mixed Gamma-Pareto Distribution model."""
    
    def test_initialization(self):
        """Test MGPD model initialization."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values,
            n_iterations=1000,
            burn_in=100,
            thin=10
        )
        
        # Check parameters
        assert model.n_iterations == 1000
        assert model.burn_in == 100
        assert model.thin == 10
        assert model.k == 2
        assert len(model.mus) == 2
        assert len(model.etas) == 2
        assert len(model.ps) == 2
        assert np.all(np.diff(model.mus) > 0)  # mus should be ordered
        
    def test_log_likelihood(self):
        """Test MGPD log-likelihood calculation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values
        )
        
        # Test log-likelihood calculation
        log_lik = model.log_likelihood()
        assert np.isfinite(log_lik)
        
    def test_log_prior(self):
        """Test MGPD log-prior calculation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values
        )
        
        # Test log-prior calculation
        log_prior = model.log_prior()
        assert np.isfinite(log_prior)
        
        # Test with invalid parameters
        model.sigma = -1.0
        assert model.log_prior() == -np.inf
        
        # Test with unordered mus
        model.sigma = 1.0
        model.mus = np.array([2.0, 1.0])
        assert model.log_prior() == -np.inf
        
    def test_fit(self):
        """Test MGPD model fitting."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values,
            n_iterations=1000,
            burn_in=100,
            thin=10
        )
        
        # Fit model
        model.fit()
        
        # Check that traces were updated
        assert 'mus' in model.traces
        assert 'etas' in model.traces
        assert 'ps' in model.traces
        assert 'u' in model.traces
        assert 'sigma' in model.traces
        assert 'xi' in model.traces
        
        # Check trace lengths
        expected_length = (model.n_iterations - model.burn_in) // model.thin
        assert len(model.traces['mus']) == expected_length
        assert len(model.traces['etas']) == expected_length
        assert len(model.traces['ps']) == expected_length
        assert len(model.traces['u']) == expected_length
        assert len(model.traces['sigma']) == expected_length
        assert len(model.traces['xi']) == expected_length
        
    def test_predict(self):
        """Test MGPD model prediction."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Test prediction
        x = np.linspace(0, 10, 100)
        predictions = model.predict(x)
        
        # Basic checks
        assert len(predictions) == len(x)
        assert np.all(predictions >= 0)  # PDF should be non-negative
        assert np.all(np.isfinite(predictions))  # No infinities or NaNs
        
    def test_get_summary(self):
        """Test MGPD model summary statistics."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=1, size=100)
        
        # Initialize model
        prior_values = {
            'a_mu': np.array([2.0, 3.0]),
            'b_mu': np.array([1.0, 1.0]),
            'c_eta': np.array([2.0, 2.0]),
            'd_eta': np.array([1.0, 1.0]),
            'alpha_p': np.array([1.0, 1.0]),
            'mu_u': np.percentile(data, 90),
            'sigma_u': 1.0
        }
        
        model = MixedGammaParetoModel(
            data=data,
            k=2,
            prior_values=prior_values,
            n_iterations=1000,
            burn_in=100
        )
        model.fit()
        
        # Get summary
        summary = model.get_summary()
        
        # Check summary contents
        assert 'mus' in summary
        assert 'etas' in summary
        assert 'ps' in summary
        assert 'u' in summary
        assert 'sigma' in summary
        assert 'xi' in summary
        assert 'tail_characteristics' in summary
        assert 'mixture_components' in summary
        
        # Check summary statistics
        for param in ['mus', 'etas', 'ps', 'u', 'sigma', 'xi']:
            assert 'mean' in summary[param]
            assert 'std' in summary[param]
            assert '2.5%' in summary[param]
            assert '97.5%' in summary[param]
            
        # Check tail characteristics
        assert 'threshold' in summary['tail_characteristics']
        assert 'tail_index' in summary['tail_characteristics']
        assert 'scale' in summary['tail_characteristics']
        assert 'tail_probability' in summary['tail_characteristics']
        
        # Check mixture components
        assert len(summary['mixture_components']) == 2
        for i in range(2):
            component = summary['mixture_components'][f'component_{i+1}']
            assert 'mean' in component
            assert 'shape' in component
            assert 'weight' in component 