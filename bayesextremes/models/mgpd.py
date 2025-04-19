"""
Mixed Gamma-Pareto Distribution model for extreme value analysis.

This module implements a semiparametric Bayesian approach to extreme value estimation
using a mixture of Gamma distributions for the bulk and a Generalized Pareto Distribution
for the tail.

References:
    do Nascimento, F.F., Gamerman, D. and Lopes, H.F., 2012. A semiparametric Bayesian
    approach to extreme value estimation. Statistics and Computing, 22(2), pp.661-675.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import gamma, truncnorm, norm, dirichlet, invgamma
from ..base import BaseModel

class MixedGammaParetoModel(BaseModel):
    """Mixed Gamma-Pareto Distribution model for extreme value analysis.
    
    This model combines a mixture of Gamma distributions for the bulk of the data
    with a Generalized Pareto Distribution (GPD) for the tail. The model is fit using
    MCMC methods.
    
    Attributes:
        n_iterations: Number of MCMC iterations
        data: Input data array
        k: Number of mixture components
        prior_values: Dictionary of prior parameters
        chains: Dictionary of MCMC chains for all parameters
    """
    
    def __init__(
        self,
        n_iterations: int,
        data: np.ndarray,
        k: int,
        prior_values: Dict[str, Union[float, np.ndarray]],
        burn_in: int = 5000,
        thin: int = 10,
        **kwargs
    ):
        """Initialize the Mixed Gamma-Pareto model.
        
        Args:
            n_iterations: Number of MCMC iterations
            data: Input data array
            k: Number of mixture components
            prior_values: Dictionary of prior parameters containing:
                - a_mu: Prior parameter vector for mu (IG distribution)
                - b_mu: Prior parameter vector for mu (IG distribution)
                - c_eta: Prior parameter vector for eta (Gamma distribution)
                - d_eta: Prior parameter vector for eta (Gamma distribution)
                - alpha_p: Prior parameter vector for mixture weights (Dirichlet)
                - mu_u: Prior mean for threshold u
                - sigma_u: Prior scale for threshold u
            burn_in: Number of burn-in iterations
            thin: Thinning factor for MCMC samples
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(n_iterations=n_iterations, data=data, **kwargs)
        self.k = k
        self.prior_values = prior_values
        self.burn_in = burn_in
        self.thin = thin
        self._validate_priors()
        self._initialize_parameters()
        self._setup_mcmc_parameters()
        self.chains = {}
        
    def _validate_priors(self) -> None:
        """Validate prior parameters."""
        required_priors = [
            'a_mu', 'b_mu', 'c_eta', 'd_eta',
            'alpha_p', 'mu_u', 'sigma_u'
        ]
        for prior in required_priors:
            if prior not in self.prior_values:
                raise ValueError(f"Missing required prior parameter: {prior}")
                
    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        # Initialize mixture parameters
        self.mus = np.linspace(
            np.percentile(self.data, 50/self.k),
            np.percentile(self.data, 50*(1 - 1/self.k)),
            self.k
        )
        self.etas = np.full(self.k, 5.0)
        self.ps = np.full(self.k, 1/self.k)
        
        # Initialize GPD parameters
        self.u = np.percentile(self.data, 90)
        self.sigma = np.std(self.data - self.u)
        self.xi = 0.1
        
    def _setup_mcmc_parameters(self) -> None:
        """Setup MCMC tuning parameters."""
        self.prop_vars = {
            'u': (0.1 * np.std(self.data))**2,
            'sigma': (0.1 * self.sigma)**2,
            'xi': 0.01,
            'mu': np.full(self.k, (0.1*np.mean(self.data))**2),
            'eta': np.full(self.k, 1.0),
            'p': np.ones(self.k)
        }
        
    def _gpd_density(self, x: np.ndarray, u: float, sigma: float, xi: float) -> np.ndarray:
        """Compute Generalized Pareto density for x > u."""
        z = (x - u) / sigma
        if xi != 0:
            return (1 / sigma) * (1 + xi * z) ** (-(1 + xi) / xi)
        else:
            return (1 / sigma) * np.exp(-z)
            
    def _gamma_mixture_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute mixture of Gamma densities."""
        pdf = np.zeros_like(x, dtype=float)
        for mu, eta, p in zip(self.mus, self.etas, self.ps):
            scale = mu / eta
            pdf += p * gamma.pdf(x, a=eta, scale=scale)
        return pdf
        
    def _gamma_mixture_cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute mixture of Gamma CDFs."""
        cdf = np.zeros_like(x, dtype=float)
        for mu, eta, p in zip(self.mus, self.etas, self.ps):
            scale = mu / eta
            cdf += p * gamma.cdf(x, a=eta, scale=scale)
        return cdf
        
    def _log_posterior(self) -> float:
        """Compute log-posterior for MGPDk model."""
        # Likelihood part
        idx_lower = self.data <= self.u
        idx_upper = self.data > self.u
        x_lower = self.data[idx_lower]
        x_upper = self.data[idx_upper]
        
        # Density below u
        pdf_lower = self._gamma_mixture_pdf(x_lower)
        
        # Tail mass above u
        H_u = self._gamma_mixture_cdf(self.u)
        pdf_upper = (1 - H_u) * np.array([
            self._gpd_density(xx, self.u, self.sigma, self.xi)
            for xx in x_upper
        ])
        
        if np.any(pdf_lower <= 0) or np.any(pdf_upper <= 0):
            return -np.inf
            
        ll = np.sum(np.log(pdf_lower)) + np.sum(np.log(pdf_upper))
        
        # Priors
        # mus: ordered inverse-Gamma IG(a_i/b_i, b_i)
        log_prior_mu = np.sum(invgamma.logpdf(
            self.mus,
            a=self.prior_values['a_mu'],
            scale=self.prior_values['b_mu']
        ))
        
        # Enforce ordering
        if not np.all(np.diff(self.mus) > 0):
            return -np.inf
            
        # etas: Gamma(c_j/d_j, c_j)
        log_prior_eta = np.sum(gamma.logpdf(
            self.etas,
            a=self.prior_values['c_eta'],
            scale=1/self.prior_values['d_eta']
        ))
        
        # ps: Dirichlet
        log_prior_p = dirichlet.logpdf(self.ps, self.prior_values['alpha_p'])
        
        # threshold u ~ N(mu_u, sigma_u^2)
        log_prior_u = norm.logpdf(
            self.u,
            loc=self.prior_values['mu_u'],
            scale=self.prior_values['sigma_u']
        )
        
        # GPD priors
        if self.sigma <= 0 or self.xi <= -0.5:
            return -np.inf
        log_prior_gpd = (
            -np.log(self.sigma) -
            np.log1p(self.xi) -
            0.5*np.log1p(2*self.xi)
        )
        
        return ll + log_prior_mu + log_prior_eta + log_prior_p + log_prior_u + log_prior_gpd
        
    def fit(self) -> None:
        """Fit the model using MCMC."""
        self.logger.info("Starting MCMC sampling...")
        
        # Initialize storage
        n_saved = (self.n_iterations - self.burn_in) // self.thin
        self.chains = {
            'mus': np.zeros((n_saved, self.k)),
            'etas': np.zeros((n_saved, self.k)),
            'ps': np.zeros((n_saved, self.k)),
            'u': np.zeros(n_saved),
            'sigma': np.zeros(n_saved),
            'xi': np.zeros(n_saved)
        }
        
        current_lp = self._log_posterior()
        save_idx = 0
        
        for it in range(self.n_iterations):
            # Update xi
            xi_prop = norm.rvs(
                loc=self.xi,
                scale=np.sqrt(self.prop_vars['xi'])
            )
            xi_old = self.xi
            self.xi = xi_prop
            prop_lp = self._log_posterior()
            
            if np.log(np.random.rand()) < prop_lp - current_lp:
                current_lp = prop_lp
            else:
                self.xi = xi_old
                
            # Update sigma
            sigma_prop = abs(norm.rvs(
                loc=self.sigma,
                scale=np.sqrt(self.prop_vars['sigma'])
            ))
            sigma_old = self.sigma
            self.sigma = sigma_prop
            prop_lp = self._log_posterior()
            
            if np.log(np.random.rand()) < prop_lp - current_lp:
                current_lp = prop_lp
            else:
                self.sigma = sigma_old
                
            # Update u
            u_prop = norm.rvs(
                loc=self.u,
                scale=np.sqrt(self.prop_vars['u'])
            )
            u_old = self.u
            self.u = u_prop
            prop_lp = self._log_posterior()
            
            if np.log(np.random.rand()) < prop_lp - current_lp:
                current_lp = prop_lp
            else:
                self.u = u_old
                
            # Update mixture components
            for j in range(self.k):
                # Update mu_j
                mu_prop = abs(norm.rvs(
                    loc=self.mus[j],
                    scale=np.sqrt(self.prop_vars['mu'][j])
                ))
                mu_old = self.mus[j]
                self.mus[j] = mu_prop
                prop_lp = self._log_posterior()
                
                if np.log(np.random.rand()) < prop_lp - current_lp:
                    current_lp = prop_lp
                else:
                    self.mus[j] = mu_old
                    
                # Update eta_j
                eta_prop = abs(norm.rvs(
                    loc=self.etas[j],
                    scale=np.sqrt(self.prop_vars['eta'][j])
                ))
                eta_old = self.etas[j]
                self.etas[j] = eta_prop
                prop_lp = self._log_posterior()
                
                if np.log(np.random.rand()) < prop_lp - current_lp:
                    current_lp = prop_lp
                else:
                    self.etas[j] = eta_old
                    
            # Update ps vector via Dirichlet proposal
            p_prop = dirichlet.rvs(self.ps * self.prop_vars['p'])[0]
            p_old = self.ps.copy()
            self.ps = p_prop
            prop_lp = self._log_posterior()
            
            if np.log(np.random.rand()) < prop_lp - current_lp:
                current_lp = prop_lp
            else:
                self.ps = p_old
                
            # Save samples
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                idx = save_idx
                self.chains['mus'][idx] = self.mus
                self.chains['etas'][idx] = self.etas
                self.chains['ps'][idx] = self.ps
                self.chains['u'][idx] = self.u
                self.chains['sigma'][idx] = self.sigma
                self.chains['xi'][idx] = self.xi
                save_idx += 1
                
            if it % 1000 == 0:
                self.logger.info(f"Iteration {it}/{self.n_iterations} completed")
                
        self.logger.info("MCMC sampling completed")
        
    def get_parameter_chains(self) -> Dict[str, np.ndarray]:
        """Get the MCMC chains for all parameters.
        
        Returns:
            Dictionary containing parameter chains
        """
        return self.chains
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for the parameter estimates.
        
        Returns:
            Dictionary containing summary statistics for each parameter
        """
        summary = {}
        
        for param, chain in self.chains.items():
            if param in ['mus', 'etas', 'ps']:
                # Handle vector parameters
                for j in range(chain.shape[1]):
                    param_name = f"{param}_{j}"
                    summary[param_name] = {
                        'mean': np.mean(chain[:, j]),
                        'std': np.std(chain[:, j]),
                        '2.5%': np.percentile(chain[:, j], 2.5),
                        '97.5%': np.percentile(chain[:, j], 97.5)
                    }
            else:
                # Handle scalar parameters
                summary[param] = {
                    'mean': np.mean(chain),
                    'std': np.std(chain),
                    '2.5%': np.percentile(chain, 2.5),
                    '97.5%': np.percentile(chain, 97.5)
                }
                
        return summary 