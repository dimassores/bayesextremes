import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from scipy.stats import gamma, truncnorm, norm, dirichlet, invgamma

class MixedGammaParetoModel:
    def __init__(
        self,
        data: np.ndarray,
        k: int,
        prior_values: Dict[str, np.ndarray],
        n_iterations: int = 1000,
        burn_in: int = 100,
        thin: int = 10
    ):
        """
        Initialize the Mixed Gamma-Pareto Distribution model.

        Parameters
        ----------
        data : np.ndarray
            Input data
        k : int
            Number of mixture components
        prior_values : Dict[str, np.ndarray]
            Dictionary of prior parameters
        n_iterations : int, optional
            Number of MCMC iterations, by default 1000
        burn_in : int, optional
            Number of burn-in iterations, by default 100
        thin : int, optional
            Thinning factor, by default 10
        """
        self.data = data
        self.k = k
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        
        # Initialize parameters
        self.mus = np.sort(np.random.gamma(shape=prior_values['a_mu'], scale=1/prior_values['b_mu']))
        self.etas = np.random.gamma(shape=prior_values['c_eta'], scale=1/prior_values['d_eta'])
        self.ps = np.random.dirichlet(prior_values['alpha_p'])
        self.u = np.random.normal(prior_values['mu_u'], prior_values['sigma_u'])
        self.sigma = np.random.gamma(shape=2, scale=1)
        self.xi = np.random.normal(0, 0.1)
        
        # Initialize traces
        self.traces = {
            'mus': [],
            'etas': [],
            'ps': [],
            'u': [],
            'sigma': [],
            'xi': []
        }
        
        # Store prior values
        self.prior_values = prior_values

    def log_likelihood(self) -> float:
        """Calculate the log-likelihood of the model."""
        # Calculate likelihood for each component
        component_likelihoods = []
        for i in range(self.k):
            # Gamma component
            gamma_lik = gamma.pdf(self.data, a=self.etas[i], scale=1/self.mus[i])
            # Pareto component (for values above threshold)
            term = 1 + self.xi * (self.data - self.u) / self.sigma
            pareto_lik = np.where(
                (self.data > self.u) & (term > 0) & (self.sigma > 0),
                np.exp((-1/self.xi - 1) * np.log(np.maximum(term, 1e-10)) - np.log(np.maximum(self.sigma, 1e-10))),
                0
            )
            # Combine components
            component_likelihoods.append(self.ps[i] * (gamma_lik + pareto_lik))
        
        # Sum over components and take log
        total_likelihood = np.sum(component_likelihoods, axis=0)
        return np.sum(np.log(total_likelihood + 1e-10))

    def log_prior(self) -> float:
        """Calculate the log-prior of the model."""
        # Check parameter constraints
        if self.sigma <= 0 or np.any(self.ps <= 0) or np.any(self.mus <= 0) or np.any(self.etas <= 0):
            return -np.inf
        
        # Check ordering of mus
        if not np.all(np.diff(self.mus) > 0):
            return -np.inf
        
        # Calculate prior for each parameter
        log_prior = 0
        
        # Prior for mus
        for i in range(self.k):
            log_prior += gamma.logpdf(self.mus[i], a=self.prior_values['a_mu'][i], scale=1/self.prior_values['b_mu'][i])
        
        # Prior for etas
        for i in range(self.k):
            log_prior += gamma.logpdf(self.etas[i], a=self.prior_values['c_eta'][i], scale=1/self.prior_values['d_eta'][i])
        
        # Prior for ps
        log_prior += dirichlet.logpdf(self.ps, alpha=self.prior_values['alpha_p'])
        
        # Prior for u
        log_prior += norm.logpdf(self.u, loc=self.prior_values['mu_u'], scale=self.prior_values['sigma_u'])
        
        # Prior for sigma
        log_prior += gamma.logpdf(self.sigma, a=2, scale=1)
        
        # Prior for xi
        log_prior += norm.logpdf(self.xi, loc=0, scale=0.1)
        
        return log_prior

    def fit(self):
        """Fit the model using MCMC."""
        for iteration in range(self.n_iterations):
            # Update parameters
            for param in ['mus', 'etas', 'ps', 'u', 'sigma', 'xi']:
                if param in ['mus', 'etas']:
                    for i in range(self.k):
                        self._metropolis_step(f"{param}[{i}]")
                else:
                    self._metropolis_step(param)
            
            # Store trace
            if iteration >= self.burn_in and (iteration - self.burn_in) % self.thin == 0:
                self.traces['mus'].append(self.mus.copy())
                self.traces['etas'].append(self.etas.copy())
                self.traces['ps'].append(self.ps.copy())
                self.traces['u'].append(self.u)
                self.traces['sigma'].append(self.sigma)
                self.traces['xi'].append(self.xi)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the PDF at given points."""
        # Calculate PDF for each component
        component_pdfs = []
        for i in range(self.k):
            # Gamma component
            gamma_pdf = gamma.pdf(x, a=self.etas[i], scale=1/self.mus[i])
            # Pareto component (for values above threshold)
            term = 1 + self.xi * (x - self.u) / self.sigma
            pareto_pdf = np.where(
                (x > self.u) & (term > 0) & (self.sigma > 0),
                np.exp((-1/self.xi - 1) * np.log(np.maximum(term, 1e-10)) - np.log(np.maximum(self.sigma, 1e-10))),
                0
            )
            # Combine components
            component_pdfs.append(self.ps[i] * (gamma_pdf + pareto_pdf))
        
        # Sum over components
        return np.sum(component_pdfs, axis=0)

    def get_summary(self) -> Dict:
        """Get summary statistics of the model."""
        summary = {}
        
        # Convert traces to numpy arrays
        traces = {param: np.array(self.traces[param]) for param in self.traces}
        
        # Parameter summaries
        for param in ['mus', 'etas', 'ps', 'u', 'sigma', 'xi']:
            summary[param] = {
                'mean': np.mean(traces[param], axis=0),
                'std': np.std(traces[param], axis=0),
                '2.5%': np.percentile(traces[param], 2.5, axis=0),
                '97.5%': np.percentile(traces[param], 97.5, axis=0)
            }
        
        # Tail characteristics
        summary['tail_characteristics'] = {
            'threshold': np.mean(traces['u']),
            'tail_index': np.mean(traces['xi']),
            'scale': np.mean(traces['sigma']),
            'tail_probability': np.mean([np.mean(x > traces['u'][i]) for i, x in enumerate(traces['mus'])])
        }
        
        # Mixture components
        summary['mixture_components'] = {}
        for i in range(self.k):
            summary['mixture_components'][f'component_{i+1}'] = {
                'mean': np.mean(traces['mus'][:, i]),
                'shape': np.mean(traces['etas'][:, i]),
                'weight': np.mean(traces['ps'][:, i])
            }
        
        return summary

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
        # Handle array parameters
        if param.startswith(('mus[', 'etas[')):
            param_name = param.split('[')[0]  # 'mus' or 'etas'
            idx = int(param.split('[')[1].rstrip(']'))  # Get index between [ and ]
            array_param = getattr(self, param_name)  # get 'mus' or 'etas' array
            current_value = array_param[idx]
        else:
            current_value = getattr(self, param)

        current_log_posterior = self.log_likelihood() + self.log_prior()

        # Propose new value
        if param == 'ps':
            # Ensure positivity and sum-to-one before proposing
            ps_safe = np.clip(self.ps, 1e-6, None)
            ps_safe /= ps_safe.sum()
            proposal = np.random.dirichlet(ps_safe * 100)
        elif param.startswith('mus['):
            proposal = np.abs(np.random.normal(current_value, 0.1))
            # Ensure ordering of mus
            if idx > 0:
                proposal = max(proposal, self.mus[idx-1] + 1e-6)
            if idx < self.k - 1:
                proposal = min(proposal, self.mus[idx+1] - 1e-6)
        elif param.startswith('etas['):
            proposal = np.abs(np.random.normal(current_value, 0.1))  # Ensure positive
        elif param in ['sigma', 'u']:
            proposal = np.abs(np.random.normal(current_value, 0.1))
        else:
            proposal = np.random.normal(current_value, 0.1)

        # Apply proposal
        if param.startswith(('mus[', 'etas[')):
            array_param = getattr(self, param_name)
            array_param[idx] = proposal
        else:
            setattr(self, param, proposal)

        proposal_log_posterior = self.log_likelihood() + self.log_prior()

        # Decide to accept/reject
        if np.log(np.random.random()) < proposal_log_posterior - current_log_posterior:
            return True
        else:
            if param.startswith(('mus[', 'etas[')):
                array_param = getattr(self, param_name)
                array_param[idx] = current_value
            else:
                setattr(self, param, current_value)
            return False
