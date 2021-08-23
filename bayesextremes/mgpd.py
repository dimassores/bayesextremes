import numpy as np
from scipy.stats import gamma, truncnorm, norm

class MGPD:

    '''
    mixed gamma generalized pareto distribution
    '''

    def __init__(self, n_iteration, data, k, prior_values):

        # input parameters
        self.n_iteration = n_iteration
        self.data = data
        self.k = k
        self.prior_values = prior_values

        # set initial values for the parameters
        self.p_array = np.array([[(1/self.k)]*self.k], dtype = np.float32)
        self.mu_array = np.array([[0]*self.k], dtype = np.float32)
        self.eta_array = np.array([[0]*self.k], dtype = np.float32)
        self.u_array = np.array([[1]], dtype = np.float32)
        self.csi_array = np.array([[1]], dtype = np.float32)
        self.sigma_array = np.array([[1]], dtype = np.float32)
        


    def log_mpgd_posterior_kernel(self, data, p, mu, eta, u, csi, sigma, a_prior, b_prior, c_prior, d_prior, mu_u, sigma_u):

        '''
        data: observed values
        p: mixture weights
        mu: mean vector from gamma mixture
        eta: shape vector from gamma mixture
        u: threshold value from extreme values represented as the GPD parameter
        csi: GPD parameter
        sigma: GPD parameter
        a_prior: prior parameter vector from IG distribution over mu
        b_prior: prior parameter vector from IG distribution over mu
        c_prior: prior parameter vector from gamma distribution over eta
        d_prior: prior parameter vector from gamma distribution over eta
        mu_u: prior mean over u
        sigma_u: prior scale over u
        '''

        data_under_u = self.data[data < u]
        data_over_u = self.data[data >= u]

        # gamma mixture kernel
        gm_kernel = 0
        for j in range(len(p)):
            gm_kernel += np.log(p[j] * gamma.pdf(x = data_under_u, a = eta[j], scale = eta[j] / mu[j])).sum()
            
            # cumulated prob for the mixture model
            cumulated_prob = p[j] * gamma.cdf(u, a = eta[j], scale = eta[j] / mu[j])
            
        # generalized pareto distribution kernel 
        log_cumulated_prob = len(data_over_u) * np.log(1 - cumulated_prob)

        gpd_kernel = (log_cumulated_prob
                    - np.sum(np.log(sigma) - ((1 + csi)/csi) * np.log1p(csi * (data_over_u - u)/sigma)
                    + np.sum((c_prior - 1) * np.log(eta) -  d_prior * eta  - (a_prior + 1) * np.log(mu) - b_prior/mu)
                    - 0.5*((u - mu_u)/sigma_u)**2 - np.log(sigma) - np.log(1+csi)
                    - 0.5 * np.log(1 + 2*csi)))
                        
        return gm_kernel + gpd_kernel
    
    #TODO: refact of metropolis step to be able to compute kernels
    def metropolis_step(self, prop, current, posterior_kernel, proposed_kernel):
        '''
        log scaled only for both posterior kernel and proposed kernel
        
        prop: proposed value
        current: current value
        posterior_kernel: dict with posterior kernel both under prop and current value
        proposed_kernel: dict proposed kernel with both under prop and current value
        '''
        ratio = (posterior_kernel['prop'] + proposed_kernel['current'])/(posterior_kernel['current'] + proposed_kernel['prop'])
        prob = min(1, ratio)
        
        rand = np.random.random_sample(1)
        if rand < prob:
            return prop
        else:
            return current

    def draw_csi(self, s):
            # latent values used in metropolis hastings steps
            self.M = max(self.data)
            self.v_csi = np.sqrt(1)

            # set auxiliar parameter
            inf_limit = -self.sigma_array[s]*(self.M - self.u_array[s])
            sup_limit = np.inf
            csi_posterior_kernel = {}
            csi_proposed_kernel = {}

            # draw potencial s+1
            csi_potential = truncnorm.rvs(a = inf_limit, b = sup_limit, loc=self.csi_array[s], scale=self.v_csi, size=1)

            # define metropolis ratio terms
            csi_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                        self.csi_array[s], self.sigma_array[s], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
            csi_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                        csi_potential, self.sigma_array[s], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
            csi_proposed_kernel['current'] = norm.logpdf(x = self.csi_array[s] - inf_limit/np.sqrt(self.v_csi))
            csi_proposed_kernel['prop'] = norm.logpdf(x = csi_potential - inf_limit/np.sqrt(self.v_csi))
            
            # compute metropolis ratio
            csi_metropolis_output = self.metropolis_step(csi_potential, self.csi_array[s], csi_posterior_kernel, csi_proposed_kernel)

            return csi_metropolis_output
            
    def draw_sigma(self, s):

        self.v_sigma = np.sqrt(1)
        inf_limit = -self.csi_array[s+1]*(self.M - self.u_array[s])
        sup_limit = np.inf
        sigma_posterior_kernel = {}
        sigma_proposed_kernel = {}
        
        if self.csi_array[s+1] > 0:

            sigma_potential = gamma.rvs(self.sigma_array[s], ((self.sigma_array[s])**2)/self.v_sigma)
            sigma_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                    self.csi_array[s+1], self.sigma_array[s], self.prior_values['a_prior'], 
                                                                    self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                    self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                    self.prior_values['sigma_u'])
        
            sigma_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                    self.csi_array[s+1], sigma_potential, self.prior_values['a_prior'], 
                                                                    self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                    self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                    self.prior_values['sigma_u'])
            
            sigma_proposed_kernel['current'] = gamma.logpdf(self.sigma_array[s], sigma_potential, (sigma_potential**2)/self.v_sigma)
            sigma_proposed_kernel['prop'] = gamma.logpdf(self.sigma_array[s], self.sigma_array[s], (self.sigma_array[s]**2)/self.v_sigma)
            
            # compute metropolis ratio
            sigma_metropolis_output = self.metropolis_step(sigma_potential, self.sigma_array[s], sigma_posterior_kernel, sigma_proposed_kernel)

            return sigma_metropolis_output
        else:
            sigma_potential = truncnorm.rvs(a = inf_limit , b = sup_limit, loc = self.sigma_array[s], scale = self.v_sigma)

            sigma_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                        self.csi_array[s+1], self.sigma_array[s], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
            sigma_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                        self.csi_array[s+1], sigma_potential, self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
            sigma_proposed_kernel['current'] = norm.logpdf(x = self.sigma_array[s] - inf_limit/np.sqrt(self.v_sigma)) 
            sigma_proposed_kernel['prop'] = norm.logpdf(x = sigma_potential - inf_limit/np.sqrt(self.v_sigma))
            
            # compute metropolis ratio
            sigma_metropolis_output = self.metropolis_step(sigma_potential, self.sigma_array[s], sigma_posterior_kernel, sigma_proposed_kernel)

            return sigma_metropolis_output

    def fit(self):

        for s in range(self.n_iteration):

            # draw from csi
            csi_sample = self.draw_csi(s)
            self.csi_array = np.insert(self.csi_array, obj = self.csi_array.shape[0], values = csi_sample, axis = 0)

            #draw from sigma
            sigma_sample = self.draw_sigma(s)
            self.sigma_array = np.insert(self.sigma_array, obj = self.sigma_array.shape[0], values = sigma_sample, axis = 0)
            #draw from u
            
            #draw from (mu_j, eta_j)
            
            #draw from p