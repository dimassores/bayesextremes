import numpy as np
from scipy.stats import gamma, truncnorm, norm, dirichlet

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
        
        # latent values used in metropolis hastings steps
        self.M = max(self.data)
        self.v_csi = np.sqrt(1)
        self.v_sigma = np.sqrt(1)
        self.v_u = np.sqrt(1)
        self.v_eta = np.sqrt(1)
        self.v_mu = np.sqrt(1)
        self.v_p = np.sqrt(1)

    def log_mpgd_posterior_kernel(self, p, mu, eta, u, csi, sigma, a_prior, b_prior, c_prior, d_prior, mu_u, sigma_u):

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

        data_under_u = self.data[self.data < u]
        data_over_u = self.data[self.data >= u]

        # gamma mixture kernel
        gm_kernel = 0

        for x_i in data_over_u:
            gm_kernel += np.log(p * gamma.pdf(x = x_i, a = eta, scale = mu/eta)).sum()
            
        # cumulated prob for the mixture model
        cumulated_prob = (p * gamma.cdf(u, a = eta, scale = mu/eta)).sum()
            
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
            
            csi_proposed_kernel['current'] = norm.logpdf(x = (self.csi_array[s] - inf_limit)/np.sqrt(self.v_csi))
            csi_proposed_kernel['prop'] = norm.logpdf(x = (csi_potential - inf_limit)/np.sqrt(self.v_csi))
            
            # compute metropolis ratio
            csi_metropolis_output = self.metropolis_step(csi_potential, self.csi_array[s], csi_posterior_kernel, csi_proposed_kernel)

            return csi_metropolis_output
            
    def draw_sigma(self, s):

        inf_limit = -self.csi_array[s+1]*(self.M - self.u_array[s])
        sup_limit = np.inf
        sigma_posterior_kernel = {}
        sigma_proposed_kernel = {}
        
        if self.csi_array[s+1] > 0:

            sigma_potential = gamma.rvs(a = self.sigma_array[s], scale = self.v_sigma/((self.sigma_array[s])**2))
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
            
            sigma_proposed_kernel['current'] = gamma.logpdf(self.sigma_array[s], a = self.sigma_array[s], scale = self.v_sigma/((self.sigma_array[s])**2))
            sigma_proposed_kernel['prop'] = gamma.logpdf(self.sigma_array[s], a = sigma_potential, scale = self.v_sigma/((sigma_potential)**2))
            
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
            
            sigma_proposed_kernel['current'] = norm.logpdf(x = (self.sigma_array[s] - inf_limit)/np.sqrt(self.v_sigma)) 
            sigma_proposed_kernel['prop'] = norm.logpdf(x = (sigma_potential - inf_limit)/np.sqrt(self.v_sigma))
            
            # compute metropolis ratio
            sigma_metropolis_output = self.metropolis_step(sigma_potential, self.sigma_array[s], sigma_posterior_kernel, sigma_proposed_kernel)

            return sigma_metropolis_output

    def draw_u(self, s):
        u_posterior_kernel = {}
        u_proposed_kernel = {}

        if self.csi_array[s+1] >= 0:
            inf_limit = min(self.data)
        else:
            inf_limit = self.M + self.sigma_array[s+1]/self.csi_array[s+1]

        sup_limit = np.inf
        
        # sample potential
        u_potential = truncnorm.rvs(a = inf_limit , b = sup_limit, loc = self.u_array[s], scale = self.v_u)

        u_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], self.u_array[s], 
                                                                        self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
        u_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s], self.eta_array[s], u_potential, 
                                                                    self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                    self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                    self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                    self.prior_values['sigma_u'])
        
        u_proposed_kernel['current'] = norm.logpdf(x = (self.u_array[s] - inf_limit)/np.sqrt(self.v_u)) 
        u_proposed_kernel['prop'] = norm.logpdf(x = (u_potential - inf_limit)/np.sqrt(self.v_u))

        u_metropolis_output = self.metropolis_step(u_potential, self.u_array[s], u_posterior_kernel, u_proposed_kernel)
        return u_metropolis_output


    def draw_mu_eta(self, s):
        eta_potential = gamma.rvs(a = self.eta_array[s], scale = self.v_eta/(self.eta_array[s])**2)

        final_mu = []
        final_eta = []
        for j in range(len(self.k)):
            mu_posterior_kernel = {}
            mu_proposed_kernel = {}
            mu_potential = gamma.rvs(a = self.mu_array[s][j], scale = self.v_mu/(self.mu_array[s][j])**2)

            if j == 0:
                if mu_potential < self.mu_array[s][j+1]:
                    mu_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s][j], self.eta_array[s][j], self.u_array[s+1], 
                                                                        self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
                    mu_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], mu_potential, eta_potential[j], self.u_array[s+1], 
                                                                                self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                                self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                                self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                                self.prior_values['sigma_u'])
                    
                    mu_proposed_kernel['current'] = (gamma.logpdf(x = self.mu_array[s][j], a = self.mu_array[s][j], scale = self.v_mu/(self.mu_array[s][j])**2) 
                                                    + gamma.logpdf(x = self.eta_array[s][j], a = self.eta_array[s][j], scale = self.v_eta/(self.eta_array[s][j])**2))

                    mu_proposed_kernel['prop'] = (gamma.logpdf(x = mu_potential, a = self.mu_array[s][j], scale = self.v_mu/(self.mu_array[s][j])**2) 
                                                    + gamma.logpdf(x = eta_potential[j], a = self.eta_array[s][j], scale = self.v_eta/(self.eta_array[s][j])**2))

                    
                    mu_metropolis_output, eta_metropolis_output = self.metropolis_step((mu_potential, eta_potential[j]), (self.mu_array[s][j], self.eta_array[s][j]), mu_posterior_kernel, mu_proposed_kernel)
                    
                    final_mu.append(mu_metropolis_output)
                    final_eta.append(eta_metropolis_output)

                else:
                    final_mu.append(self.mu_array[s][j])
                    final_eta.append(self.eta_array[s][j])
            else: 
                if (mu_potential > self.mu_array[s][j-1]) and (mu_potential < self.mu_array[s][j+1]):
                    mu_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s][j], self.eta_array[s][j], self.u_array[s+1], 
                                                                        self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
                    mu_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], mu_potential, eta_potential[j], self.u_array[s+1], 
                                                                                self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                                self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                                self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                                self.prior_values['sigma_u'])
                    
                    mu_proposed_kernel['current'] = (gamma.logpdf(x = self.mu_array[s][j], a = self.mu_array[s][j], scale = self.v_mu/(self.mu_array[s][j])**2) 
                                                    + gamma.logpdf(x = self.eta_array[s][j], a = self.eta_array[s][j], scale = self.v_eta/(self.eta_array[s][j])**2))

                    mu_proposed_kernel['prop'] = (gamma.logpdf(x = mu_potential, a = self.mu_array[s][j], scale = self.v_mu/(self.mu_array[s][j])**2) 
                                                    + gamma.logpdf(x = eta_potential[j], a = self.eta_array[s][j], scale = self.v_eta/(self.eta_array[s][j])**2))

                    
                    mu_metropolis_output, eta_metropolis_output = self.metropolis_step((mu_potential, eta_potential[j]), (self.mu_array[s][j], self.eta_array[s][j]), mu_posterior_kernel, mu_proposed_kernel)
                    
                    final_mu.append(mu_metropolis_output)
                    final_eta.append(eta_metropolis_output)
                else:
                    final_mu.append(self.mu_array[s][j])
                    final_eta.append(self.eta_array[s][j])

        return np.array(final_mu), np.array(final_eta)

    def draw_p(self, s):

        p_posterior_kernel = {}
        p_proposed_kernel = {}

        
        # sample potential
        p_potential = dirichlet.rvs(self.v_p*self.p_array[s])

        p_posterior_kernel['current'] = self.log_mpgd_posterior_kernel(self.data, self.p_array[s], self.mu_array[s+1], self.eta_array[s+1], self.u_array[s+1], 
                                                                        self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                        self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                        self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                        self.prior_values['sigma_u'])
            
        p_posterior_kernel['prop'] = self.log_mpgd_posterior_kernel(self.data, p_potential, self.mu_array[s+1], self.eta_array[s+1], self.u_array[s+1], 
                                                                    self.csi_array[s+1], self.sigma_array[s+1], self.prior_values['a_prior'], 
                                                                    self.prior_values['b_prior'], self.prior_values['c_prior'], 
                                                                    self.prior_values['d_prior'], self.prior_values['mu_u'], 
                                                                    self.prior_values['sigma_u'])
        
        p_proposed_kernel['current'] = dirichlet.logpdf(self.p_array[s]) 
        p_proposed_kernel['prop'] = dirichlet.logpdf(p_potential) 

        p_metropolis_output = self.metropolis_step(p_potential, self.p_array[s], p_posterior_kernel, p_proposed_kernel)

        return p_metropolis_output


    def fit(self):

        for s in range(self.n_iteration):
            
            # draw from csi
            csi_sample = self.draw_csi(s)
            self.csi_array = np.insert(self.csi_array, obj = self.csi_array.shape[0], values = csi_sample, axis = 0)

            # draw from sigma
            sigma_sample = self.draw_sigma(s)
            self.sigma_array = np.insert(self.sigma_array, obj = self.sigma_array.shape[0], values = sigma_sample, axis = 0)

            # draw from u
            u_sample  = self.draw_u(s)
            self.u_array = np.insert(self.u_array, obj = self.u_array.shape[0], values = u_sample, axis = 0)

            # draw from mu and eta
            mu_sample, eta_sample  = self.draw_mu_eta(s)
            self.eta_array = np.insert(self.eta_array, obj = self.eta_array.shape[0], values = eta_sample, axis = 0)
            self.mu_array = np.insert(self.mu_array, obj = self.mu_array.shape[0], values = mu_sample, axis = 0)
            
            # draw from p

            p_sample = self.draw_p(s)
            self.p_array = np.insert(self.p_array, obj = self.p_array.shape[0], values = p_sample, axis = 0)

    def get_p_chain(self):
        return self.p_array

    def get_mu_chain(self):
        return self.mu_array

    def get_eta_chain(self):
        return self.eta_array

    def get_u_chain(self):
        return self.u_array

    def get_sigma_chain(self):
        return self.sigma_array

    def get_csi_chain(self):
        return self.csi_array

