"""
Utilities for generating and sampling from a Gaussian Process representation of an EoS Posterior
"""
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.interpolate as interpolate

class GP:
    def __init__(self, mean, covariance, tabulation_points):
        """
        Create a Gaussian Process from mean, covariance, and the points 
        which are tabulated at
        """
        self.mean = mean
        self.covariance = covariance
        self.tabulation_points = tabulation_points
    @staticmethod
    def from_samples(samples, tabulation_points, weights=None, *args, **kwargs):
        """
        Create a gaussian process from samples from a distribution

        In effect, approximate a distribution by it's first two moments
        """
        mean = np.mean(samples, axis=0)
        covariance = np.cov(samples, rowvar=False)
        tabulation_points = tabulation_points
        return GP(mean, covariance, tabulation_points)
    def sample(self, tabulation_name="logrho", sample_name="phi"):
        eos_sample = np.random.multivariate_normal(self.mean, self.covariance)
        return pd.DataFrame({tabulation_name:self.tabulation_points, sample_name:eos_sample})
    def stitch_to_crust(self, eos_sample,  crust_eos, rho_column="baryon_density", p_column = "pressurec2", e_column = "energy_densityc2"):
        rho_boundary = np.array(crust_eos[rho_column])[-1]
        p_boundary = np.array(crust_eos[p_column])[-1]
        e_boundary = np.array(crust_eos[e_column])[-1]
        rho_to_integrate_to = np.exp(np.array(eos_sample["logrho"]))
        cs2_of_rho = interpolate.interp1d(rho_to_integrate_to, 1/(np.exp(eos_sample["phi"]) + 1), kind = 7)
        
        def eos_rhs (rho, e_and_p):
            e = e_and_p[0]
            p = e_and_p[1]
            de_drho = (e + p)/rho
            return np.array([de_drho, de_drho * cs2_of_rho(rho)])
        sol = integrate.solve_ivp(eos_rhs,
                                  (rho_to_integrate_to[0],
                                   rho_to_integrate_to[-1]),
                                  y0=np.array([e_boundary, p_boundary]),
                                  t_eval=rho_to_integrate_to,
                                  dense_output = True,
                                  method = "LSODA")
        e_extension = sol.y[0, :]
        p_extension = sol.y[1, :]
        return pd.DataFrame({rho_column:np.concatenate([crust_eos[rho_column],
                                                        rho_to_integrate_to[1:]]),
                             p_column:np.concatenate([crust_eos[p_column],
                                                      p_extension[1:]]),
                             e_column:np.concatenate([crust_eos[e_column],
                                                     e_extension[1:]
                             ])})
    def condition(self, value, point_to_condition_at="minimal"):
        """
        An executable to generate the predictive mean function value and predictive covariance matrix of our updated Gaussian Process 
        based on observational data. An initialized GPR draw/set of extensions will only need to do minimal point conditioning, where we
        condition on the last meta-model phi value (since our covariance matrix will be instantiated on nuclear theory EoS's). However,
        during the EoS regeneration process, we'll need to be able to condition our covariance matrix on surviving EoS's from post-LWP analyses.
        """
        if point_to_condition_at == "minimal":
            mu_1 = self.mean[1:]
            mu_2 = self.mean[0]
            sigma_11 = self.covariance[1:, 1:]
            sigma_12 = self.covariance[0, 1:]
            sigma_22 = self.covariance[0,0]
            mu_new = mu_1 + sigma_12 * 1/sigma_22 * (value - mu_2)
            sigma_new =  sigma_11 - 1/sigma_22 * np.outer(sigma_12, sigma_12)
        else:
            raise ValueError("have to condition at minimal point for now")
        return GP(mu_new, sigma_new, tabulation_points = self.tabulation_points[1:])
        
        
    def modify_correlations(self, kernel):
        change_to_covariance = kernel
        self.covariance += change_to_covariance
        
