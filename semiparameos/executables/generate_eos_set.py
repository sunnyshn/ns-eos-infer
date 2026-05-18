import gaussian_process

import h5py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.stats
from scipy.stats import uniform, loguniform
import os

import sys
sys.path.append("/home/sunny.ng/semiparameos/")
import utils
from utils import clean_conditioning

import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling import eos_prior

import tqdm

default_prior_set = eos_prior.EoSPriorSet.get_default()

### Nucleon mass in grams
mass_of_nucleon = 1/(6.02 * 10**23)
### Nucleon mass in MeV/c2
nucleon_mass_in_MeV = 939
### Conversion factors for Mev,fm to CGS units
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14

fm_in_cgs = 1e-13
rho_nuc_in_cgs = .16 * mass_of_nucleon / (fm_in_cgs)**3

default_hyperparameters = {"gamma_low":.1, "gamma_high":1.5,
                           "alpha_low":.1, "alpha_high":5.,
                           "corr_len_low":1e-3, "corr_len_high":1e1}

def white_noise(sigma, tabulation_points):
    wn_matrix = ((sigma)**2)*np.identity(len(tabulation_points))
    return wn_matrix
                                             
def rbf(gamma, corr_len, tabulation_points):
    """ 
    A function to calculate elements of a covariance kernel matrix using the radial basis
    (squared exponential) covariance function.
    
    Input: n-dimensional tabulation points for GPR, a correlation strength factor gamma,
    and correlation length scale parameter.
    
    Returns: nxn matrix containing covariance values according to squared exponential kernel. 
    """
    ### Need to convert densities to be in log for scaling factors
    ### (Assuming tabulation points are densities in cgs units)
    logrho = np.log(tabulation_points)
    sqexpo = lambda x, x_prime: gamma * np.exp(-(((np.linalg.norm([x - x_prime]))**2)/(2*(corr_len**2))))
    K = np.empty((len(tabulation_points), len(tabulation_points)))
    for i, dens in enumerate(logrho):
        for j, dens_prime in enumerate(logrho):
            K[i,j] = sqexpo(dens, dens_prime)
    return K                                             

def rqf(gamma, alpha, corr_len, tabulation_points):
    """
    This function calculates the elements of a covariance matrix according to
    the rational quadratic covariance function. Using this kernel equates to using
    multiple squared exponential kernels, across multiple length scales. Much like
    the squared exponential kernel, the gamma parameter determines the overall strength
    of the correlation, and l corresponds to the characteristic length scale of these
    correlations. Lastly, alpha acts as a weight to each of the length scales. 
    
    Note: It is required for alpha and l > 0. 
    """
    logrho = np.log(tabulation_points)
    ratquad = lambda x, x_prime: (gamma**2)*((1. +  (((np.linalg.norm([x - x_prime], 2))**2)/(2*alpha*(corr_len**2))))**(-alpha))
    K = np.empty((len(tabulation_points), len(tabulation_points)))
    for i, dens in enumerate(logrho):
        for j, dens_prime in enumerate(logrho):
            K[i,j] = ratquad(dens, dens_prime)
    return K

def collect_data(eos_posterior, rho_vals_to_interp,  eos_prior_set=default_prior_set,  N=10000, weight_columns=[]):
    """
    Get a set of EoSs based sampled from some posterior with some weight columns, this will have to be replaced if we use a different EoS
    distribution, most likely.
    """
    data = np.zeros((N, len(rho_vals_to_interp)))
    eos_chosen = eos_posterior.sample(size=N, weight_columns=weight_columns)["eos"]
    for i, eos_index in enumerate(eos_chosen):
        eos_data = pd.read_csv(eos_prior_set.get_eos_path(int(eos_index)))
        phi_data = np.log(np.gradient(eos_data["energy_densityc2"], eos_data["pressurec2"]) - 1.0)
        data[i, :] = interpolate.griddata(eos_data["baryon_density"], phi_data, rho_vals_to_interp)
        
    return data

def agn_collect_data(eos_dir, rho_vals_to_interp, tov_dir = ".", weights = False,
                     truncate_central = False, truncate_at_sat = False,
                     weight_columns = []):
    """
    Gather nuclear EoSs with equal weights for training the Gaussian Process.
    *Phase transitions may pose an issue.*
    """ 
    eos_path = f"{eos_dir}/eos_tables/" 
    eos_list = os.listdir(eos_path)
    data = np.zeros((len(eos_list), len(rho_vals_to_interp)))
    ### no need to sample according to weights --> use all
    if not weights:
        for i, eos in enumerate(eos_list):
            try:
                eos_data = pd.read_csv(f"{eos_path}/{eos}", header = 1, sep = r"\s+")
            except:
                raise KeyError(f"{eos} not found within directory. Available EoS's are: {eos_list}")
                
            if truncate_central:
                ### Truncate at central density corresponding to M_tov
                macros_file = f"{os.path.splitext(eos)[0]}.tab"
                eos_data = clean_conditioning.mtov_truncate(eos_dir,
                                                            macros_file,
                                                            eos_data) ### eos_data assumed to be a dataframe
            
            if truncate_at_sat: 
                ### Truncate after nuclear saturation density to ignore crust inhomogeneities
                ### BREAKS CONDITIONING. DUE TO MISSING VALUES WHERE INTERPOLANT SEARCHES FOR SOUND SPEED
                
                eos_data = eos_data[eos_data["n(fm-3)"] > 0.16] 
            
            phi_data = np.log(np.gradient(eos_data["rho(MeVfm-3)"],eos_data["P(MeVfm-3)"]) - 1.0)
            data[i,:] = interpolate.griddata(eos_data["n(fm-3)"]*mass_of_nucleon/fm_in_cgs**3, phi_data, rho_vals_to_interp)
            
            ### "Extend" nuclear training data interpolants to higher densities
            if any(np.isnan(x) for x in data[i,:]):
                idxs = np.where(np.isnan(data[i,:]))[0]
                if any(np.diff(idxs)) > 1: # check if missing points are close together
                    raise ValueError("Inconsistent sound speed detected.")
                elif all(np.diff(idxs)) == 1: # confirm NaN's are only at points past original interpolation range
                    causal_val = data[i,:][(np.where(np.isnan(data[i,:]))[0][0] - 1)]
                    data[i,:] = np.where(np.isnan(data[i,:]), causal_val, data[i,:]) # replace NaN's with phi based on last causal sound speed value
            
        return data

def get_consistent_metamodel(path="./eos_mm+chi+PSR/eos0.out", transition_density = rho_nuc_in_cgs):
    """
    Reinterpolate the metamodel to

    (1) Guarantee 1st law of thermodynmics consistency to a predictable precision
    (2) Guarantee there is a point where we need one for constructing the GP, and
    that the density of points is large enough for the sound speed calculation
    to be reliably carried out there.

    We also convert the table into cgs units.
    """
    # Constructing a metamodel with the boundary condition we need
    metamodel = pd.read_csv(path, names=["number_density", "energy_densityc2","pressure_nuclear"], sep = r"\s+", skiprows = 1)
    metamodel_cgs = pd.DataFrame({"baryon_density": np.array(metamodel["number_density"]) * mass_of_nucleon/fm_in_cgs**3, "energy_densityc2":np.array(metamodel["energy_densityc2"]),
    "pressurec2" :np.array(metamodel["pressure_nuclear"]) / nucleon_mass_in_MeV * mass_of_nucleon / fm_in_cgs**3})
    
    # rho_low_interp = np.geomspace(1e6, rho_nuc_in_cgs, 400)
    rho_low_interp = np.geomspace(1e6, transition_density, 400)
    p_interp = interpolate.griddata(metamodel_cgs["baryon_density"], metamodel_cgs["pressurec2"], rho_low_interp)
    # Only need for the boundary condition
    e_interp_0 = interpolate.griddata(metamodel_cgs["baryon_density"], metamodel_cgs["energy_densityc2"], rho_low_interp[0])
    # internal_energy = int p/rho dlogrho
    internal_energy_interp = integrate.cumulative_simpson(y = (p_interp/rho_low_interp), x = np.log(rho_low_interp), initial = 0.0) + (e_interp_0 - rho_low_interp[0])/rho_low_interp[0]
    metamodel_consistent = pd.DataFrame({"baryon_density": rho_low_interp, "pressurec2": p_interp, "energy_densityc2":rho_low_interp*(1 + internal_energy_interp)})
    return metamodel_consistent

cs2_of_data = lambda data : np.gradient(data["pressurec2"], data["energy_densityc2"])
phi_of_cs2 = lambda cs2 : np.log(1/cs2 - 1)
    
def get_example_draws(gp, metamodel, samples=15):

    metamodel_phi0 = phi_of_cs2(cs2_of_data(metamodel))
    gp_0 = gp.condition(metamodel_phi0[-1])
    eoss_new = []
    
    for i in range(samples):
        eos_sample = gp_0.sample()
        eos_new = gp_0.stitch_to_crust(eos_sample, metamodel)
        eoss_new.append(eos_new)
    return eoss_new

def marginalize_hyperparams(gamma_low = .1, gamma_high = .9,
                            alpha_low = .1, alpha_high = 5.,
                            corr_len_low = 1e-2, corr_len_high = 1e1):
    
    """
    Marginalize over hyperparameter values used in Gaussian Process generation
    via normal distributions over each hyperparameter. 
    
    Reminder: 
        - Gamma is unbounded, but yields numerical instability when significantly comparable
        to variance of phi values in GP.
        - Alpha must be a positive* integer.
        - Correlation length must also be postive-definite -- extremely* small values cause
        numerical instabilities in generating phi --> sound speed. 
    """
    
    ### Build uniform distributions for each hyperparameter and sample
    hyp_gamma = uniform(gamma_low, gamma_high).rvs()
    hyp_alpha = uniform(alpha_low, alpha_high).rvs()
    hyp_corr_len = loguniform(corr_len_low, corr_len_high).rvs()
    
    return {"gamma":hyp_gamma, "alpha":hyp_alpha, "corr_len":hyp_corr_len} 

def randomize_metamodel(mm_directory, downsample = True,
                        sample_size = 3000, seed = None):
    
    ### spit out a list of randomized metamodel eos numbers
    np.random.seed(seed)
    
    directory = os.listdir(mm_directory)
    total_num_eos = len(directory)
    mm_list = np.arange(total_num_eos)
    np.random.shuffle(mm_list) # randomize
    
    if downsample:
        mm_list = np.random.choice(mm_list, size = sample_size, replace = False) # don't allow for same EoS to be chosen again
    
    return mm_list

def write_eoss(eoss_new, tag="eos-draw", outdir="EoS", indices=None):
    if indices is None:
        indices = np.arange(len(eoss_new))
    for i, index in enumerate(indices):
        eoss_new[i].to_csv(f"{outdir}/{tag}-{index:04d}.csv", index=False)
        
def plot_eoss(eoss_new, ax=None, *args, **kwargs):
    if ax is None:
        ax=plt.gca()
    for eos in eoss_new:
        ax.plot(eos["baryon_density"], eos["pressurec2"], *args, **kwargs)
def plot_eoss_cs2(eoss_new, ax=None, *args, **kwargs):
    if ax is None:
        ax=plt.gca()
    for eos in eoss_new:
        ax.plot(eos["baryon_density"], np.gradient(eos["pressurec2"], eos["energy_densityc2"]), *args, **kwargs)

def get_full_and_restricted_mm_extensions(metamodel_number, gp, mm_eos_dir,
                                          rho_to_stitch = rho_nuc_in_cgs,
                                          random_mm = False,
                                          creation_index = None,
                                          epsilon=.1, # Controls the strength of correlations; range of [0.0 - 1.0)
                                          scale = 1.0, # Controls the overall size of GP band (range of ~0 -  1?)
                                          overall_outdir=".",
                                          uniform_var = False,
                                          sqexp = False,
                                          rational_quad = False,
                                          gamma = 1.,
                                          corr_len = 10e-3,
                                          alpha = 3.,
                                          uv_sigma = 0.1):
    """
    This function is the main executable for generating the Gaussian Process extensions.
    
    An aside on the hyperparameters: 
    
    Epsilon: represents the Pearson correlation coefficient. This general bounds for values
    is between 0.0 and 1.0 (eps ~ {0.0, 1.0}). Negative values will most likely yield negative
    covariances even possibly on the diagonal, thus possibly resulting in a non-positive semi-definite
    covariance matrix.
    * *this will break the conditioning process* * 

    Output: 
    - 'EoSNewRestricted' will yield the predictive covariance matrix additionally enhanced by the
    numerical stability coefficient epsilon, scale parameter, and the chosen covariance matri(ces) along
    with their hyperparameters. 
    """

    # metamodel_consistent = get_consistent_metamodel(f"{mm_eos_dir}/eos{metamodel_number}.out")
    metamodel_consistent = get_consistent_metamodel(f"{mm_eos_dir}/eos_mmpoly{metamodel_number}_clean.out", transition_density = rho_to_stitch)
        
    # Check overall out directory is created
    if not os.path.isdir(overall_outdir):
        os.makedirs(overall_outdir) 
        
    sigma = np.array([gp.covariance[i,i]**.5 for i in range(gp.covariance.shape[0])])
    
    # this is the normalized covariance
    covar_whitened = np.diag(1/sigma) @ gp.covariance @ np.diag(1/sigma)
    # This is hacky
    covar_whitened_enhanced = np.array([[epsilon * (1 - covar_whitened[i, j]) + covar_whitened[i,j] for i in range(gp.covariance.shape[0])] for j in range(gp.covariance.shape[0])])
    covar_new = np.diag(scale*sigma) @ covar_whitened_enhanced @ np.diag(scale*sigma)
    gp.covariance  = covar_new
    
    # Modify covariance matrix with kernels
    if uniform_var:
        gp.modify_correlations(white_noise(uv_sigma, gp.tabulation_points))
    
    if sqexp:
        gp.modify_correlations(rbf(gamma, corr_len, gp.tabulation_points))
    
    if rational_quad:
        gp.modify_correlations(rqf(gamma, alpha, corr_len, gp.tabulation_points))
        
    # Sample with the new covariance
    eoss_mod = get_example_draws(gp, metamodel_consistent)
    
    if random_mm:
        metamodel_number = creation_index
    
    mod_outdir = os.path.join(overall_outdir, f"EoSNewRestricted-{metamodel_number}")
    if not os.path.isdir(mod_outdir):
        os.makedirs(mod_outdir)
        write_eoss(eoss_mod, outdir=mod_outdir)
        

if __name__ == "__main__":
    
    verbose = True
    
    ### Range of density regime to create the GP in: 
    ### Stitching density will be the initial value in the density range here***
    rho_vals_to_interp = np.geomspace(rho_nuc_in_cgs, 10*rho_nuc_in_cgs, 100)
    
    ### Collect nuclear theory EoS's for GP training
    eos_samples = agn_collect_data(eos_dir = "/home/sunny.ng/XGEOSRecovery/eos_table/nuclear_set_PRD109-103029",
                                rho_vals_to_interp = rho_vals_to_interp,
                                truncate_central = True,
                                truncate_at_sat = False)
    
    # Create Gaussian Process
    gp = gaussian_process.GP.from_samples(eos_samples, np.log(rho_vals_to_interp))
    # fig, axs = plt.subplots(2, 1, sharex=True)

    ### Metamodel EoS directory
    mm_eos_dir = "/home/sunny.ng/semiparameos/set_huth_0.16_5PP/eos"
    
    ### Need to create an outdir to save EoS's into ####################################
    result_folder = "marginalized_hyp"
    overall_outdir = f"/home/sunny.ng/semiparameos/result/{result_folder}"
    ####################################################################################
    ### Hyperparameters ################################################################
    # Rational Quadratic Hyperparameters:
    gp_hyp_gamma = .7
    gp_hyp_alpha = 1.
    gp_hyp_corr_len = 1e-2
    ####################################################################################
    
    num_eos = 2000
    eos_to_be_used = np.arange(num_eos)
    marginalized_hyperparameters = True
    random_metamodel = True
    
    if random_metamodel:
        eos_to_be_used = randomize_metamodel(mm_eos_dir, sample_size = num_eos)
    
    for c_idx, metamodel_number in tqdm.tqdm(enumerate(eos_to_be_used)):
        if verbose:
            print(f"Generating extensions for eos_{metamodel_number}.")
        if marginalized_hyperparameters:
            sampled_hyperparams = marginalize_hyperparams(gamma_low = .7, gamma_high = 2.0,
                                                          alpha_low = 1e-1, alpha_high = 1e1,
                                                          corr_len_low = 1e-2, corr_len_high = 2e1)
            gp_hyp_gamma = sampled_hyperparams["gamma"]
            gp_hyp_alpha = sampled_hyperparams["alpha"]
            gp_hyp_corr_len = sampled_hyperparams["corr_len"]
            print(gp_hyp_gamma, gp_hyp_alpha, gp_hyp_corr_len)
        
        ### Main executable for generating EoS set
        get_full_and_restricted_mm_extensions(metamodel_number, gp,
                                              mm_eos_dir,
                                              rho_to_stitch = rho_vals_to_interp[0],
                                              random_mm = random_metamodel,
                                              creation_index = c_idx,
                                              rational_quad = True,
                                              gamma = gp_hyp_gamma,
                                              alpha = gp_hyp_alpha,
                                              corr_len = gp_hyp_corr_len,
                                              overall_outdir=overall_outdir)
        

    # Getting a GP approximant to the nonparametric EoS distribution
    # eos_posterior = EoSPosterior.from_csv("/home/isaac.legred/NewPulsar/collated_np_all_post.csv", label="astro")
    
    # This next line may have to be changed, basically we just need data in a table which represents EoS samples,
    # Each row must be an eos and each column must be the values of phi(logp)
     
    # N_samples = 1000
    # eos_samples = collect_data(eos_posterior, rho_vals_to_interp, N = N_samples, weight_columns=[result.WeightColumn("logweight_total")])
    
    
