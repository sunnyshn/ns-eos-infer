__author__ = "sunnyng.sshn@gmail.com"
# --------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bilby
import scipy as sp
from scipy.stats import norm, skewnorm
from scipy.optimize import minimize_scalar
import h5py
import os
import tqdm
import analyze_nicer
try:
    import temperance
except: ImportError("Cannot import temperance.")
try:
    import universality
except: ImportError("Cannot import universality.")

# attributes = ["pressurec2", "energy_densityc2", "baryon_densityc2", "M", "R", "Lambda"]
eos_keys = ["eos","ns","mm_id"]
attributes = {"pressure":["pressurec2","eos"], "energy_density":["energy_densityc2","eos"], "baryon_density":["baryon_densityc2","eos"],
              "mass":["M","ns"], "radius":["R","ns"], "tidal":["Lambda","ns"]}

psr_events_dict = {"J0348":{"Mass":2.01, "lower_bound":0.04, "upper_bound":0.04},
                   "J0740":{"Mass":2.08, "lower_bound":0.07, "upper_bound":0.07},
                   "J1614":{"Mass":1.908, "lower_bound":0.016, "upper_bound":0.016}}
 
def fit_skewnorm_from_asymmetric_bounds(mean, lower_error, upper_error, confidence=0.683):
    lower_bound = mean - lower_error
    upper_bound = mean + upper_error
    scale_guess = (lower_error + upper_error) / 2
    def objective(alpha):
        delta = alpha / np.sqrt(1 + alpha**2)
        loc = mean - scale_guess * delta * np.sqrt(2 / np.pi)
        cdf_upper = skewnorm.cdf(upper_bound, alpha, loc=loc, scale=scale_guess)
        cdf_lower = skewnorm.cdf(lower_bound, alpha, loc=loc, scale=scale_guess)
        return abs((cdf_upper - cdf_lower) - confidence)
    result = minimize_scalar(objective, bounds=(-20, 20), method='bounded')
    alpha = result.x
    delta = alpha / np.sqrt(1 + alpha**2)
    loc = mean - scale_guess * delta * np.sqrt(2 / np.pi)
    return alpha, loc, scale_guess

def load_eos(eos_samples, idx, tgt_columns = None, all_columns = False, index_weights = False):
    ### tgt_columns consist of: "pressure, energy_density, baryon_density, mass, radius, tidal"
    eos_idx = f"eos_{idx:06d}"
    if all_columns:
        pressure = eos_samples["eos"][eos_idx]["pressurec2"]
        edens = eos_samples["eos"][eos_idx]["energy_densityc2"]
        bary_dens = eos_samples["eos"][eos_idx]["baryon_density"]
        masses = eos_samples["ns"][eos_idx]["M"]
        radii = eos_samples["ns"][eos_idx]["R"]
        tidal = eos_samples["ns"][eos_idx]["Lambda"]
        eos = {"pressurec2":pressure, "energy_densityc2":edens, "baryon_density":bary_dens,
               "M":masses, "R":radii, "Lambda":tidal}
        return eos
    else:
        eos = {}
        for attr in tgt_columns:
            eos[attr] = eos_samples[attributes[attr][1]][eos_idx][attributes[attr][0]]
    return eos

def get_single_mmax(eos, idx):
    eos_idx = f"eos_{idx:06d}"
    # Assumes M_tov = M_max for a given EoS
    mmax = max(eos["ns"][eos_idx]["M"])
    return mmax

def get_all_mmax(eos_samples, verbose = True):
    eos_mmax = []
    if verbose:
        print(f"Obtaining M_max for EoSs...")
    for eos_num in tqdm.tqdm(range(len(eos_samples["eos"]))):
        try:
            eos_mmax.append(get_single_mmax(eos_samples, eos_num))
        except:
            eos_mmax.append(0.5) # if EoS doesn't have any appended masses
    return np.array(eos_mmax)

def weigh_pulsar(mmax, mass, sigma, skew = False, alpha = 0.0,
                 mmin = 1.0, log_outputs = True):
    """
    This function injests either a singular, or an array of, EoS's M_TOV's at a time.
    Under the assumption that M_tov == M_max for a given EoS. Arbitrary choice of M_min = 1.0.
    """
    likelihood = sp.stats.norm.cdf(mmax, loc = mass, scale = sigma)
    normalization = (1./(np.array(mmax) - mmin)) # flat mass prior assumption
    weight = (normalization * likelihood)
    
    if skew:
        skew_likelihood = sp.stats.skewnorm.cdf(mmax, alpha = alpha, loc = mass, scale = sigma)
        weight = (normalization * skew_likelihood)
        
    if log_outputs:
        try:
            weight = np.log(weight)
        except:
            RunTimeWarning(f"Unable to logweight EoS likelihood, returning 0 likelihood equivalent.")
            weight = -np.inf
        return np.array(weight)
    
    else:
        return np.array(weight)

def get_joint_likelihood(all_astro_weights_file,
                         psr_events = [], # expects string name for each event, i.e events[i] = "J0740"
                         gw_events = [],
                         xray_events = [],
                         log = True,
                         save_and_overwrite = False):
    
    ### ASSUMPTION: all marginalized weights are in log
    joint_astro_df = pd.read_csv(all_astro_weights_file)
    joint_likelihood = np.zeros(len(joint_astro_df["eos_index"])) # array should be as large as the total number of EoS's
    if psr_events:
        for psr_obs in psr_events:
            joint_likelihood += joint_astro_df[f"PSR_{psr_obs}"]
    
    if gw_events:
        for gw_obs in gw_events:
            joint_likelihood += joint_astro_df[f"GW_{gw_obs}"]

    if xray_events:
        for xray_obs in xray_events:
            joint_likelihood += joint_astro_df[f"XRay_{xray_obs}"]
            
    joint_astro_df["joint_logweight"] = joint_likelihood
    final_joint_astro_df = joint_astro_df.copy()
    
    if save_and_overwrite:
        final_joint_astro_df.to_csv(all_astro_weights_file, index = False)
    
    return final_joint_astro_df

def process_weights_to_csv(weights_df, # already contains PSR weights
                           gw_weights = None, # path to directory holding GW weights
                           nicer_weights = None, # path to directory holding NICER weights (???)
                           save_to_csv = False,
                           outpath_dir = ".",
                           output_filename = "all_astro_likelihoods"):
    
    # check if output directory to house weights in, exists
    if os.path.isdir(outpath_dir):
        pass
    elif not os.path.isdir(outpath_dir):
        os.makedirs(outpath_dir)
    
    if gw_weights:
        # TODO: Load weights from LWP output (usually logweights)
#         lwp_weights = "/home/sunny.ng/lwp/lwp_result/GW170817_GP_lwp/result/GW170817_GP_lwp_eos.csv"
#         for likelihood_file in gw_weights:
#             try:
#                 gw_logvarweight = np.array(pd.read_csv(lwp_weights)["logvarweight"])
                
#             except:
#                 if FileNotFoundError:
#                     raise FileNotFoundError("GW weights are missing! Check path again.")
        pass
    elif nicer_weights:
        # nicer_weights = pd.read_csv(nicer_weights)
        # weights_df[f"XRay_{}"]
        pass
    
    if save_to_csv:
            weights_df.to_csv(f"{outpath_dir}/{output_filename}.csv", index = False)
    
    return weights_df

def clean_weights(weights_df):
    
    ### don't foresee EoS indices spitting any NaN values, therefore this would only clean weights
    ### use with caution, function is oblivious to an entire set of EoS's returning NaN's which would
    ### indicate something possibly breaking
    
    weights_df = weights_df.fillna(-np.inf) 
    
    return weights_df

def weigh_all_pulsar(likelihood_df, eos_mtov, verbose = True): 
    
    for event in psr_events_dict:
        if verbose:
            print(f"Weighing EoS's with PSR {event}...")
        
        test_mass = psr_events_dict[event]["Mass"]
        test_lower = psr_events_dict[event]["lower_bound"]
        test_upper = psr_events_dict[event]["upper_bound"]
        
        ### Getting PSR information
        pulsar_mass = test_mass
        if test_lower == test_upper:
            measure_unc = test_upper
        elif test_lower != test_upper:
            print(f"PSR {event} has asymmetric bounds! Fitting to skewed normal distribution for likelihood calculation...")
            skew_alpha, skew_mass, skew_scale = fit_skewnorm_from_asymmetric_bounds(test_mass,
                                                                    test_lower,
                                                                    test_upper)
            pulsar_mass = skew_mass
            measure_unc = skew_scale
        
        try:
            psr_weights = weigh_pulsar(eos_mtov, pulsar_mass, measure_unc) # returns numpy array holding PSR weights
            likelihood_df[f"PSR_{event}"] = psr_weights
            
            ### clean out NaN likelihood values
            likelihood_df = clean_weights(likelihood_df)
            
            astro_likelihood_df = likelihood_df.copy()
            
        except:
            raise ValueError("Weighting error.")
                     
    return astro_likelihood_df
                     
                     
if __name__ == "__main__":
    
    ### Begin astrophysical weighing with pulsar likelihoods, then concatenate GW and NICER weights afterwards
    verbose = True
    
    ### EoS samples location #####################################################
    eos_file_path = "/home/sunny.ng/semiparameos/generated_eoss/NLSLTR_EOS_prior.h5"
    result_file_name = os.path.splitext(os.path.split(eos_file_path)[1])[0]
    
    try:
        eos_samples = h5py.File(eos_file_path)
    except:
        if FileNotFoundError:
            raise ("EoS samples missing!")
    ##############################################################################

    ### Instantiate weights dataframe 
    init_astro_df = pd.DataFrame(eos_samples["id"], columns = ["eos_index"])
    
    ### Get M_tov from each EoS and store it along with the likelihoods
    eos_mtov = get_all_mmax(eos_samples, verbose)
    init_astro_df["Mmax"] = eos_mtov
    
    ### Weigh multiple pulsars
    all_astro_weights_df = weigh_all_pulsar(init_astro_df, eos_mtov)
    
    ### Process weights to .csv
    process_weights_to_csv(all_astro_weights_df, save_to_csv = True,
                           outpath_dir = "/home/sunny.ng/semiparameos/astro_likelihoods",
                           output_filename = f"{result_file_name}_posterior")
    
    ### Obtain joint likelihood
    psr_for_weighing = ["J0348","J0740","J1614"]
    get_joint_likelihood(f"/home/sunny.ng/semiparameos/astro_likelihoods/{result_file_name}_posterior.csv",
                         psr_events = psr_for_weighing,
                         save_and_overwrite = True) 
                
    eos_samples.close()
