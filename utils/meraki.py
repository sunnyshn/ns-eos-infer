import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as font_manager
import os
import bilby
import scipy.interpolate as interpolate
import h5py
import scipy.stats as sst 
import corner
from scipy.stats import gaussian_kde
import corner
import matplotlib.patches as mpatches
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


### Units
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14
sat_dens = 2.8*(np.power(10.0,14.))
c = bilby.core.utils.speed_of_light
c_cgs = c*(1e2)

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

class EOS:
    def __init__(self, samples, eos_to_be_used, load_micro=False, load_macro=False):
        """
        An EoS object to obtain diagnostics from with convinience.
        """
        self.samples = samples
        self.eos_index_rng = eos_to_be_used

        if load_micro:
            try:
                self.load_micro_data()
            except Exception as error:
                raise KeyError("Unable to map EoS microscopic data to enumerated list.") from error
        if load_macro:
            try:
                self.load_macro_data()
            except Exception as error:
                raise KeyError("Unable ot map EoS macroscopic data to enumerated list.") from error
        
    @staticmethod
    def load_samples(h5file, filetype = "h5", load_micro = False, load_macro = False):

        if filetype == "h5":
            samples = h5py.File(h5file)
            eos_to_be_used = np.arange(len(samples["eos"]))
            
            return EOS(samples, eos_to_be_used, load_micro = load_micro, load_macro = load_macro)
        else:
            raise KeyError("File type not supported.")

    def load_micro_data(self):
        mic_data = {eos_num : np.array(self.samples['eos'][eos_id]) for eos_num, eos_id in enumerate(self.samples['eos'])}
        self.micro_data = mic_data

    def load_macro_data(self):
        mac_data = {eos_num : np.array(self.samples['ns'][eos_id]) for eos_num, eos_id in enumerate(self.samples['ns'])}
        self.macro_data = mac_data

    except ValueError:
        print("All EoS's are unlikely.")

    def get_pe_quantiles(self, interp_densities, quantiles, cgs_press = True, get_median = False, store_sigmas = False):
        pdvals = []
        # Obtain quantiles for the prior set of EoS's
        for eos in self.eos_index_rng:
            eos_densities = self.micro_data[eos]["baryon_density"]
            eos_pressures = self.micro_data[eos]["pressurec2"]
            if cgs_press:
                eos_pressures = self.micro_data[eos]["pressurec2"]*gcm3_to_dynecm2
            pdvals.append(np.interp(interp_densities, eos_densities, eos_pressures))
        pdvals = np.array(pdvals)
        
        # Create object to hold upper and lower bounds 
        pd_sigmas = np.zeros((len(interp_densities),len(quantiles)))
        if get_median:
            pd_median = np.zeros((len(interp_densities), 1))
        for i in range(len(interp_densities)):
            pd_sigmas[i]=np.percentile(np.array(pdvals[:,i]),quantiles)
            if get_median:
                pd_median[i] = np.median(np.array(pdvals[:,i]))
        
        del pdvals
        if get_median:
            return pd_sigmas, pd_median
        else:
            return pd_sigmas

        if store_sigmas:
            self.pd_quantiles = pd_sigmas

    def get_cs_quantiles(self, interp_densities, quantiles, get_median = False, store_sigmas = False):
        csvals = []
        # Obtain sound speed quantiles for a set of EoS's
        for eos in self.eos_index_rng:
            eos_rest_mass = self.micro_data[eos]["baryon_density"]
            eos_densities = self.micro_data[eos]["energy_densityc2"]
            eos_pressures = self.micro_data[eos]["pressurec2"]
            eos_cs = np.gradient(eos_pressures, eos_densities)
            csvals.append(np.interp(interp_densities, eos_rest_mass, eos_cs))
        csvals = np.array(csvals)
    
        ### Obtaining quantiles 
        cs_sigmas = np.zeros((len(interp_densities),len(quantiles)))
        if get_median:
            cs_median = np.zeros((len(interp_densities), 1))
        for i in range(len(interp_densities)):
            cs_sigmas[i] = np.percentile(np.array(csvals[:,i]), quantiles)
            if get_median:
                cs_median[i] = np.median(np.array(csvals[:,i]))
        
        del csvals
        if get_median:
            return cs_sigmas, cs_median
        else:
            return cs_sigmas

        if store_sigmas:
            self.cs_quantiles = cs_sigmas

### Functions outside EOS object class -----------------------------------------------------------------------------

def get_and_load_weights(weights, astro_tag, eos_set_to_use, Neff = False, replace_eos = True, size_data = 10000):
    """
    Returns a collection of downsampled EoS according to their likelihood from some astrophysical observation.
    Assumes correlation and matched weights with respective EoS set via "eos_to_be_used".
    """
    ### sample EoS's according to astro weights
    try:
        # Resampling EoS's with weighted values according to likelihoods
        astro_exp_weights = np.exp(weights[astro_tag])
        # Normalize values to have probabilities add up to 1
        astro_exp_weights = astro_exp_weights/(sum(astro_exp_weights))
        # Resample EoS's according to weights --> gives posterior distribution of EoS's
        astro_weight_eos = np.random.choice(eos_set_to_use, size=size_data, replace=replace_eos, p=astro_exp_weights)
        
        if Neff:
            Neff = ((sum(astro_exp_weights))**2)/(sum(astro_exp_weights**2))
            print(f"Number of effective EoS's: {int(Neff)}")
            
        return astro_weight_eos

### Functions for calculating properties of dense matter -----------------------------------------------------------
def phi(eps, press):
    try:
        phi = np.log((np.gradient(eps, press) - 1.0))
    except:
        print(f"Error at {eps}.")
    return phi

def trace_anom(pressure, density): ### Trace Anomaly calculation
    trace_anomaly = (1./3.) - (pressure/density)
    return trace_anomaly

def chem_potent(p, epsilon, baryon): ### Chemical Potential Calculation, defined via Enthalpy at zero temperature
    mu = ((p + epsilon)/baryon)
    return mu

