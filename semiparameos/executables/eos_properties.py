import numpy as np
import pandas as pd
import os
import h5py
import bilby
import scipy.interpolate as interpolate
import scipy.stats as sst

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

def find_pts(data, idx):
    cs = np.gradient(data[idx]["pressurec2"], data[idx]["energy_densityc2"])
    num_neg = np.where(np.diff(cs[data[idx]["baryon_density"] > rho_nuc_in_cgs]) < 0.0)[0]
    num_pts = len(np.where(np.diff(num_neg) > 1.0)[0])
    return num_pts