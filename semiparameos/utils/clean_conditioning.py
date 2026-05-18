__author__ = "sunnyng.sshn@gmail.com"
# --------------------------------------

### Importing standard packages
import numpy as np
import pandas as pd
import h5py
import os
import scipy
import bilby

### Units #################################################
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
############################################################

def mev_to_cgs(density):
    cgs_density = density*MeV_to_gcm3
    return cgs_density

def smoothen_PT(eos_file, directory_path, save = False):
    
    """
    We want to counteract sound speed calculation failures due to divergences in the numerical derivatives. Primary method
    of handling phase transitions will be to interpolate over adjacent pressures in pressure-density.
    
    Caveat: This will induce a favoring for emulating weak (Gibbs) phase transitions in the EoS's. 
    """
    
    ### assuming EoS is in .dat format as per Fortin+ PRD 2020 EoS's
    nucdf = pd.read_csv(f"{directory_path}/{eos_file}", header = 1, sep = "\s+")

    ### Want to interpolate over phase transition points in the data --> repeated values in the pressure as a funciton of density
    press = nucdf["P(MeVfm-3)"]
    edens = nucdf["rho(MeVfm-3)"]
    
    ### Find indices of PT pressure points
    press_inc = np.diff(press) # search for PT's via zeroes in difference array
    pt_idxs = np.where(press_inc <= 0.0)[0]
    
    ### Let's keep on using np.diff... now searching for segments of PT's
    cont_pts = np.where(np.diff(pt_idxs) == 1.) ### should return indices of points from segmented phase transitions
    #grid_points = np.linspace(dens[cont_pts][0], dens[cont_pts][-1], len(cont_pts)) # not sure if necessary
    ### TODO: segment multiple PTs...
    
    interp_press = np.interp(dens[cont_pts][0], dens[cont_pts][0], press[cont_pts][0])
    nucdf["P(MeVfm-3)"].iloc[cont_pts] = interp_press
    smooth_eos = nucdf
    if save:
        nucdf.to_csv(f"{directory_path}/{eos_file}_cleaned.csv")
    
    return smooth_eos

def mtov_truncate(directory, macros_file, eos_df, branches = "singular", save_eoss = False, **kwargs):
    
    """
    A helper function that intakes an equation of state (requires macro observables and pressure-density quantities)
    and searches for the maximum mass (TOV mass) and the corresponding central density.
    Truncates the equation of state up to the max central density, and returns the truncated EoS. 
    """
    
    path = os.path.join(os.getcwd(), directory)
    ### assumes singular branch in M(R) as opposed to multiple branches, finding first maximum mass encountered
    if branches == "singular":
        
        ext = os.path.splitext(macros_file)[-1]
        if ext == ".csv":
            macro_df = pd.read_csv(macros_file, **kwargs)
        
        elif ext == ".tab":
            ### use columns with central pressure, central density, radius, mass, and tidal
            macros = np.loadtxt(f"{path}/tov_tables/{macros_file}", unpack = False,
                                usecols = (0,1,3,4,10), ## assumes consistent naming convention
                    dtype=[('Pressure_central', 'f8'), ('rest_mass_density_central', 'f8'), ('Radius', 'f8'),
                             ('Mass', 'f8'), ('Lambda', 'f8')])
            macro_df = pd.DataFrame(macros)
        
        else:
            raise KeyError("File format currently not supported.")
            
        ### Find M_TOV as a maximum mass search
        max_mass_idx = macro_df["Mass"].idxmax()
        ### Assumes central density value stored is in CGS
        cen_dens = macro_df["rest_mass_density_central"].iloc[max_mass_idx]

        ### Read in EoS file
        # eos_df = pd.read_csv(f"{path}/eos_tables/{eos_file}", header = 1, sep = "\s+")
    
        ### WE CAN SIMPLY SUPPLY THE EOS DATAFRAME WHILE WE LOAD EOSS INTO GP GENERATION
        
        ### Find and truncate EoS values at central density corresponding to M_TOV
        trunc_idxs = np.where(mev_to_cgs(eos_df["rho(MeVfm-3)"]) <= cen_dens) 
        trunc_eos_df = eos_df.iloc[:len(np.where(mev_to_cgs(eos_df["rho(MeVfm-3)"]) <= cen_dens)[0])]
            
        if save_eoss == True:
            trunc_eos_df.to_csv(f"{eos_file}_causal.csv", index = False)
            return
        
        elif save_eoss == False:
            
            return trunc_eos_df
    
    else:
        print("Can only handle singular branch for now.")
    
    return


if __name__ == "__main__":
    
    eos_dir = "/home/sunny.ng/XGEOSRecovery/eos_table/nuclear_set_PRD109-103029"
    eos = "DD2_noY.dat"
    macros = "DD2_noY.tab"

    # mtov_truncate(macros, eos, eos_dir)