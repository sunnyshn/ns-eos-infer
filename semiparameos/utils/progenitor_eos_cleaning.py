__author__ = "sunnyng.sshn@gmail.com"
# --------------------------------------

import numpy as np
import pandas as pd
import h5py
import warnings
import bilby
import os
import sys
sys.path.append("/home/sunny.ng/semiparameos/executables")

import TOVSolver
from TOVSolver import monotonicity_check

def mm_eos_cleaning(eos_folder_path, idx):
    # clean all EoS's with name format "eos{#}.out"
    metamodel = pd.read_csv(f"{eos_folder_path}/eos{idx}.out", names=["number_density", "energy_densityc2","pressure_nuclear"], sep = "\s+")
    ### monotonicity check returns variables in order of Pressure, Energy density, and then baryon density
    corr_mm_eos = monotonicity_check(metamodel["pressure_nuclear"], metamodel["energy_densityc2"], metamodel["number_density"])
    mono_corr_eos = pd.DataFrame(corr_mm_eos)
    ### re-arrange variables to original metamodel EoS format
    mono_corr_eos = mono_corr_eos[["baryon_density", "energy_densityc2", "pressurec2"]] # 
    np.savetxt(f"{eos_folder_path}/eos{idx}.out", mono_corr_eos)
    return

if __name__ == "__main__":

    mm_eos_path = "/home/sunny.ng/lwp/meta_model_eos/eos_tables"
    directory = os.listdir(mm_eos_path)
    verbose = True
    
    for mm_eos in range(len(directory)):
        mm_eos_cleaning(mm_eos_path, mm_eos)
        if verbose:
            print(f"Metamodel EoS {mm_eos} cleaned.")
    
    
