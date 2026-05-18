__author__ = "sunnyng.sshn@gmail.com"
# --------------------------------------

# Standard Libraries ------------
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import h5py
import warnings
import os
import tqdm
# -------------------------------
# Non-standard Libraries
import bilby
import pyreprimand as pyr
import lal
import lalsimulation as lalsim
# -------------------------------
### Get rid of warnings from importing LAL
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

### Unit conversion/Constants -------------------------------------------------
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14
sat_dens = 2.8*(np.power(10.0,14.)) ### CGS
c = bilby.core.utils.speed_of_light
### Nucleon mass in grams
mass_of_nucleon = 1/(6.02 * 10**23)
### Nucleon mass in MeV/c2
nucleon_mass_in_MeV = 939
### Conversion factors for Mev,fm to CGS units
fm_in_cgs = 1e-13
rho_nuc_in_cgs = .16 * mass_of_nucleon / (fm_in_cgs)**3
### LAL conversions
MeV_to_J = 1.6021766339999e-13 # Joules per MeV
fm_to_m = 1.0e-15 # m per fm
conversion_factor = MeV_to_J * np.power(fm_to_m,-3) # TO SI
G_C2_SI =  lal.G_SI / (lal.C_SI * lal.C_SI) # m/kg
G_C4_SI = G_C2_SI /(lal.C_SI * lal.C_SI)  
# -----------------------------------------------------------------------------
solvers = ["Reprimand","LAL"]

## Here we append M-R information to the EoS pressure density posterior samples

### EoS samples
# eos_samples = h5py.File("/home/sunny.ng/TOVSol/MMGP_full_covariance.h5", "r+")
# num_eos = len(eos_samples["eos"])
# eos_to_be_used = np.arange(num_eos)
# ### Enumerating Pressure-Density Relations for iteration
# micro_data = {eos_num : np.array(eos_samples['eos'][eos_id]) for eos_num, eos_id in enumerate(eos_samples['eos'])}

def monotonicity_check(press, edens, bary_dens):
    press_inc = np.diff(press)
    while any(press_inc<=0):
        zero_indices = np.where(press_inc <= 0)[0]
        edens = np.delete(edens,zero_indices)
        bary_dens = np.delete(bary_dens,zero_indices)
        press = np.delete(press,zero_indices)
        press_inc = np.diff(press)
    
    ### Need to re-interpolate over densities to have equally spaced increments for sound speed calculation
    rng = np.geomspace(edens[0], edens[-1], len(edens)) 
    bary_rng = np.geomspace(bary_dens[0], bary_dens[-1], len(bary_dens))
    corr_press = np.interp(rng, edens, press) 
    mono_eos = {"pressurec2":corr_press,"energy_densityc2":rng,"baryon_density":bary_rng}
    
    return mono_eos

def laltov_monotonicity_check(macro_filepath, rho_load, overwrite = True): # because of course LAL needs special treatment... 
    press, edens = np.loadtxt(macro_filepath, unpack = True) # loads two np.arrays, compatible with monotonic check
    bary_dens = np.geomspace(rho_load[0],rho_load[-1], len(rho_load))
    press_inc = np.diff(press)
    while any(press_inc<=0):
        zero_indices = np.where(press_inc <= 0)[0]
        edens = np.delete(edens,zero_indices)
        press = np.delete(press,zero_indices)
        bary_dens = np.delete(bary_dens,zero_indices)
        press_inc = np.diff(press)
    # store corrected EoS information again...
    ### Need to re-interpolate over densities to have equally spaced increments for sound speed calculation
    rng = np.geomspace(edens[0], edens[-1], len(edens)) 
    bary_rng = np.geomspace(bary_dens[0], bary_dens[-1], len(bary_dens))
    corr_press = np.interp(rng, edens, press)
    corr_df = pd.DataFrame(columns = ["pressurec2", "energy_densityc2"]) 
    corr_df["pressurec2"] = corr_press 
    corr_df["energy_densityc2"] = rng
    corr_arr = corr_df.to_records(index = False) # same save file routine
    if overwrite:
        np.savetxt(macro_filepath, corr_arr)
    return bary_rng

### Need to convert everything to geometric units!
def cgs_density_to_geometric(rho):
    return (rho/2.8e14 * .00045)

def press_to_si(pressure, geo = True):
    ### convert g/cm^3 back to MeV/fm^-3, then to SI using above conversions
    si_pressure = pressure / MeV_to_gcm3 * conversion_factor
    if geo:
        si_pressure *= G_C4_SI
    return si_pressure

def edens_to_si(edens, geo = True):
    si_edens = edens / MeV_to_gcm3 * conversion_factor
    if geo:
        si_edens *= G_C4_SI
    return si_edens

def eos_instance(pressure, rho, eps, eps_0):
    """
    This function is mainly meant for use of the RePrimAnd TOV solver.
    Function instantiates an EoS object for the RePrimAnd TOV solver to calculate macroscopic attributes based on. 
    """
    rho_min = rho[1] ### Avoiding floating point error, value mismatch, 
    rho_max = rho[-2] ### currently rho[0] raises a "not within range error" 
    rng = pyr.range(rho_min, rho_max)
    eos = pyr.make_eos_barotr_spline(rho = np.array(rho), press = np.array(pressure),
                                     csnd = np.sqrt(np.array(np.gradient(pressure, eps))),
                                     temp = [], efrac = [], eps_0 = eps_0, n_poly = (3.), ### n_poly = 3 --> adiabatic index of 4/3
                                     rg_rho = rng, ### Need to change units from cgs to SI
                                     units = pyr.units(1.,1.,1.), 
                                     pts_per_mag = 200)
    
    return eos
    
def TOVSolve(pressure, rho, eps, eps_0,
             min_central_dens = rho_nuc_in_cgs, # min and max central densities to search with, using RePrimAnd
             max_central_dens = 15*rho_nuc_in_cgs,
             solver = "Reprimand",
             eos_tov_file_dir = "/home/sunny.ng/semiparameos/tov_results",
             macro_idx = 0,
             truncate_at_mtov = False,
             lal_solver_style = "ODE"):
    
    """
    The main executable for TOV solving, based on a supplied EoS (in CGS units).
    This function contains two possible solving routines, based on solvers given by,
    (1) RePrimAnd
    (2) LAL(Simulation). 
    Under the assumption each EoS is supplied in CGS* units, TOVSolve automatically handles
    unit conversions to cooperate with each type of solver.
    """
    
    
    if solver == "Reprimand":
        ### returns an eos_barotrop object (expects geometric units)
        eos = eos_instance(cgs_density_to_geometric(pressure),
                           cgs_density_to_geometric(rho),
                           cgs_density_to_geometric(eps),
                           cgs_density_to_geometric(eps_0))
        
        ### find central density corresponding to maximum mass (in geometric units) based on supplied EoS
        max_central_density = pyr.find_rhoc_tov_max_mass(eos, rhobr0 = cgs_density_to_geometric(min_central_dens),  ### arbitrary minimum density
                                   rhobr1 = cgs_density_to_geometric(max_central_dens), ### arbitrary max density for large enough finding range
                                   nbits = 28, acc = 1e-8, max_steps = 30) ### numerical defaults given by RePrimAnd

        ### Create a density range from arbitary minimum to first* maximum mass central density
        central_densities = np.linspace(cgs_density_to_geometric(min_central_dens), max_central_density, 100)

        ### TOV Solving
        tidal = np.array([pyr.get_tov_properties(eos, rhoc).deformability.lambda_tidal for rhoc in central_densities])
        masses = np.array([pyr.get_tov_properties(eos, rhoc).grav_mass for rhoc in central_densities])
        ### 1.477 is for converting radii from geometric units to km
        radii = [((pyr.get_tov_properties(eos, rhoc).circ_radius)*1.477) for rhoc in central_densities] 

        ### Now make a structured array with tidal and masses
        tidal_masses = np.zeros((len(central_densities)),
                                 dtype={'names':("Lambda", "M", "R"), ### name of the groups in the structured array
                              'formats':("f8", "f8", "f8")}) ### assumed all floats
        
        tidal_masses["M"] = masses
        tidal_masses["R"] = radii
        tidal_masses["Lambda"] = tidal

        if truncate_at_mtov:
            mtov_idx = np.argmax(tidal_masses["M"])
            tidal_masses = tidal_masses[:mtov_idx]
        
        return tidal_masses
   
    elif solver == "LAL":
        
        ### Need to write eos's to txt file for lal solver to read... I guess. 
        outpath = eos_tov_file_dir
        
        if not os.path.isdir(os.path.join(os.getcwd(),outpath)):
            os.makedirs(outpath)
        
        # check if macro file already exists
        macro_file = f"tov_draw_{macro_idx:06d}.txt" 
        #file_exist = os.path.isfile(os.path.join(eos_tov_file_dir, macro_file))
        
        #if file_exist:
        #    pass
        #elif not file_exist:    
        macro_df = pd.DataFrame(columns = ["pressurec2","energy_densityc2"])
        macro_df["pressurec2"] = press_to_si(pressure) ### converted pressure to SI (and then geometrized...)
        macro_df["energy_densityc2"] = edens_to_si(eps) ### converted energy density to SI (and then geometrized...)
        macro_arr = macro_df.to_records(index = False)
        np.savetxt(f"{eos_tov_file_dir}/{macro_file}", macro_arr)
        del macro_df # clear the dataframe
        
        solve_rhoc = rho
        
        # TOV routine
        try:
            eos = lalsim.SimNeutronStarEOSFromFile(f"{eos_tov_file_dir}/{macro_file}")
        except: # implement monotonicity check again...
            print("Attempting montonic correction for LAL EoS read in...")
            try:
                solve_rhoc = laltov_monotonicity_check(f"{eos_tov_file_dir}/{macro_file}", rho) # make eos monotonic
                eos = lalsim.SimNeutronStarEOSFromFile(f"{eos_tov_file_dir}/{macro_file}")
            except:
                raise ValueError(f"EoS {macro_idx} breaks TOV.")
            
        mass_rad_tide = [[],[],[]]
        
        if lal_solver_style == "Family":
            fam =  lalsim.CreateSimNeutronStarFamily(eos)
            min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)
            mass_increment = 0.01 ### arbitrary increment in M(R) calculation
            m = max(min_mass, 0.01*lal.MSUN_SI) # check for lowest supported mass, and take greater of the two
            mass_1 = m/lal.MSUN_SI

            while m < max_mass:
                radius = lalsim.SimNeutronStarRadius(m, fam)
                love_2 = lalsim.SimNeutronStarLoveNumberK2(m, fam)
                tidal = (2/3) * love_2 * (((radius/m)*(lal.C_SI**2/lal.G_SI))**5)
                mass_1 = m/lal.MSUN_SI # converts m to units of msun

                ### store macro data
                mass_rad_tide[0].append(mass_1)
                mass_rad_tide[1].append(radius/1e3) # converts to km
                mass_rad_tide[2].append(tidal)
                m += mass_increment * lal.MSUN_SI
            
            ### Create structured array to store macro data
            tidal_masses = np.zeros(len(mass_rad_tide[0]),
                                     dtype={'names':("Lambda", "M", "R", "rhoc"), ### name of the groups in the structured array
                                     'formats':("f8", "f8", "f8", "f8")})

            tidal_masses["M"] = (mass_rad_tide[0])
            tidal_masses["R"] = (mass_rad_tide[1])
            tidal_masses["Lambda"] = (mass_rad_tide[2])
            tidal_masses["rhoc"] = solve_rhoc
            
            if truncate_at_mtov:
                mtov_idx = np.argmax(tidal_masses["M"])
                tidal_masses = tidal_masses[:mtov_idx]
            
            del eos
            return tidal_masses
        
        elif lal_solver_style == "ODE":
        
            # Iterate through central pressures given by EoS values
            # at this point in the solving routine, EoS values should already be stored in a .txt file
            macro_pressure, macro_eps = np.loadtxt(os.path.join(eos_tov_file_dir, macro_file), unpack = True) # units should already be converted
            
            # Only solve for rhoc greater than half saturation density 
            #macro_pressure = macro_pressure[rho > 0.05*rho_nuc_in_cgs]
            #macro_eps = macro_eps[rho > 0.05*rho_nuc_in_cgs]
            
            for p in range(len(macro_pressure)):
                ### TOVODEIntegrate result [0,1,2] --> radius[m], mass [kg?], love_2 respectively
                macro_obs = lalsim.SimNeutronStarTOVODEIntegrate(macro_pressure[p]/G_C4_SI, eos) 
                
                mass_rad_tide[0].append(macro_obs[1] / lal.MSUN_SI) # store mass values
                mass_rad_tide[1].append(macro_obs[0] / 1e3) # store radius values
                
                # calculate tidal deformability
                love_2 = macro_obs[2] 
                tidal = (2/3) * love_2 * (((macro_obs[0]/macro_obs[1])*(lal.C_SI**2/lal.G_SI))**5)
                mass_rad_tide[2].append(tidal)
            
            ### Create structured array to store macro data
            tidal_masses = np.zeros(len(mass_rad_tide[0]),
                                     dtype={'names':("Lambda", "M", "R", "rhoc"), ### name of the groups in the structured array
                                     'formats':("f8", "f8", "f8", "f8")})
                           
            tidal_masses["M"] = (mass_rad_tide[0])
            tidal_masses["R"] = (mass_rad_tide[1])
            tidal_masses["Lambda"] = (mass_rad_tide[2])
            tidal_masses["rhoc"] = solve_rhoc
            
            if truncate_at_mtov:
                mtov_idx = np.argmax(tidal_masses["M"])
                tidal_masses = tidal_masses[:mtov_idx]
            
            del eos
            return tidal_masses
    else:
        raise KeyError(f"Not a valid solver. Solvers available: {solvers}")
        
    
def get_seq(eos, acc = pyr.star_acc_simple()):
    tov_branch = pyr.make_tov_branch_stable(eos, acc)
    return tov_branch

def restart_samples(loaded_samples, close = False):
    if "ns" in loaded_samples.keys():
        del loaded_samples["ns"]
        if close:
            loaded_samples.close()
    else:
        raise KeyError("ns group not found within EoS sample file.")
    return

if __name__ == "__main__":
    
    ### EoS samples ##########################################################################
    GP_result_file = "MM_exp"
    eos_samples_file_path = f"/home/sunny.ng/semiparameos/generated_eoss/{GP_result_file}.h5" 
    ##########################################################################################

    eos_samples = h5py.File(f"{eos_samples_file_path}", "r+")
    tov_result_folder_name = os.path.splitext(os.path.split(eos_samples_file_path)[1])[0]
    
    ### Create range for iterating through
    num_eos = len(eos_samples["eos"])
    eos_to_be_used = np.arange(num_eos)
    
    ### Enumerating Pressure-Density Relations for iteration
    micro_data = {eos_num : np.array(eos_samples['eos'][eos_id]) for eos_num, eos_id in enumerate(eos_samples['eos'])}
    
    ### TOV SOLVER CHOICE #################################
    tov_solver_choice = "LAL"
    #######################################################
    
    ### Delete "ns" group if it already exists to restart TOV routine. Mainly to address partially created TOV sets due to errors occuring. 
    restart_tov_generation = False
    if restart_tov_generation:
        restart_samples(eos_samples)
    
    try:
        ns = eos_samples.create_group("ns")
    except:
        raise KeyError("ns group already exists.")
    
    for eqn in tqdm.tqdm(range(len(eos_to_be_used))):
        
        try:
            macro_draw = TOVSolve(pressure = micro_data[eqn]["pressurec2"],
                                 rho = micro_data[eqn]["baryon_density"],
                                 eps = micro_data[eqn]["energy_densityc2"],
                                 eps_0 = micro_data[eqn]["energy_densityc2"][0],
                                 macro_idx = eqn,
                                 solver = tov_solver_choice,
                                 lal_solver_style = "ODE",
                                 truncate_at_mtov = True,
                                 eos_tov_file_dir = f"/home/sunny.ng/semiparameos/tov_results/{tov_result_folder_name}") ### need to change the home directory
            ns[f"eos_{eqn:06d}"] = macro_draw   
            print(f"Macro draw: eos_{eqn:06d} - generated.")
        except:
            
            try: 
                print(f"TOV Solving for EoS {eqn} did not work. Attempting correction...")
                corr_eos = monotonicity_check(micro_data[eqn]["pressurec2"],
                                              micro_data[eqn]["energy_densityc2"],
                                              micro_data[eqn]["baryon_density"])
                macro_draw = TOVSolve(pressure = corr_eos["pressurec2"],
                                     rho = corr_eos["baryon_density"],
                                     eps = corr_eos["energy_densityc2"],
                                     eps_0 = corr_eos["energy_densityc2"][0],
                                     macro_idx = eqn,
                                     solver = tov_solver_choice,
                                     truncate_at_mtov = True,
                                     lal_solver_style = "ODE",
                                     eos_tov_file_dir = f"/home/sunny.ng/semiparameos/tov_results/{tov_result_folder_name}")

                ns[f"eos_{eqn:06d}"] = macro_draw
                print(f"Macro draw: eos_{eqn:06d} - generated.")
                continue
            
            except:
                raise ValueError("TOV solving broken.")
                eos_samples.close()
    
    eos_samples.close()
    
    
