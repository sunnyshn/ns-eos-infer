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

def find_pts(data, idx, threshold = 0.0):
    """
    Function to find phase transitions within a given EoS set. 
    Threshold acts as minimum difference in the sound speed to qualify as a phase transition. 
    """
    cs = np.gradient(data[idx]["pressurec2"], data[idx]["energy_densityc2"])
    num_neg = np.where(np.diff(cs[data[idx]["baryon_density"] > rho_nuc_in_cgs]) < -threshold)[0]
    num_pts = len(np.where(np.diff(num_neg) > 1.0)[0])
    return num_pts

def plot_eoss(data_with_key, x_key, y_key, alpha, clr,
              axis = None, loglog = False,
              semilogx = False, semilogy = False):
    if semilogx:
        plt.semilogx()
    if semilogy:
        plt.semilogy()
    if loglog:
        plt.loglog()
    if axis:
        plt.axis([axis[0], axis[1], axis[2], axis[3]])
        
    plt.plot(data_with_key[f"{x_key}"], data_with_key[f"{y_key}"], alpha = alpha, color = clr)
    plt.gcf().set_dpi(500)
    plt.grid(alpha = 0.2)
    plt.minorticks_on()
    plt.show()
    
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

def mm_eos_to_cgs(mm_eos):
    mm_eos_cgs = {"baryon_density": np.array(mm_eos["number_density"]) * mass_of_nucleon/fm_in_cgs**3,
                  "energy_densityc2":np.array(mm_eos["energy_densityc2"]),
                  "pressurec2" :np.array(mm_eos["pressure_nuclear"]) / nucleon_mass_in_MeV * mass_of_nucleon / fm_in_cgs**3}
    return mm_eos_cgs

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

    except ValueError:
        print("All EoS's are unlikely.")
        
def get_pe_quantiles(eos_set, eos_data, interp_densities, quantiles, cgs_press = True, get_median = False):
    pdvals = []
    # Obtain quantiles for the prior set of EoS's
    for eos in eos_set:
        eos_densities = eos_data[eos]["baryon_density"]
        eos_pressures = eos_data[eos]["pressurec2"]
        if cgs_press:
            eos_pressures = eos_data[eos]["pressurec2"]*gcm3_to_dynecm2
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

def get_cs_quantiles(eos_set, eos_data, interp_densities, quantiles, get_median = False):
    csvals = []
    # Obtain sound speed quantiles for a set of EoS's
    for eos in eos_set:
        eos_rest_mass = eos_data[eos]["baryon_density"]
        eos_densities = eos_data[eos]["energy_densityc2"]
        eos_pressures = eos_data[eos]["pressurec2"]
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

def get_mr_quantiles(eos_set, eos_mac_data, interp_masses, quantiles, cutoff = 30., verbose =  False, get_median = False):
    mrvals = []
    num_eos_cut = 0
    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"][eos_mac_data[eos]["R"] < cutoff] # cuts to ignore radii values at
        eos_radii = eos_mac_data[eos]["R"][eos_mac_data[eos]["R"] < cutoff] # larger mass from white dwarf branch 
        if len(eos_radii) == 0:
            num_eos_cut += 1
            continue
        ### Try interpolating with scipy function
        massrad_func = interpolate.interp1d(eos_mass, eos_radii, bounds_error = False, fill_value = "NaN")
        interp_rad = massrad_func(interp_masses)
        mrvals.append(interp_rad)
    mrvals = np.array(mrvals)
    
    ## Obtaining quantiles 
    mr_sigmas = np.zeros((len(interp_masses),len(quantiles)))
    if get_median:
        mr_median = np.zeros((len(interp_masses), 1))

    ### Expecting NaN's from Eos's that don't reach end mass from interpolation range
    for i in range(len(interp_masses)):
        mr_sigmas[i]=np.nanpercentile(mrvals[:,i],quantiles)
        if get_median:
            mr_median[i] = np.nanmedian(mrvals[:,i])

    if verbose:
        print("Number of surviving EoS after radius cuts: ", str(len(eos_set) - num_eos_cut))
    
    del mrvals
    if get_median:
        return mr_sigmas, mr_median
    else:
        return mr_sigmas

def get_lambda_quantiles(eos_set, eos_mac_data, interp_masses, quantiles, cutoff = 30., verbose=False, get_median = False):
    lambda_vals = []
    num_eos_cut = 0
    
    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"][eos_mac_data[eos]["R"] < cutoff]
        eos_lambda = eos_mac_data[eos]["Lambda"][eos_mac_data[eos]["R"] < cutoff]
        if len(eos_lambda) == 0:
            num_eos_cut += 1
            continue

        ### Try interpolating with scipy function
        mass_lambda_func = interpolate.interp1d(eos_mass, eos_lambda, bounds_error=False, fill_value= "NaN")
        interp_lambda = mass_lambda_func(interp_masses)
        lambda_vals.append(interp_lambda)

    lambda_vals = np.array(lambda_vals)
    
    ## Obtaining quantiles
    lambda_sigmas = np.zeros((len(interp_masses), len(quantiles)))
    if get_median:
        lambda_median = np.zeros((len(interp_masses), 1))
    for i in range(len(interp_masses)):
        lambda_sigmas[i] = np.nanpercentile(lambda_vals[:,i], quantiles)
        if get_median:
            lambda_median[i] = np.nanmedian(lambda_vals[:,i])
    
    if verbose:
        print("Number of surviving EoS after radius cuts: ", str(len(eos_set) - num_eos_cut))

    del lambda_vals
    if get_median:
        return lambda_sigmas, lambda_median
    else:
        return lambda_sigmas    

def get_max_mass_quantiles(eos_set, eos_mac_data, quantiles, cutoff = 3.0):
    max_masses = []
    num_eos_cut = 0

    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"]
        if len(eos_mass) == 0:
            continue
        max_mass = np.nanmax(eos_mass)
        max_masses.append(max_mass)

    max_masses = np.array(max_masses)
    mass_quantiles = np.nanpercentile(max_masses, quantiles)
    
    return mass_quantiles

def diagnose_eos_set(eos_set, weights = None, astro_key = None, percentiles = [5, 95], verbose = True):
    
    """
    Intakes an EoS set in h5 format, and calculates the fiducial pressure, sound speed, radii and tidal
    deformability at 1.4, 2.0, and M_tov in M_sun. 
    """
    
    eos_to_be_used = np.arange(len(eos_set["eos"]))
    percentiles_median = [50,95]
    
    if weights is not None:
        if verbose:
            print("Sampling EoS according to weights...")
        weighted_eos = get_and_load_weights(weights, astro_key, eos_to_be_used)
        eos_to_be_used = weighted_eos
        max_plot_mass = np.max(weights["Mmax"])
    
    # Interpolation range
    plot_densities = np.geomspace(np.power(10.0,13.0), 10.*rho_nuc_in_cgs, 5000)
    plot_masses = np.linspace(0.5, 4.0, 5000)
    
    # Enumerate EoS's
    micro_data = {eos_num : np.array(eos_set['eos'][eos_id]) for eos_num, eos_id in enumerate(eos_set['eos'])}
    macro_data = {eos_num : np.array(eos_set['ns'][eos_id]) for eos_num, eos_id in enumerate(eos_set['ns'])}
    # Calculate Pressure quantiles + fiducial values
    p_sigmas, p_median = get_pe_quantiles(eos_to_be_used, micro_data, plot_densities, percentiles, get_median = True)
    press_int_func_lower = interpolate.interp1d(plot_densities, p_sigmas[:,0], kind = 5)
    press_int_func_upper = interpolate.interp1d(plot_densities, p_sigmas[:,1], kind = 5)
    press_int_func_median = interpolate.interp1d(plot_densities, p_median, axis = 0, kind = 5)
    p_med_at2sat = press_int_func_median(2*rho_nuc_in_cgs) # Pressure at 2x saturation density
    p_med_at6sat = press_int_func_median(6*rho_nuc_in_cgs) # Pressure at 6x saturation density
    p_lower2sat = press_int_func_lower(2*rho_nuc_in_cgs) 
    p_upper2sat = press_int_func_upper(2*rho_nuc_in_cgs) 
    p_lower6sat = press_int_func_lower(6*rho_nuc_in_cgs)
    p_upper6sat = press_int_func_upper(6*rho_nuc_in_cgs)

    # Calculate Mass-Radius quantiles + canonical radii
    mr_sigmas, mr_median = get_mr_quantiles(eos_to_be_used, macro_data, plot_masses, percentiles,
                                            cutoff = 30., verbose =  False, get_median = True)
    mr_int_func_median = interpolate.interp1d(plot_masses, mr_median, axis = 0, kind = 5)
    mr_int_func_lower = interpolate.interp1d(plot_masses, mr_sigmas[:,0], kind = 5)
    mr_int_func_upper = interpolate.interp1d(plot_masses, mr_sigmas[:,1], kind = 5)
    median_rad_canon = mr_int_func_median(1.4)
    median_rad_2 = mr_int_func_median(2.)
    lower_rad_canon = mr_int_func_lower(1.4)
    upper_rad_canon = mr_int_func_upper(1.4)
    lower_rad_2 = mr_int_func_lower(2.)
    upper_rad_2 = mr_int_func_upper(2.)

    # Calculate Sound Speed quantiles + fiducial sound speed
    cs_sigmas, cs_median = get_cs_quantiles(eos_to_be_used, micro_data, plot_densities, percentiles, get_median = True)
    cs_int_func_lower = interpolate.interp1d(plot_densities, cs_sigmas[:,0], kind = 5)
    cs_int_func_upper = interpolate.interp1d(plot_densities, cs_sigmas[:,1], kind = 5)
    cs_int_func_median = interpolate.interp1d(plot_densities, cs_median, axis = 0, kind = 5)
    cs_at2sat = cs_int_func_median(2*rho_nuc_in_cgs) 
    cs_at6sat = cs_int_func_median(6*rho_nuc_in_cgs) 
    cs_lower2sat = cs_int_func_lower(2*rho_nuc_in_cgs) 
    cs_upper2sat = cs_int_func_upper(2*rho_nuc_in_cgs) 
    cs_lower6sat = cs_int_func_lower(6*rho_nuc_in_cgs)
    cs_upper6sat = cs_int_func_upper(6*rho_nuc_in_cgs)

    # Calculate Lambda quantiles
    lamb_sigmas, lamb_median = get_lambda_quantiles(eos_to_be_used, macro_data, plot_masses, percentiles,
                                       cutoff = 30., verbose =  False, get_median = True)
    lamb_int_func_lower = interpolate.interp1d(plot_masses, lamb_sigmas[:,0], kind = 5)
    lamb_int_func_upper = interpolate.interp1d(plot_masses, lamb_sigmas[:,1], kind = 5)
    lamb_int_func_median = interpolate.interp1d(plot_masses, lamb_median, axis = 0, kind = 5)
    median_lamb = lamb_int_func_median(1.4)
    lower_lamb = lamb_int_func_lower(1.4)
    upper_lamb = lamb_int_func_upper(1.4)

    # Get (median) max mass
    med_max_mass = get_max_mass_quantiles(eos_to_be_used, macro_data, [50], cutoff = 30.)
    max_mass_sigmas = get_max_mass_quantiles(eos_to_be_used, macro_data, [5,95], cutoff = 30.)
    
    # Check radius at max mass
    # median_rad_tov = mr_int_func_median(med_max_mass)
    # lower_rad_tov = mr_int_func_lower(med_max_mass)
    # upper_rad_tov = mr_int_func_upper(med_max_mass)

    
    if verbose:
        print(f"For the set {eos_set}: \n" 
                f"Median pressure at 2x nuclear saturation density: {p_med_at2sat} \n"
                f"Lower & upper quartile pressure at 2x nuclear saturation density: {p_lower2sat}, {p_upper2sat} \n"
                f"Median pressure at 6x nuclear saturation density: {p_med_at6sat} \n"
                f"Lower & upper quartile pressure at 6x nuclear saturation density: {p_lower6sat}, {p_upper6sat} \n"
                f"Median radius at 1.4 solar masses: {median_rad_canon} \n"
                f"Lower & upper quartile radius at 1.4 solar masses: {lower_rad_canon}, {upper_rad_canon} \n"
                f"Median radius at 2.0 solar masses: {median_rad_2} \n"
                f"Lower & upper quartile radius at 2.0 solar masses: {lower_rad_2}, {upper_rad_2} \n"
                f"Median max mass: {med_max_mass} \n"
                f"Lower and Upper bound on max mass: {max_mass_sigmas} \n"
                # f"Median radius at median max mass: {median_rad_tov} \n"
                # f"Lower & upper quartile radius at median max mass: {lower_rad_tov}, {upper_rad_tov} \n"
                f"Median sound speed at 2x nuclear saturation density: {cs_at2sat} \n"
                f"Lower & upper quartile sound speed at 2x nuclear saturation density: {cs_lower2sat}, {cs_upper2sat} \n"
                f"Median sound speed at 6x nuclear saturation density: {cs_at6sat} \n"
                f"Lower & upper quartile sound speed at 6x nuclear saturation density: {cs_lower6sat}, {cs_upper6sat} \n"
                f"Median lambda at 1.4 solar masses: {median_lamb} \n"
                f"Lower & upper quartile lambda at 1.4 solar masses: {lower_lamb}, {upper_lamb}")
    return {"pressure_2sat": p_med_at2sat, "p_lower_2sat": p_lower2sat, "p_upper_2sat": p_upper2sat,
            "pressure_6sat": p_med_at6sat, "p_lower_6sat": p_lower6sat, "p_upper_6sat": p_upper6sat,
            "radius_1.4":  median_rad_canon, "radius_lower_1.4":lower_rad_canon, "radius_upper_1.4": upper_rad_canon,
            "radius_2.0":  median_rad_2, "radius_lower_2.0":lower_rad_2, "radius_upper_2": upper_rad_2,
            "median_max_mass": med_max_mass,
            # "radius_at_tov": median_rad_tov, "radius_lower_tov": lower_rad_tov, "radius_upper_tov": upper_rad_tov,
            "cs_2sat": cs_at2sat, "cs_lower_2sat": cs_lower2sat, "cs_upper_2sat": cs_upper2sat,
            "cs_6sat": cs_at6sat, "cs_lower_6sat": cs_lower6sat, "cs_upper_6sat": cs_upper6sat,
            "lambda": median_lamb, "lambda_lower": lower_lamb, "lambda_upper": upper_lamb}


def get_max_mass(eos_set, eos_mac_data, cutoff = 30.0):
    max_masses = []
    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"][eos_mac_data[eos]["R"] < cutoff]
        if len(eos_mass) == 0:
            max_masses.append(np.nan)  # preserve alignment
        else:
            max_mass = np.nanmax(eos_mass)
            max_masses.append(max_mass)

    max_masses = np.array(max_masses)
    return max_masses

def get_each_radius(eos_set, eos_mac_data, mass_target, cutoff = 30.0):
    radii_at_target_mass = []
    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"][eos_mac_data[eos]["R"] < cutoff]
        eos_radii = eos_mac_data[eos]["R"][eos_mac_data[eos]["R"] < cutoff]

        if len(eos_radii) == 0:
            radii_at_target_mass.append(np.nan)
            continue

        massrad_func = interpolate.interp1d(eos_mass, eos_radii, bounds_error=False, fill_value=np.nan)
        interp_radius = massrad_func(mass_target)
        radii_at_target_mass.append(interp_radius)

    radii_at_target_mass = np.array(radii_at_target_mass)
    return radii_at_target_mass

def load_eos(h5_samples, load_micro = False, load_macro = True, return_all = False, get_macro = True, get_micro = False):
    samples = h5py.File(h5_samples)
    eos_to_be_used = np.arange(len(samples["eos"]))
    if load_micro:
        micro_data = {eos_num : np.array(samples['eos'][eos_id]) for eos_num, eos_id in enumerate(samples['eos'])}
        if get_micro:
            return micro_data
    if load_macro:
        macro_data = {eos_num : np.array(samples['ns'][eos_id]) for eos_num, eos_id in enumerate(samples['ns'])}
        if get_macro:
            return macro_data

    if return_all:
        return micro_data, macro_data, eos_to_be_used

def plot_corner_mmax_radius(control_eos_macro, legred_eos_macro, weights, plot_kdes = False):
    macro_control = control_eos_macro
    macro_leg = legred_eos_macro
    
    # eos to use
    indices = np.arange(len(weights))
    eos_to_be_used_leg = np.arange(len(legred_eos_macro))

    #liklihood weights
    likelihood_subset = get_and_load_weights(weights, 'joint_logweight', indices, Neff = False)
    
    #control values
    max_mass_control = get_max_mass(likelihood_subset, macro_control)
    radii_control = get_each_radius(likelihood_subset, macro_control, 1.4, cutoff = 18)
    
    # legred values
    max_mass_leg = get_max_mass(eos_to_be_used_leg, macro_leg)
    radii_leg = get_each_radius(eos_to_be_used_leg, macro_leg, 1.4, cutoff = 18.0)
    
    semiparam_data = np.vstack((max_mass_control, radii_control,)).T
    legred_data = np.vstack((max_mass_leg, radii_leg,)).T
    
    # remove nans?
    semiparam_data = semiparam_data[~np.isnan(semiparam_data).any(axis=1)]
    legred_data = legred_data[~np.isnan(legred_data).any(axis=1)]
    
    mrlim = [(1.8,3.8),(8.,18.)]
    
    figure = corner.corner(
        semiparam_data,
        color="#5A91DD",
        plot_datapoints=False,
        fill_contours=False,
        plot_contours=True,
        plot_density = True,
        labels=[r"$M_{\rm max}$ [$M_{\odot}$]", r"$R_{1.4}$ [km]"],
        hist_kwargs={"density": True},
        range = mrlim,
        smooth= 1.0,
    )
    
    corner.corner(
        legred_data,
        fig=figure,
        color="#d38718",
        plot_datapoints=False,
        fill_contours=False,
        plot_contours=True,
        plot_density = True,
        hist_kwargs={"density": True},
        range = mrlim,
        smooth= 1.0,
    )
            
    blue_patch = mpatches.Patch(color="#5A91DD", label=r"MM+$\chi$+GP")
    red_patch = mpatches.Patch(color="#d38718", label="GP (Legred et al.)")
    
    figure.legend(
        handles=[blue_patch, red_patch],
        loc="upper right",
        bbox_to_anchor=(0.9, 0.85),
        title = "PSR Informed",
        title_fontsize = 8.0,
        fontsize=8.0,
    )
        
    # Get the axes on the diagonal
    axes = np.array(figure.axes).reshape((2, 2))

    # Replace the top-left diagonal with a custom PDF for variable x
    ref_ax = axes[-1,0]
    upl_ax = axes[0, 0]
    botr_ax = axes[-1,-1]
    upl_ax.clear()
    botr_ax.clear()

    # upl_ax.get_shared_x_axes().join(upl_ax, axes[1,0])
   
    # Draw PDF manually
    mass_vals = np.linspace(mrlim[0][0],mrlim[0][1], 5000)
    l_mass_vals = np.linspace(mrlim[0][0],mrlim[0][1], 5000)
    rad_vals = np.linspace(mrlim[1][0],mrlim[1][1], 5000)
    l_rad_vals = np.linspace(mrlim[1][0],mrlim[1][1], 5000)
    semi_m_kde = gaussian_kde(semiparam_data[:,0])
    leg_m_kde = gaussian_kde(legred_data[:,0])
    semi_r_kde = gaussian_kde(semiparam_data[:,1])
    leg_r_kde = gaussian_kde(legred_data[:,1])
    upl_ax.plot(mass_vals, semi_m_kde(mass_vals), color="#5A91DD")
    upl_ax.plot(mass_vals, leg_m_kde(mass_vals), color="#d38718")
    upl_ax.set_xlim(mrlim[0])
    upl_ax.set_ylim(bottom=0.)
    upl_ax.set_yticks([])
    upl_ax.set_ylabel("")
    upl_ax.xaxis.set_major_locator(ref_ax.xaxis.get_major_locator())
    upl_ax.xaxis.set_major_formatter(ref_ax.xaxis.get_major_formatter())
    
    botr_ax.plot(rad_vals, semi_r_kde(rad_vals), color="#5A91DD")
    botr_ax.plot(l_rad_vals, leg_r_kde(rad_vals), color="#d38718")
    botr_ax.set_xlim(mrlim[1])
    botr_ax.set_yticks([])
    botr_ax.set_ylabel("")
    botr_ax.set_ylim(bottom=0.)
    botr_ax.xaxis.set_major_locator(ref_ax.yaxis.get_major_locator())
    botr_ax.xaxis.set_major_formatter(ref_ax.yaxis.get_major_formatter())
    
    
    for ax in figure.get_axes():
        ax.minorticks_on()
        ax.grid(alpha = .2)
    
    upl_ax.set_autoscale_on(False)
    botr_ax.set_autoscale_on(False)
    plt.gcf().set_dpi(500)
    figure.show()

    
def load_ppf_micro_eos(mm_eos_dir, psr_weights):
    eos_idxs = np.array(pd.read_csv(psr_weights)["eos_index"])
    mm_eoss = {}
    for idx in eos_idxs:
        mm_eos_load = pd.read_csv(f"{mm_eos_dir}/eos_mmpoly{idx}_clean.out",
                                  names=["number_density", "energy_densityc2","pressure_nuclear"], sep = r"\s+", skiprows = 1)
        mm_eoss[idx] = mm_eos_to_cgs(mm_eos_load)
    return mm_eoss

def load_ppf_macro_eos(mm_tov_dir, psr_weights):
    eos_idxs = np.array(pd.read_csv(psr_weights)["eos_index"])
    ppf_macro = {}
    for idx in eos_idxs:
        R, M, tidal = np.loadtxt(f"{mm_tov_dir}/tov{idx}.out", usecols = (2,3,8), unpack = True)
        mm_macros = np.array((M, R, tidal), dtype = [('M', 'f8'), ('R', 'f8'), ('tidal', 'f8')])
        ppf_macro[idx] = mm_macros
    
    return ppf_macro

def load_micro_data(h5_samples):
    mic_data = {eos_num : np.array(h5_samples['eos'][eos_id]) for eos_num, eos_id in enumerate(h5_samples['eos'])}
    return mic_data

def load_macro_data(h5_samples):
    mac_data = {eos_num : np.array(h5_samples['ns'][eos_id]) for eos_num, eos_id in enumerate(h5_samples['ns'])}
    return mac_data

def parse_eos(h5samples, weighted_eos, new_filename):
    idx = 0
    data = h5py.File(new_filename, "w")
    data.create_group("eos")
    data.create_group("ns")
    for i in weighted_eos:
        data["eos"][f"eos_{idx:06d}"] = h5samples["eos"][f"eos_{i:06d}"][()]
        data["ns"][f"eos_{idx:06d}"] = h5samples["ns"][f"eos_{i:06d}"][()]
        idx += 1
    data.create_dataset("id", data=np.arange(idx))
    data.close()
    return 

