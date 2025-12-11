import numpy as np
from scipy.interpolate import interp1d

def nm_to_ev(wl_nm):
    return 1239.84193 / wl_nm

def ev_to_nm(ev):
    return 1239.84193 / ev

def estimate_bandgap(wavelength_nm, k):
    """
    Estimates the direct bandgap from Tauc plot ((alpha*E)^2 vs E).
    This is a rough estimation finding the x-intercept of the steepest slope.
    """
    E = nm_to_ev(wavelength_nm)
    alpha = 4 * np.pi * k / (wavelength_nm * 1e-7) # cm^-1, wl in nm -> cm is 1e-7
    # Actually unit doesn't matter for intercept, just shape
    
    # Tauc y-axis for direct gap
    y = (alpha * E)**2
    
    # Find max slope region (inflection point of first derivative)
    # Filter for region where absorption is significant but not saturated
    # Heuristic: look at region where y is between 10% and 90% of max
    mask = (y > 0.1 * np.max(y)) & (y < 0.9 * np.max(y))
    
    if np.sum(mask) < 5:
        # Fallback if mask is too small
        mask = y > 0.05 * np.max(y)
        
    E_region = E[mask]
    y_region = y[mask]
    
    if len(E_region) < 2:
        return 1.5 # Default fallback
        
    # Numerical derivative
    dy_de = np.gradient(y_region, E_region)
    
    # Find point of max slope
    max_slope_idx = np.argmax(dy_de)
    slope = dy_de[max_slope_idx]
    E0 = E_region[max_slope_idx]
    y0 = y_region[max_slope_idx]
    
    # Linear equation: y - y0 = slope * (E - E0)
    # x-intercept (y=0): -y0 = slope * (E_g - E0) => E_g = E0 - y0/slope
    Eg = E0 - y0 / slope
    
    return max(0.1, Eg) # Ensure positive

def interpolate_material_properties(
    target_wavelengths,
    target_bandgap_ev,
    ref1_data,
    ref1_bandgap_ev,
    ref2_data=None,
    ref2_bandgap_ev=None,
    mix_fraction=0.5
):
    """
    Approximates n, k for a target bandgap using one or two reference materials.
    
    Args:
        target_wavelengths (array): Wavelengths in nm to generate data for.
        target_bandgap_ev (float): Desired bandgap in eV.
        ref1_data (dict): {'wavelength_nm': array, 'n': array, 'k': array}
        ref1_bandgap_ev (float): Bandgap of reference 1.
        ref2_data (dict, optional): Second reference material data.
        ref2_bandgap_ev (float, optional): Bandgap of reference 2.
        mix_fraction (float): Weight for ref1 (0.0 to 1.0). If 1.0, only ref1 is used.
                              If ref2 is provided, n_mix = alpha * n1 + (1-alpha) * n2.
                              
    Returns:
        tuple: (n_pred, k_pred) arrays corresponding to target_wavelengths.
    """
    
    # 1. Convert Target Wavelengths to Energy
    E_target = nm_to_ev(target_wavelengths)
    
    # 2. Calculate Reduced Energy for Target
    # x = E / Eg
    x_target = E_target / target_bandgap_ev
    
    # 3. Process Reference 1
    E_ref1 = nm_to_ev(ref1_data['wavelength_nm'])
    x_ref1 = E_ref1 / ref1_bandgap_ev
    
    # Create interpolators for Ref 1 in reduced energy space
    # Sort by x (Energy decreases with Wavelength, so x increases as Wavelength decreases)
    # We need x to be strictly increasing for interp1d usually, or at least sorted
    sort_idx1 = np.argsort(x_ref1)
    n1_interp = interp1d(x_ref1[sort_idx1], ref1_data['n'][sort_idx1], kind='linear', bounds_error=False, fill_value="extrapolate")
    k1_interp = interp1d(x_ref1[sort_idx1], ref1_data['k'][sort_idx1], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    n_pred = n1_interp(x_target)
    k_pred = k1_interp(x_target)
    
    # 4. Process Reference 2 if available
    if ref2_data is not None and ref2_bandgap_ev is not None:
        E_ref2 = nm_to_ev(ref2_data['wavelength_nm'])
        x_ref2 = E_ref2 / ref2_bandgap_ev
        
        sort_idx2 = np.argsort(x_ref2)
        n2_interp = interp1d(x_ref2[sort_idx2], ref2_data['n'][sort_idx2], kind='linear', bounds_error=False, fill_value="extrapolate")
        k2_interp = interp1d(x_ref2[sort_idx2], ref2_data['k'][sort_idx2], kind='linear', bounds_error=False, fill_value="extrapolate")
        
        n2_pred = n2_interp(x_target)
        k2_pred = k2_interp(x_target)
        
        # Weighted Average
        # mix_fraction is weight for Ref 1
        # Ensure mix_fraction is between 0 and 1
        alpha = np.clip(mix_fraction, 0.0, 1.0)
        
        n_pred = alpha * n_pred + (1 - alpha) * n2_pred
        k_pred = alpha * k_pred + (1 - alpha) * k2_pred
        
    return n_pred, k_pred
