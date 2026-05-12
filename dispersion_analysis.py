import numpy as np
from scipy.optimize import minimize
import elli

def fit_tauc_lorentz(wavelengths_nm, n_meas, k_meas, num_oscillators=2):
    """
    Fits Tauc-Lorentz parameters to measured n,k data using pyElli.
    """
    # Target: eps_meas
    eps_meas = (n_meas + 1j * k_meas)**2
    
    # Objective function
    def objective(params):
        # params structure: [Eg, eps_inf, A1, E1, C1, A2, E2, C2, ...]
        Eg = params[0]
        eps_inf = params[1] # pyElli might not have explicit eps_inf in TL, usually added via constant?
        # Actually TaucLorentz in pyElli doesn't seem to have eps_inf. 
        # We usually model eps_inf as a constant offset or a Cauchy term.
        # Let's assume we add a constant real offset.
        
        osc_params = params[2:]
        
        # Constraints
        if Eg < 0 or eps_inf < 1: return 1e9
        if np.any(osc_params <= 0): return 1e9
        
        # Create model
        tl = elli.TaucLorentz(Eg=Eg)
        
        # Add oscillators
        for i in range(len(osc_params) // 3):
            A = osc_params[3*i]
            E = osc_params[3*i+1]
            C = osc_params[3*i+2]
            tl.rep_params.append({'A': A, 'E': E, 'C': C})
            
        # Calculate model eps
        try:
            model_eps = tl.dielectric_function(wavelengths_nm)
            # Add eps_inf (real offset)
            model_eps += eps_inf
            
            # Error
            return np.sum(np.abs(model_eps - eps_meas)**2)
        except:
            return 1e9

    # Initial guess
    # Eg, eps_inf, (A, E, C) * num_oscillators
    x0 = [1.5, 2.0] 
    for i in range(num_oscillators):
        x0.extend([50.0, 3.0 + i*0.5, 1.0])
        
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-3)
    
    return {
        'params': res.x.tolist(),
        'num_oscillators': num_oscillators
    }

def predict_tauc_lorentz(fit_result, wavelengths_nm):
    params = fit_result['params']
    num_oscillators = fit_result['num_oscillators']
    
    Eg = params[0]
    eps_inf = params[1]
    osc_params = params[2:]
    
    tl = elli.TaucLorentz(Eg=Eg)
    for i in range(num_oscillators):
        A = osc_params[3*i]
        E = osc_params[3*i+1]
        C = osc_params[3*i+2]
        tl.rep_params.append({'A': A, 'E': E, 'C': C})
        
    model_eps = tl.dielectric_function(wavelengths_nm)
    model_eps += eps_inf
    
    n = np.sqrt(model_eps)
    return n.real, n.imag
