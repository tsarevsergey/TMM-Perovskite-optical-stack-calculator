import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tmm.tmm_core import coh_tmm, unpolarized_RT, absorp_in_each_layer
import matplotlib.pyplot as plt
import os
from dispersion_analysis import fit_tauc_lorentz, predict_tauc_lorentz

class Material:
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.data = self._load_data()
        # Ensure sorted by wavelength for correct extrapolation
        self.data = self.data.sort_values('Wavelength_nm')
        
        # Use constant extrapolation for safety (prevents negative k)
        # bounds_error=False allows extrapolation
        # fill_value=(lower, upper) uses the edge values
        self.n_fn = interp1d(self.data['Wavelength_nm'], self.data['n'], kind='linear', 
                             bounds_error=False, 
                             fill_value=(self.data['n'].iloc[0], self.data['n'].iloc[-1]))
        self.k_fn = interp1d(self.data['Wavelength_nm'], self.data['k'], kind='linear', 
                             bounds_error=False, 
                             fill_value=(self.data['k'].iloc[0], self.data['k'].iloc[-1]))
        
        # Dispersion Model Parameters (lazy loaded)
        self.model_params = None
        self.use_model_extrapolation = True # Default to True as per user request for better extrapolation

    def _fit_model(self):
        if self.model_params is not None:
            return

        # Fit Tauc-Lorentz
        # Use all available data
        try:
            params = fit_tauc_lorentz(self.data['Wavelength_nm'].values, 
                                      self.data['n'].values, 
                                      self.data['k'].values,
                                      num_oscillators=2)
            self.model_params = params
            # print(f"Fitted model for {self.name}: {params}")
        except Exception as e:
            print(f"Failed to fit model for {self.name}: {e}")
            self.use_model_extrapolation = False

    def _load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Material file not found: {self.filepath}")
        return pd.read_csv(self.filepath)

    def get_n(self, wavelength):
        # Check if wavelength is within range
        min_wl = self.data['Wavelength_nm'].min()
        max_wl = self.data['Wavelength_nm'].max()
        
        if min_wl <= wavelength <= max_wl:
            return self.n_fn(wavelength) + 1j * self.k_fn(wavelength)
        else:
            # Extrapolate
            if self.use_model_extrapolation:
                self._fit_model()
                if self.model_params:
                    # Use model
                    # predict_tauc_lorentz returns scalar or array
                    n_pred, k_pred = predict_tauc_lorentz(self.model_params, wavelength)
                    
                    val = n_pred + 1j * k_pred
                    # If we got a 1-element array (because predict_tauc_lorentz uses atleast_1d), extract scalar
                    if np.ndim(val) > 0 and val.size == 1:
                        return val.item()
                    return val
            
            # Fallback to constant extrapolation (already handled by interp1d if we used it, 
            # but here we are explicitly checking bounds)
            # Actually, our interp1d is configured for constant extrapolation.
            return self.n_fn(wavelength) + 1j * self.k_fn(wavelength)

def calculate_optical_properties(stack, wavelengths, angle_deg=0, spectrum=None):
    """
    Calculates R, T, A for a given stack over a range of wavelengths.
    Optionally weights by a spectrum.
    
    stack: List of tuples (Material, thickness_nm). Use float('inf') for semi-infinite layers.
    wavelengths: Array of wavelengths in nm.
    angle_deg: Incident angle in degrees.
    spectrum: Optional array of intensity values corresponding to wavelengths.
    """
    
    R_list = []
    T_list = []
    A_list = []
    
    angle_rad = np.deg2rad(angle_deg)
    
    for wl in wavelengths:
        n_list = []
        d_list = []
        
        for mat, d in stack:
            if mat is None: # Air or Vacuum
                n_list.append(1.0)
            else:
                n_list.append(mat.get_n(wl))
            d_list.append(d)
            
        # Using unpolarized light approximation
        # For more precision, calculate 's' and 'p' separately and average
        data = unpolarized_RT(n_list, d_list, angle_rad, wl)
        R = data['R']
        T = data['T']
        A = 1 - R - T
        
        R_list.append(R)
        T_list.append(T)
        A_list.append(A)
        
    R_list = np.array(R_list)
    T_list = np.array(T_list)
    A_list = np.array(A_list)
    
    results = {
        'wavelengths': wavelengths,
        'R': R_list,
        'T': T_list,
        'A': A_list
    }
    
    if spectrum is not None:
        # Ensure spectrum is same length as wavelengths
        if len(spectrum) != len(wavelengths):
             # If spectrum is a function, evaluate it
            if callable(spectrum):
                spectrum_vals = spectrum(wavelengths)
            else:
                # Interpolate provided spectrum to current wavelengths
                # Assuming spectrum is (N,) array matching some other wavelength base? 
                # For simplicity, let's assume user passes spectrum matching wavelengths or we handle it outside.
                # If spectrum is just an array of same shape, use it.
                spectrum_vals = np.array(spectrum)
        else:
            spectrum_vals = np.array(spectrum)

        total_power = np.trapz(spectrum_vals, wavelengths)
        
        integrated_R = np.trapz(R_list * spectrum_vals, wavelengths) / total_power
        integrated_T = np.trapz(T_list * spectrum_vals, wavelengths) / total_power
        integrated_A = np.trapz(A_list * spectrum_vals, wavelengths) / total_power
        
        results['integrated'] = {
            'R': integrated_R,
            'T': integrated_T,
            'A': integrated_A
        }
        
    return results

def calculate_absorbed_power_per_layer(stack, wavelengths, angle_deg=0, spectrum=None):
    """
    Calculates absorption in each layer + R and T.
    Returns a dictionary with 'wavelengths' and 'layer_absorption' (list of arrays).
    layer_absorption[0] is Reflection.
    layer_absorption[-1] is Transmission.
    layer_absorption[1:-1] are absorptions in physical layers.
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Initialize arrays
    # Shape: (num_layers, num_wavelengths)
    # num_layers includes ambient (R) and substrate (T)
    num_layers = len(stack)
    layer_absorptions = np.zeros((num_layers, len(wavelengths)))
    
    for i, wl in enumerate(wavelengths):
        n_list = []
        d_list = []
        
        for mat, d in stack:
            if mat is None:
                n_list.append(1.0)
            else:
                n_list.append(mat.get_n(wl))
            d_list.append(d)
            
        # Calculate for s and p
        data_s = coh_tmm('s', n_list, d_list, angle_rad, wl)
        data_p = coh_tmm('p', n_list, d_list, angle_rad, wl)
        
        abs_s = absorp_in_each_layer(data_s)
        abs_p = absorp_in_each_layer(data_p)
        
        # Average
        abs_avg = (abs_s + abs_p) / 2.0
        
        layer_absorptions[:, i] = abs_avg
        
    results = {
        'wavelengths': wavelengths,
        'layer_data': layer_absorptions # Rows are layers, Cols are wavelengths
    }
    
    if spectrum is not None:
        if len(spectrum) != len(wavelengths):
             spectrum_vals = np.array(spectrum) # Assume matching if not checking/interpolating
        else:
            spectrum_vals = np.array(spectrum)
            
        total_power = np.trapz(spectrum_vals, wavelengths)
        
        integrated_abs = []
        for l_idx in range(num_layers):
            int_val = np.trapz(layer_absorptions[l_idx] * spectrum_vals, wavelengths) / total_power
            integrated_abs.append(int_val)
            
        results['integrated_layer_abs'] = integrated_abs
        
    return results

def main():
    # 1. Load Materials
    try:
        ito = Material('ITO', 'materials/ITO.csv')
        a_si = Material('a-Si', 'materials/a-Si.csv')
        nio = Material('NiO', 'materials/NiO.csv')
        pac2z = Material('2PACz', 'materials/2PACz.csv')
        perovskite = Material('Perovskite', 'materials/Perovskite.csv')
        c60 = Material('C60', 'materials/C60.csv')
        sno2 = Material('SnO2', 'materials/SnO2.csv')
    except FileNotFoundError as e:
        print(e)
        print("Please ensure material CSV files are in the 'materials' directory.")
        return

    # 2. Define Stack
    # Structure: ITO / a-Si / ITO / NiO / 2PACz / Perovskite / C60 / SnO2 / ITO
    # Thicknesses are placeholders (nm)
    stack = [
        (None, float('inf')),       # Air (Input Medium)
        (ito, 100),
        (a_si, 50),
        (ito, 100),
        (nio, 20),
        (pac2z, 5),
        (perovskite, 500),
        (c60, 20),
        (sno2, 20),
        (ito, 100),
        (None, float('inf'))        # Air (Output Medium)
    ]
    
    # 3. Define Wavelengths and Spectrum
    wavelengths = np.linspace(300, 1200, 100)
    
    # Dummy Spectrum (Flat) - Replace with AM1.5G data
    spectrum = np.ones_like(wavelengths) 
    
    # 4. Calculate
    print("Calculating optical properties...")
    results = calculate_optical_properties(stack, wavelengths, spectrum=spectrum)
    
    # 5. Output Results
    print("\n--- Integrated Results (Weighted by Spectrum) ---")
    print(f"Reflection: {results['integrated']['R']*100:.2f}%")
    print(f"Transmission: {results['integrated']['T']*100:.2f}%")
    print(f"Absorption:   {results['integrated']['A']*100:.2f}%")
    
    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, results['R'], label='Reflection')
    plt.plot(wavelengths, results['T'], label='Transmission')
    plt.plot(wavelengths, results['A'], label='Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Fraction')
    plt.title('Optical Properties of Perovskite Tandem Stack')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
