import numpy as np
from scipy.optimize import differential_evolution
from tandem_wrapper import calculate_absorbed_power_per_layer
import copy

class StackOptimizer:
    def __init__(self, base_stack, variable_layers_config, targets_config, wavelengths, crosstalk_penalty=1.0):
        self.base_stack = base_stack
        self.variable_layers_config = variable_layers_config
        self.targets_config = targets_config
        self.wavelengths = wavelengths
        self.crosstalk_penalty = crosstalk_penalty
        self.spectrum = np.ones_like(wavelengths)
        
        # Pre-calculate masks
        t1 = targets_config[0]
        self.mask1 = (wavelengths >= t1['band_center'] - t1['band_width']/2) & \
                     (wavelengths <= t1['band_center'] + t1['band_width']/2)
                     
        t2 = targets_config[1]
        self.mask2 = (wavelengths >= t2['band_center'] - t2['band_width']/2) & \
                     (wavelengths <= t2['band_center'] + t2['band_width']/2)
                     
        self.idx1 = t1['layer_index']
        self.idx2 = t2['layer_index']

    def objective(self, thicknesses):
        # Reconstruct stack
        # Note: We need to be careful about deepcopying here if we are in a worker process
        # But creating a new list of tuples is generally safe and fast enough
        current_stack = list(self.base_stack) # Shallow copy of list is enough if we replace tuples
        
        for i, config in enumerate(self.variable_layers_config):
            layer_idx = config['index']
            mat, _ = self.base_stack[layer_idx] # Get original material
            current_stack[layer_idx] = (mat, thicknesses[i])
            
        try:
            results = calculate_absorbed_power_per_layer(current_stack, self.wavelengths, spectrum=self.spectrum)
            layer_data = results['layer_data']
            
            res_idx1 = self.idx1 - 1
            res_idx2 = self.idx2 - 1
            
            S1 = np.sum(layer_data[res_idx1, self.mask1])
            S2 = np.sum(layer_data[res_idx2, self.mask2])
            C1 = np.sum(layer_data[res_idx1, self.mask2])
            C2 = np.sum(layer_data[res_idx2, self.mask1])
            
            score = -1.0 * ((S1 + S2) - self.crosstalk_penalty * (C1 + C2))
            return score
        except Exception:
            return 1e9

def optimize_stack(
    base_stack, 
    variable_layers_config, 
    targets_config, 
    wavelengths, 
    crosstalk_penalty=1.0,
    progress_callback=None
):
    # Create optimizer instance
    optimizer = StackOptimizer(base_stack, variable_layers_config, targets_config, wavelengths, crosstalk_penalty)
    
    # Callback wrapper
    iteration = [0]
    def callback_wrapper(xk, convergence):
        iteration[0] += 1
        if progress_callback:
            progress_callback(iteration[0], xk, convergence)

    bounds = [(conf['min'], conf['max']) for conf in variable_layers_config]
    
    # Use 'workers=-1' to use all available CPU cores
    # Note: 'updating' parameter might need to be 'deferred' for better parallelization 
    # but 'immediate' is default. 'deferred' is better for parallel.
    
    result = differential_evolution(
        optimizer.objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=10, 
        popsize=5, 
        tol=0.05, 
        mutation=(0.5, 1), 
        recombination=0.7,
        disp=False,
        callback=callback_wrapper,
        workers=-1, # Parallelize!
        updating='deferred' # Better for parallelization
    )
    
    # Reconstruct best result
    best_thicknesses = result.x
    optimized_stack = list(base_stack)
    for i, config in enumerate(variable_layers_config):
        layer_idx = config['index']
        mat, _ = base_stack[layer_idx]
        optimized_stack[layer_idx] = (mat, best_thicknesses[i])
        
    # Final metrics
    final_res = calculate_absorbed_power_per_layer(optimized_stack, wavelengths, spectrum=optimizer.spectrum)
    l_data = final_res['layer_data']
    
    res_idx1 = optimizer.idx1 - 1
    res_idx2 = optimizer.idx2 - 1
    
    S1 = np.sum(l_data[res_idx1, optimizer.mask1])
    S2 = np.sum(l_data[res_idx2, optimizer.mask2])
    C1 = np.sum(l_data[res_idx1, optimizer.mask2])
    C2 = np.sum(l_data[res_idx2, optimizer.mask1])
    
    metrics = {
        'S1': S1, 'S2': S2,
        'C1': C1, 'C2': C2,
        'Total_Score': -result.fun
    }
    
    return {
        'optimized_stack': optimized_stack,
        'best_thicknesses': best_thicknesses,
        'metrics': metrics
    }
