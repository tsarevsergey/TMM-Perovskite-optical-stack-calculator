import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tandem_wrapper import Material, calculate_absorbed_power_per_layer

# 1. Layout Configuration
st.set_page_config(page_title="TMM Analysis", layout="centered")

# 2. CSS for Centering Table Values
st.markdown("""
<style>
    .stDataFrame div[data-testid="stDataFrameResizable"] table td {
        text-align: center !important;
    }
    .stDataFrame div[data-testid="stDataFrameResizable"] table th {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

MATERIALS_DIR = 'materials'

@st.cache_data
def get_available_materials():
    files = glob.glob(os.path.join(MATERIALS_DIR, '*.csv'))
    materials = [os.path.basename(f).replace('.csv', '') for f in files]
    return sorted(materials)

@st.cache_resource
def load_material(name):
    filepath = os.path.join(MATERIALS_DIR, f"{name}.csv")
    return Material(name, filepath)

def plot_material(material):
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('n', color=color)
    ax1.plot(material.data['Wavelength_nm'], material.data['n'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('k', color=color)
    ax2.plot(material.data['Wavelength_nm'], material.data['k'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Optical Constants for {material.name}")
    st.pyplot(fig)

def main():
    st.title("Perovskite Tandem TMM Analysis")
    
    tabs = st.tabs(["Material Viewer", "Simulation"])
    
    with tabs[0]:
        st.header("Material Viewer")
        materials = get_available_materials()
        selected_material = st.selectbox("Select Material", materials)
        
        if selected_material:
            try:
                mat = load_material(selected_material)
                plot_material(mat)
            except Exception as e:
                st.error(f"Error loading material: {e}")

    with tabs[1]:
        st.header("Device Simulation")
        
        # Default Stack Definition
        default_stack_data = [
            {"Material": "ITO", "Thickness (nm)": 100},
            {"Material": "NiO", "Thickness (nm)": 40},
            {"Material": "MAPbI3", "Thickness (nm)": 500},
            {"Material": "C60", "Thickness (nm)": 20},
            {"Material": "SnO2", "Thickness (nm)": 40},
            {"Material": "ITO", "Thickness (nm)": 100},
            {"Material": "SnO2", "Thickness (nm)": 40},
            {"Material": "C60", "Thickness (nm)": 5},
            {"Material": "MAPbBr3", "Thickness (nm)": 400},
            {"Material": "C60", "Thickness (nm)": 20},
            {"Material": "SnO2", "Thickness (nm)": 40},
            {"Material": "ITO", "Thickness (nm)": 140},
        ]
        
        st.subheader("Stack Configuration")
        st.info("Edit the table below. Top is incident side.")
        
        stack_df = pd.DataFrame(default_stack_data)
        edited_stack_df = st.data_editor(
            stack_df,
            column_config={
                "Material": st.column_config.SelectboxColumn(
                    "Material",
                    width="medium",
                    options=get_available_materials(),
                    required=True,
                ),
                "Thickness (nm)": st.column_config.NumberColumn(
                    "Thickness (nm)",
                    min_value=0,
                    max_value=10000,
                    step=1,
                    format="%d",
                ),
            },
            num_rows="dynamic",
            use_container_width=True
        )
        
        light_direction = st.radio("Light Direction", ["Top (Incident on first layer)", "Bottom (Incident on last layer)"], horizontal=True)
        
        col1, col2 = st.columns(2)
        with col1:
            min_wl = st.number_input("Min Wavelength (nm)", value=300, step=10)
        with col2:
            max_wl = st.number_input("Max Wavelength (nm)", value=1200, step=10)
        
        colormap_name = st.selectbox(
            "Color Scale", 
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Spectral', 'coolwarm', 'RdYlBu', 'RdYlGn'],
            index=0
        )
            
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Calculating..."):
                try:
                    # Build Stack
                    stack = [(None, float('inf'))] # Air
                    layer_names = ["Reflection"] # Index 0
                    
                    # Extract layers
                    layers_to_process = []
                    for _, row in edited_stack_df.iterrows():
                        mat_name = row['Material']
                        thickness = row['Thickness (nm)']
                        mat = load_material(mat_name)
                        layers_to_process.append((mat, thickness, mat_name))
                    
                    # Reverse if Bottom illumination
                    if "Bottom" in light_direction:
                        layers_to_process = layers_to_process[::-1]
                    
                    for mat, thickness, name in layers_to_process:
                        stack.append((mat, thickness))
                        layer_names.append(f"{name} ({thickness}nm)")
                        
                    stack.append((None, float('inf'))) # Air
                    layer_names.append("Transmission") # Index -1
                    
                    # Wavelengths
                    wavelengths = np.linspace(min_wl, max_wl, 500)
                    spectrum = np.ones_like(wavelengths) # Flat spectrum
                    
                    # Calculate Per Layer
                    results = calculate_absorbed_power_per_layer(stack, wavelengths, spectrum=spectrum)
                    
                    # Store in session state
                    st.session_state['simulation_results'] = {
                        'results': results,
                        'layer_names': layer_names,
                        'wavelengths': wavelengths,
                        'min_wl': min_wl,
                        'max_wl': max_wl
                    }
                    
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    st.exception(e)

        # Display Results if available
        if 'simulation_results' in st.session_state:
            res_data = st.session_state['simulation_results']
            results = res_data['results']
            layer_names = res_data['layer_names']
            wavelengths = res_data['wavelengths']
            
            # Use stored min/max to ensure consistency
            plot_min_wl = res_data['min_wl']
            plot_max_wl = res_data['max_wl']

            layer_data = results['layer_data'] # Shape: (num_layers, num_wl)
            integrated = results['integrated_layer_abs']
            
            # Prepare Stackplot Data
            # Order: Active Layers (1 to N-2), then Transmission (-1), then Reflection (0)
            # This puts active layers at the bottom
            
            # Indices for plotting
            # 0: Reflection
            # 1..N-2: Layers
            # N-1: Transmission
            
            num_layers = layer_data.shape[0]
            
            # Create ordered list for stackplot
            # We want: Layer 1, Layer 2, ..., Layer N-2, Transmission, Reflection
            
            stack_indices = list(range(1, num_layers - 1)) + [num_layers - 1, 0]
            stack_labels = [layer_names[i] for i in stack_indices]
            stack_y = [layer_data[i, :] for i in stack_indices]
            stack_integrated = [integrated[i] for i in stack_indices]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create custom palette
            # Active layers: varying colors
            # Transmission: Light Gray
            # Reflection: Dark Gray/Black
            
            # Generate colors for active layers
            cmap = plt.get_cmap(colormap_name)
            active_colors = [cmap(i) for i in np.linspace(0, 1, len(stack_indices)-2)]
            colors = active_colors + ['#e0e0e0', '#808080'] # Trans, Refl
            
            ax.stackplot(wavelengths, *stack_y, labels=[f"{l} ({v*100:.1f}%)" for l, v in zip(stack_labels, stack_integrated)], colors=colors, alpha=0.8)
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Fraction of Light')
            ax.set_title('Light Distribution (Stacked)')
            ax.set_xlim(plot_min_wl, plot_max_wl)
            ax.set_ylim(0, 1)
            
            # Legend outside
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            st.pyplot(fig)
            

            # Absorption Inspection
            st.subheader("Layer Absorption Inspection")
            inspect_wl = st.slider("Inspection Wavelength (nm)", min_value=int(plot_min_wl), max_value=int(plot_max_wl), value=500, step=10)
            
            # Define range +/- 5nm
            wl_mask = (wavelengths >= inspect_wl - 5) & (wavelengths <= inspect_wl + 5)
            
            if np.any(wl_mask):
                st.write(f"Average absorption in range {inspect_wl-5} nm - {inspect_wl+5} nm:")
                inspection_data = []
                for i, name in enumerate(layer_names):
                    # Average absorption in the window
                    avg_abs = np.mean(layer_data[i, wl_mask])
                    inspection_data.append({"Layer": name, "Absorption (%)": f"{avg_abs*100:.2f}%"})
                
                st.table(pd.DataFrame(inspection_data))
            else:
                st.warning("No data points in the selected range.")

            # Download Data
            # Create DataFrame with all columns
            df_dict = {'Wavelength_nm': wavelengths}
            for i, name in enumerate(layer_names):
                df_dict[name] = layer_data[i, :]
                
            csv_data = pd.DataFrame(df_dict).to_csv(index=False)
            
            st.download_button(
                label="Download Detailed Results CSV",
                data=csv_data,
                file_name="tmm_layer_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
