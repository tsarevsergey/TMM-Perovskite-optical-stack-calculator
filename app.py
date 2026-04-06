import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import json
import hashlib
from tandem_wrapper import Material, calculate_absorbed_power_per_layer
import material_interpolation as mi
import optimizer as opt

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

def _coerce_nm_value(value, default=0):
    if pd.isna(value):
        return int(default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)

def normalize_detector_stack(records):
    normalized = []
    for row in records:
        normalized.append({
            "Material": str(row.get("Material") or "ITO"),
            "Thickness (nm)": max(0, _coerce_nm_value(row.get("Thickness (nm)"), 0)),
        })
    return normalized

def normalize_optimizer_stack(records):
    normalized = []
    for row in records:
        thickness_nm = max(0, _coerce_nm_value(row.get("Thickness (nm)"), 0))
        min_default = int(thickness_nm * 0.5)
        max_default = max(thickness_nm, int(thickness_nm * 1.5))
        min_nm = max(0, _coerce_nm_value(row.get("Min (nm)"), min_default))
        max_nm = max(min_nm, _coerce_nm_value(row.get("Max (nm)"), max_default))
        normalized.append({
            "Material": str(row.get("Material") or "ITO"),
            "Thickness (nm)": thickness_nm,
            "Optimize": bool(row.get("Optimize", False)),
            "Min (nm)": min_nm,
            "Max (nm)": max_nm,
        })
    return normalized

def reset_editor_state(data_key, version_key, records, normalizer):
    st.session_state[data_key] = normalizer(records)
    st.session_state[version_key] = st.session_state.get(version_key, 0) + 1

def sync_editor_state(data_key, edited_df, normalizer):
    current_data = normalizer(edited_df.to_dict('records'))
    if current_data != st.session_state[data_key]:
        st.session_state[data_key] = current_data

def load_uploaded_json_once(uploaded_file, signature_key):
    if uploaded_file is None:
        st.session_state.pop(signature_key, None)
        return None

    raw_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(raw_bytes).hexdigest()
    if st.session_state.get(signature_key) == signature:
        return None

    st.session_state[signature_key] = signature
    return json.loads(raw_bytes.decode("utf-8"))

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
    
    tabs = st.tabs(["Material Viewer", "Detector Maker", "Material Mixer", "Optimization"])
    
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
        st.header("Detector Maker")
        
        # Session State for Stack
        if 'detector_stack_data' not in st.session_state:
            st.session_state['detector_stack_data'] = normalize_detector_stack([
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
            ])
        if 'detector_stack_editor_version' not in st.session_state:
            st.session_state['detector_stack_editor_version'] = 0

        # --- Load/Save Controls ---
        col_load, col_save = st.columns([2, 1])
        
        with col_load:
            uploaded_stack = st.file_uploader("Load Stack Configuration (JSON)", type=['json'], key="stack_uploader")
            if uploaded_stack is not None:
                try:
                    loaded_data = load_uploaded_json_once(uploaded_stack, "stack_uploader_signature")
                    if loaded_data is None:
                        loaded_data = st.session_state['detector_stack_data']
                    # Basic validation
                    if isinstance(loaded_data, list) and len(loaded_data) > 0 and "Material" in loaded_data[0]:
                        if loaded_data != st.session_state['detector_stack_data']:
                            reset_editor_state(
                                'detector_stack_data',
                                'detector_stack_editor_version',
                                loaded_data,
                                normalize_detector_stack,
                            )
                            st.success("Stack loaded!")
                    else:
                        st.error("Invalid stack file format.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        
        st.subheader("Stack Configuration")
        st.info("Edit the table below. Top is incident side. You can also add/remove rows using the toolbar or buttons below.")
        
        # Buttons for explicit Add/Remove
        col_add, col_rem = st.columns(2)
        with col_add:
            if st.button("Add Layer"):
                new_rows = st.session_state['detector_stack_data'] + [{"Material": "ITO", "Thickness (nm)": 100}]
                reset_editor_state(
                    'detector_stack_data',
                    'detector_stack_editor_version',
                    new_rows,
                    normalize_detector_stack,
                )
                st.rerun()
        with col_rem:
            if st.button("Remove Last Layer"):
                if len(st.session_state['detector_stack_data']) > 0:
                    new_rows = st.session_state['detector_stack_data'][:-1]
                    reset_editor_state(
                        'detector_stack_data',
                        'detector_stack_editor_version',
                        new_rows,
                        normalize_detector_stack,
                    )
                    st.rerun()
        
        stack_df = pd.DataFrame(st.session_state['detector_stack_data'])
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
            width="stretch", # Replaces use_container_width=True
            key=f"stack_editor_{st.session_state['detector_stack_editor_version']}"
        )
        
        # Sync edits back to session state immediately so the first selection/change sticks.
        sync_editor_state('detector_stack_data', edited_stack_df, normalize_detector_stack)
        current_stack_df = pd.DataFrame(st.session_state['detector_stack_data'])
        
        with col_save:
            # Prepare JSON for download
            # Convert DF back to list of dicts
            stack_json = current_stack_df.to_json(orient="records")
            st.download_button(
                label="Save Configuration",
                data=stack_json,
                file_name="detector_stack.json",
                mime="application/json"
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
                    for _, row in current_stack_df.iterrows():
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
                        'layer_details': [None] + [
                            {"material_name": name, "thickness_nm": thickness}
                            for _, thickness, name in layers_to_process
                        ] + [None],
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
            layer_details = res_data.get('layer_details', [])
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
            

            # Layer-by-layer inspection
            st.subheader("Layer-by-Layer Inspection")
            inspect_wl = st.slider("Inspection Wavelength (nm)", min_value=int(plot_min_wl), max_value=int(plot_max_wl), value=500, step=10)
            
            # Define range +/- 5nm
            wl_mask = (wavelengths >= inspect_wl - 5) & (wavelengths <= inspect_wl + 5)
            
            if np.any(wl_mask):
                st.write(f"Average fraction of incident light in range {inspect_wl-5} nm - {inspect_wl+5} nm:")
                inspection_data = []
                for i, name in enumerate(layer_names):
                    # Average absorption in the window
                    avg_fraction = np.mean(layer_data[i, wl_mask])
                    row = {
                        "Component": name,
                        "Fraction of Incident Light (%)": f"{avg_fraction*100:.2f}%"
                    }

                    detail = layer_details[i] if i < len(layer_details) else None
                    if detail is not None:
                        material = load_material(detail["material_name"])
                        k_value = float(np.imag(material.get_n(inspect_wl)))
                        thickness_nm = detail["thickness_nm"]
                        single_pass = 1 - np.exp(-4 * np.pi * k_value * thickness_nm / inspect_wl)
                        row["k @ λ"] = f"{k_value:.3f}"
                        row["Single-Pass Estimate (%)"] = f"{single_pass*100:.2f}%"
                    else:
                        row["k @ λ"] = "-"
                        row["Single-Pass Estimate (%)"] = "-"

                    inspection_data.append(row)
                
                st.table(pd.DataFrame(inspection_data))
                st.caption("Single-pass estimate ignores interface reflection and interference. It is shown only as a quick sanity check for physical layers.")
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


    with tabs[2]:
        st.header("Material Mixer")
        st.info("Approximate new material properties by shifting/mixing bandgaps of existing materials.")
        
        col_ref1, col_ref2 = st.columns(2)
        
        # Reference 1
        with col_ref1:
            st.subheader("Reference 1")
            materials = get_available_materials()
            ref1_name = st.selectbox("Select Material 1", materials, key="ref1_select")
            
            ref1_mat = load_material(ref1_name)
            
            # Auto-estimate Eg
            est_eg1 = mi.estimate_bandgap(ref1_mat.data['Wavelength_nm'].values, ref1_mat.data['k'].values)
            
            ref1_eg = st.number_input("Bandgap (eV)", value=float(f"{est_eg1:.3f}"), step=0.01, key="ref1_eg", format="%.3f")
            
        # Reference 2
        with col_ref2:
            st.subheader("Reference 2 (Optional)")
            use_ref2 = st.checkbox("Use Second Reference")
            
            ref2_mat = None
            ref2_eg = None
            mix_frac = 1.0
            
            if use_ref2:
                ref2_name = st.selectbox("Select Material 2", materials, key="ref2_select", index=min(1, len(materials)-1))
                ref2_mat = load_material(ref2_name)
                
                est_eg2 = mi.estimate_bandgap(ref2_mat.data['Wavelength_nm'].values, ref2_mat.data['k'].values)
                ref2_eg = st.number_input("Bandgap (eV)", value=float(f"{est_eg2:.3f}"), step=0.01, key="ref2_eg", format="%.3f")
                
                mix_frac = st.slider(f"Weight of {ref1_name}", 0.0, 1.0, 0.5, help=f"1.0 = Pure {ref1_name}, 0.0 = Pure {ref2_name}")
        
        st.divider()
        
        # Target Configuration
        st.subheader("Target Properties")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            target_eg = st.number_input("Target Bandgap (eV)", value=1.75, step=0.01, format="%.3f")
            new_mat_name = st.text_input("New Material Name", value=f"Mixed_Perovskite_{target_eg}eV")
            
        with col_t2:
            st.write("Preview Settings")
            show_preview = st.checkbox("Show Preview", value=True)

        if show_preview:
            # Generate preview data
            preview_wl = np.linspace(300, 1200, 500)
            
            ref1_data_dict = {
                'wavelength_nm': ref1_mat.data['Wavelength_nm'].values,
                'n': ref1_mat.data['n'].values,
                'k': ref1_mat.data['k'].values
            }
            
            ref2_data_dict = None
            if use_ref2 and ref2_mat:
                ref2_data_dict = {
                    'wavelength_nm': ref2_mat.data['Wavelength_nm'].values,
                    'n': ref2_mat.data['n'].values,
                    'k': ref2_mat.data['k'].values
                }
            
            try:
                n_new, k_new = mi.interpolate_material_properties(
                    preview_wl, target_eg,
                    ref1_data_dict, ref1_eg,
                    ref2_data_dict, ref2_eg,
                    mix_fraction=mix_frac
                )
                
                # Plotting
                fig_mix, ax_mix = plt.subplots(figsize=(10, 5))
                ax_mix.plot(preview_wl, n_new, label=f"{new_mat_name} (n)", color='blue')
                ax_mix.plot(preview_wl, k_new, label=f"{new_mat_name} (k)", color='red')
                
                # Plot References for comparison
                ax_mix.plot(ref1_mat.data['Wavelength_nm'], ref1_mat.data['n'], '--', alpha=0.5, color='lightblue', label=f"{ref1_name} (n)")
                ax_mix.plot(ref1_mat.data['Wavelength_nm'], ref1_mat.data['k'], '--', alpha=0.5, color='pink', label=f"{ref1_name} (k)")
                
                if use_ref2 and ref2_mat:
                    ax_mix.plot(ref2_mat.data['Wavelength_nm'], ref2_mat.data['n'], ':', alpha=0.5, color='cyan', label=f"{ref2_name} (n)")
                    ax_mix.plot(ref2_mat.data['Wavelength_nm'], ref2_mat.data['k'], ':', alpha=0.5, color='orange', label=f"{ref2_name} (k)")
                
                ax_mix.set_xlabel("Wavelength (nm)")
                ax_mix.set_ylabel("Optical Constants")
                ax_mix.set_title(f"Approximation: Eg {target_eg} eV")
                ax_mix.set_ylim(0, 5)
                ax_mix.legend()
                st.pyplot(fig_mix)
                
                if st.button("Save New Material"):
                    if not new_mat_name:
                        st.error("Please enter a name for the new material.")
                    else:
                        # Save to CSV
                        save_path = os.path.join(MATERIALS_DIR, f"{new_mat_name}.csv")
                        
                        # Create DataFrame
                        df_save = pd.DataFrame({
                            'Wavelength_nm': preview_wl,
                            'n': n_new,
                            'k': k_new
                        })
                        
                        df_save.to_csv(save_path, index=False)
                        st.success(f"Saved {new_mat_name} to {save_path}")
                        
                        # Clear cache to show new material in lists
                        st.cache_data.clear()
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error generating approximation: {e}")

    with tabs[3]:
        st.header("Stack Optimization")
        st.info("Optimize layer thicknesses for multiband detection (Maximize Signal, Minimize Crosstalk).")
        
        # 1. Stack Definition for Optimization
        st.subheader("1. Define Stack & Optimization Ranges")
        
        # Load from file option
        uploaded_opt_stack = st.file_uploader("Load Stack for Optimization (JSON)", type=['json'], key="opt_stack_uploader")
        
        if uploaded_opt_stack is not None:
            try:
                loaded_data = load_uploaded_json_once(uploaded_opt_stack, "opt_stack_uploader_signature")
                if loaded_data is None:
                    loaded_data = st.session_state.get('opt_stack_data', [])
                # Convert to optimization format
                # We need to add "Optimize", "Min", "Max" columns if they don't exist
                new_opt_data = []
                for layer in loaded_data:
                    new_layer = layer.copy()
                    if "Optimize" not in new_layer:
                        new_layer["Optimize"] = False
                    if "Min (nm)" not in new_layer:
                        # Default range: +/- 50% or fixed bounds
                        th = layer.get("Thickness (nm)", 0)
                        new_layer["Min (nm)"] = int(th * 0.5)
                        new_layer["Max (nm)"] = int(th * 1.5)
                    new_opt_data.append(new_layer)
                
                normalized_opt_data = normalize_optimizer_stack(new_opt_data)
                if normalized_opt_data != st.session_state.get('opt_stack_data'):
                    reset_editor_state(
                        'opt_stack_data',
                        'opt_stack_editor_version',
                        normalized_opt_data,
                        normalize_optimizer_stack,
                    )
                    st.success("Stack loaded into optimizer!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        
        if 'opt_stack_data' not in st.session_state:
            # Initialize with default stack but add optimization columns
            default_opt_data = normalize_optimizer_stack([
                {"Material": "ITO", "Thickness (nm)": 100, "Optimize": False, "Min (nm)": 50, "Max (nm)": 150},
                {"Material": "NiO", "Thickness (nm)": 40, "Optimize": False, "Min (nm)": 10, "Max (nm)": 100},
                {"Material": "MAPbI3", "Thickness (nm)": 500, "Optimize": True, "Min (nm)": 300, "Max (nm)": 800},
                {"Material": "C60", "Thickness (nm)": 20, "Optimize": False, "Min (nm)": 5, "Max (nm)": 50},
                {"Material": "SnO2", "Thickness (nm)": 40, "Optimize": False, "Min (nm)": 10, "Max (nm)": 100},
                {"Material": "ITO", "Thickness (nm)": 100, "Optimize": False, "Min (nm)": 50, "Max (nm)": 150},
                {"Material": "SnO2", "Thickness (nm)": 40}, # Missing fields in original default, let's fix
                {"Material": "C60", "Thickness (nm)": 5, "Optimize": False, "Min (nm)": 1, "Max (nm)": 20},
                {"Material": "MAPbBr3", "Thickness (nm)": 400, "Optimize": True, "Min (nm)": 200, "Max (nm)": 600},
                {"Material": "C60", "Thickness (nm)": 20, "Optimize": False, "Min (nm)": 5, "Max (nm)": 50},
                {"Material": "SnO2", "Thickness (nm)": 40, "Optimize": False, "Min (nm)": 10, "Max (nm)": 100},
                {"Material": "ITO", "Thickness (nm)": 140, "Optimize": False, "Min (nm)": 50, "Max (nm)": 200},
            ])
            st.session_state['opt_stack_data'] = default_opt_data
        if 'opt_stack_editor_version' not in st.session_state:
            st.session_state['opt_stack_editor_version'] = 0
            
        opt_df = pd.DataFrame(st.session_state['opt_stack_data'])

        with st.form("optimizer_form"):
            edited_opt_df = st.data_editor(
                opt_df,
                column_config={
                    "Material": st.column_config.SelectboxColumn("Material", options=get_available_materials(), required=True),
                    "Thickness (nm)": st.column_config.NumberColumn("Initial Thickness", min_value=0, format="%d"),
                    "Optimize": st.column_config.CheckboxColumn("Optimize?", default=False),
                    "Min (nm)": st.column_config.NumberColumn("Min", min_value=0, format="%d"),
                    "Max (nm)": st.column_config.NumberColumn("Max", min_value=0, format="%d"),
                },
                num_rows="dynamic",
                width="stretch",
                key=f"opt_editor_{st.session_state['opt_stack_editor_version']}"
            )

            form_opt_df = pd.DataFrame(normalize_optimizer_stack(edited_opt_df.to_dict('records')))

            # Light Direction
            opt_light_dir = st.radio("Light Direction", ["Top (Incident on Layer 1)", "Bottom (Incident on Last Layer)"], horizontal=True, key="opt_light_dir")

            # Indices in stack: 1 to N (0 is Air, -1 is Air)
            # In the DF, index i corresponds to Stack Layer i+1
            layer_options = {f"Layer {i+1}: {row['Material']}": i for i, row in form_opt_df.iterrows()}
            layer_option_labels = list(layer_options.keys())
            layer_select_options = layer_option_labels if layer_option_labels else ["No layers available"]

            col_target1, col_target2 = st.columns(2)
            
            with col_target1:
                st.markdown("#### Detector 1 (Top/Blue)")
                t1_layer_key = st.selectbox("Select Absorber Layer 1", options=layer_select_options, index=8 if len(layer_option_labels)>8 else 0, disabled=not layer_option_labels)
                t1_center = st.number_input("Center Wavelength (nm)", value=450, step=10, key="t1_c")
                t1_width = st.number_input("Bandwidth (nm)", value=100, step=10, key="t1_w")
                
            with col_target2:
                st.markdown("#### Detector 2 (Bottom/Red)")
                t2_layer_key = st.selectbox("Select Absorber Layer 2", options=layer_select_options, index=2 if len(layer_option_labels)>2 else 0, disabled=not layer_option_labels)
                t2_center = st.number_input("Center Wavelength (nm)", value=750, step=10, key="t2_c")
                t2_width = st.number_input("Bandwidth (nm)", value=100, step=10, key="t2_w")
                
            crosstalk_pen = st.slider("Crosstalk Penalty Weight", 0.0, 5.0, 1.0, help="Higher value = stricter penalty for crosstalk.")
            start_optimization = st.form_submit_button("Start Optimization", type="primary")

        sync_editor_state('opt_stack_data', form_opt_df, normalize_optimizer_stack)
        current_opt_df = pd.DataFrame(st.session_state['opt_stack_data'])
        
        if start_optimization:
            with st.spinner("Optimizing... This may take a minute."):
                try:
                    if not layer_options:
                        raise ValueError("Add at least one layer before starting optimization.")

                    # Prepare Inputs
                    # 1. Base Stack
                    base_stack = [(None, float('inf'))]
                    variable_layers_config = []
                    
                    # Iterate DF
                    # Stack indices: Air(0), Layer1(1), Layer2(2)...
                    for i, row in current_opt_df.iterrows():
                        mat_name = row['Material']
                        thickness = float(row['Thickness (nm)'])
                        mat = load_material(mat_name)
                        base_stack.append((mat, thickness))
                        
                        if row['Optimize']:
                            min_nm = float(row['Min (nm)'])
                            max_nm = float(row['Max (nm)'])

                            if not np.isfinite(min_nm) or not np.isfinite(max_nm):
                                raise ValueError(f"Layer {i+1} has an invalid optimization range.")
                            if max_nm < min_nm:
                                raise ValueError(f"Layer {i+1} has Max < Min.")

                            # Config for optimizer
                            # Index in base_stack is i + 1
                            variable_layers_config.append({
                                'index': i + 1,
                                'min': min_nm,
                                'max': max_nm
                            })
                    
                    base_stack.append((None, float('inf')))
                    
                    if not variable_layers_config:
                        raise ValueError("Select at least one layer to optimize.")
                    
                    # 2. Targets
                    # Map DF index to Stack index
                    idx1 = layer_options[t1_layer_key] + 1
                    idx2 = layer_options[t2_layer_key] + 1
                    
                    targets_config = [
                        {'layer_index': idx1, 'band_center': t1_center, 'band_width': t1_width},
                        {'layer_index': idx2, 'band_center': t2_center, 'band_width': t2_width}
                    ]
                    
                    # Handle Light Direction (Reverse Stack if Bottom)
                    # Stack indices: 0 (Air), 1..N (Layers), N+1 (Air)
                    # Total len = N + 2
                    if "Bottom" in opt_light_dir:
                        # Reverse the stack (including Air layers)
                        base_stack = base_stack[::-1]
                        
                        # Update Indices
                        # If length is L, new_idx = L - 1 - old_idx
                        L = len(base_stack)
                        
                        for conf in variable_layers_config:
                            conf['index'] = L - 1 - conf['index']
                            
                        for conf in targets_config:
                            conf['layer_index'] = L - 1 - conf['layer_index']
                    
                    # 3. Wavelengths
                    # Cover the full range of interest
                    min_wl_opt = min(t1_center - t1_width, t2_center - t2_width) - 50
                    max_wl_opt = max(t1_center + t1_width, t2_center + t2_width) + 50
                    wavelengths_opt = np.linspace(max(300, min_wl_opt), min(1200, max_wl_opt), 300)
                    
                    # Run Optimization
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(iter_num, xk, convergence):
                        # Max iter is 10, but it might stop early or go slightly over if popsize logic differs
                        # Just show iteration count
                        progress = min(iter_num / 10.0, 1.0)
                        progress_bar.progress(progress)
                        
                        # Format current best thicknesses
                        th_str = ", ".join([f"{x:.1f}" for x in xk])
                        status_text.text(f"Iteration {iter_num}: Best Thicknesses so far: [{th_str}]")

                    opt_res = opt.optimize_stack(
                        base_stack,
                        variable_layers_config,
                        targets_config,
                        wavelengths_opt,
                        crosstalk_penalty=crosstalk_pen,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Optimization Finished!")
                    
                    # Display Results
                    st.success("Optimization Complete!")
                    
                    # Metrics
                    m = opt_res['metrics']
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Signal 1 (Band 1)", f"{m['S1']:.2f}")
                    col_m2.metric("Signal 2 (Band 2)", f"{m['S2']:.2f}")
                    col_m3.metric("Total Score", f"{m['Total_Score']:.2f}")
                    
                    col_c1, col_c2 = st.columns(2)
                    col_c1.metric("Crosstalk 1 (L1 in B2)", f"{m['C1']:.2f}", delta=-m['C1'], delta_color="inverse")
                    col_c2.metric("Crosstalk 2 (L2 in B1)", f"{m['C2']:.2f}", delta=-m['C2'], delta_color="inverse")
                    
                    # Optimized Thicknesses Table
                    st.subheader("Optimized Thicknesses")
                    res_data = []
                    best_th = opt_res['best_thicknesses']
                    
                    # Map back to names
                    # variable_layers_config has indices relative to the stack passed to optimizer
                    # If we reversed, we need to be careful.
                    # BUT: 'best_thicknesses' is just a list of values corresponding to 'variable_layers_config' order.
                    # The order of 'variable_layers_config' list itself was NOT changed, only the 'index' values inside it.
                    # So best_th[k] still corresponds to the k-th entry in our original loop over the dataframe.
                    
                    # We can just iterate the original DF rows that were marked for optimization
                    
                    opt_indices = [i for i, row in current_opt_df.iterrows() if row['Optimize']]
                    
                    for k, df_idx in enumerate(opt_indices):
                        row = current_opt_df.iloc[df_idx]
                        res_data.append({
                            "Layer": row['Material'],
                            "Initial (nm)": row['Thickness (nm)'],
                            "Optimized (nm)": f"{best_th[k]:.2f}",
                            "Change": f"{best_th[k] - row['Thickness (nm)']:.2f}"
                        })
                        
                    st.table(pd.DataFrame(res_data))
                    
                    # --- LOGGING ---
                    try:
                        history_file = "optimization_history.csv"
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Create summary string for thicknesses
                        th_summary = "; ".join([f"{d['Layer']}: {d['Optimized (nm)']}" for d in res_data])
                        
                        new_record = {
                            "Timestamp": timestamp,
                            "Score": f"{m['Total_Score']:.4f}",
                            "S1": f"{m['S1']:.2f}",
                            "S2": f"{m['S2']:.2f}",
                            "C1": f"{m['C1']:.2f}",
                            "C2": f"{m['C2']:.2f}",
                            "Thicknesses": th_summary
                        }
                        
                        df_log = pd.DataFrame([new_record])
                        
                        if not os.path.exists(history_file):
                            df_log.to_csv(history_file, index=False)
                        else:
                            df_log.to_csv(history_file, mode='a', header=False, index=False)
                            
                        st.success(f"Result saved to {history_file}")
                        
                    except Exception as e:
                        st.warning(f"Could not save history: {e}")
                        
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    st.exception(e)

        # --- HISTORY VIEWER ---
        st.divider()
        with st.expander("Optimization History"):
            history_file = "optimization_history.csv"
            if os.path.exists(history_file):
                hist_df = pd.read_csv(history_file)
                st.dataframe(hist_df, width="stretch")
                
                if st.button("Clear History"):
                    os.remove(history_file)
                    st.rerun()
            else:
                st.info("No history found.")

if __name__ == "__main__":
    main()
