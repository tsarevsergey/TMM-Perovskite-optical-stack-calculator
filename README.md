# Perovskite Tandem TMM Analysis

A Python-based Transfer Matrix Method (TMM) application for analyzing optical properties of perovskite tandem solar cell structures.

## Features

- **Material Library**: Pre-loaded optical constants (n, k) for common materials:
  - Transparent conductors: ITO, FTO
  - Transport layers: NiO, SnO2, C60
  - Perovskites: MAPbI3, MAPbBr3, CsPbBr1.3Cl1.7
  - Metals: Ag
  - Other: a-Si, 2PACz

- **Advanced Extrapolation**: Uses pyElli library with Tauc-Lorentz dispersion model for physically accurate extrapolation beyond measured wavelength ranges

- **Interactive Streamlit Interface**:
  - Material viewer with n/k plots
  - Stack editor with drag-and-drop functionality
  - Customizable illumination direction (top/bottom)
  - Multiple colormap options for visualization
  - Real-time absorption inspection at specific wavelengths
  - Downloadable results as CSV

- **Optical Calculations**:
  - Layer-by-layer absorption profiles
  - Reflection and transmission spectra
  - Integrated metrics (total R, T, A percentages)
  - Stacked area visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TMM.git
cd TMM

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit App (Recommended)

```bash
streamlit run app.py
```

### Command Line

```python
from tandem_wrapper import Material, calculate_absorbed_power_per_layer
import numpy as np

# Load materials
ito = Material("ITO", "materials/ITO.csv")
perovskite = Material("MAPbI3", "materials/MAPbI3.csv")

# Define stack: (material, thickness_nm)
stack = [
    (None, float('inf')),  # Air
    (ito, 100),
    (perovskite, 500),
    (None, float('inf'))   # Air
]

# Calculate
wavelengths = np.linspace(300, 1200, 500)
spectrum = np.ones_like(wavelengths)
results = calculate_absorbed_power_per_layer(stack, wavelengths, spectrum)
```

## Project Structure

```
TMM/
├── app.py                    # Streamlit interface
├── tandem_wrapper.py         # Core TMM calculations
├── dispersion_analysis.py    # Tauc-Lorentz extrapolation
├── materials/                # Optical constants database
│   ├── ITO.csv
│   ├── MAPbI3.csv
│   └── ...
└── requirements.txt
```

## Material File Format

CSV files with three columns:
```
Wavelength_nm,n,k
300,2.1,0.05
310,2.0,0.04
...
```

## Dependencies

- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- tmm
- pyElli
- streamlit

## License

MIT License

## Acknowledgments

- Built using the [tmm](https://github.com/sbyrnes321/tmm) library by Steven Byrnes
- Dispersion modeling with [pyElli](https://github.com/PyEllips/pyElli)
