# TMM Perovskite Optical Stack Calculator

A Python and Streamlit application for transfer-matrix-method (TMM) analysis of perovskite and thin-film optical stacks.

Repository: <https://github.com/tsarevsergey/TMM-Perovskite-optical-stack-calculator>

## Features

- Material library with optical constants for common stack materials, including ITO, NiO, SnO2, C60, MAPbI3, MAPbBr3, CsPbBr1.3Cl1.7, CsPbI3, a-Si, cSi, Ag, Ge, P3HT, PTAA, PVC, TiO2, ZnO, PbS, and InAs.
- Interactive Streamlit interface for:
  - viewing material `n` and `k` data
  - building detector stacks
  - switching top or bottom illumination
  - plotting reflection, transmission, and per-layer absorption
  - inspecting absorption near a selected wavelength
  - exporting simulation results as CSV
- Material mixer for approximate bandgap-shifted or blended optical constants.
- Thickness optimizer for two-band detector signal and crosstalk studies.
- Optional Tauc-Lorentz model extrapolation through `pyElli`, with constant extrapolation fallback.

## Installation

Python 3.11 is recommended. Python 3.13 may need to build older NumPy wheels from source because this project currently pins `numpy<2.0.0`.

```bash
git clone https://github.com/tsarevsergey/TMM-Perovskite-optical-stack-calculator.git
cd TMM-Perovskite-optical-stack-calculator
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Usage

### Streamlit App

```bash
streamlit run app.py
```

### Python API

```python
import numpy as np

from tandem_wrapper import Material, calculate_absorbed_power_per_layer

ito = Material("ITO", "materials/ITO.csv")
nio = Material("NiO", "materials/NiO.csv")
mapbi3 = Material("MAPbI3", "materials/MAPbI3.csv")
c60 = Material("C60", "materials/C60.csv")
sno2 = Material("SnO2", "materials/SnO2.csv")

stack = [
    (None, float("inf")),  # air
    (ito, 100),
    (nio, 40),
    (mapbi3, 500),
    (c60, 20),
    (sno2, 40),
    (None, float("inf")),  # air
]

wavelengths = np.linspace(300, 1200, 500)
spectrum = np.ones_like(wavelengths)

results = calculate_absorbed_power_per_layer(stack, wavelengths, spectrum=spectrum)
```

## Project Structure

```text
.
|-- app.py
|-- tandem_wrapper.py
|-- optimizer.py
|-- material_interpolation.py
|-- dispersion_analysis.py
|-- clean_k_values.py
|-- materials/
|   |-- ITO.csv
|   |-- MAPbI3.csv
|   |-- sources.txt
|   `-- ...
|-- saved_stacks/
|-- requirements.txt
|-- CITATION.cff
`-- LICENSE
```

## Material File Format

Material CSV files must contain these columns:

```csv
Wavelength_nm,n,k
300,2.1,0.05
310,2.0,0.04
```

## Notes

- Layer thicknesses are in nanometers.
- The first and last stack entries should normally be semi-infinite ambient/substrate layers, represented by `(None, float("inf"))`.
- The included optical constants come from mixed sources. Review `materials/sources.txt` and verify values before using the results in publication-quality analysis.
- Generated optimization logs are written to `optimization_history.csv`, which is intentionally ignored by git.

## Citation

If you use this software in research, please cite this repository. A machine-readable citation file is provided in `CITATION.cff`.

This project builds on Steven Byrnes' `tmm` package and `pyElli`; cite those upstream tools as appropriate for your work.

## License

This project is licensed for non-commercial use under the PolyForm Noncommercial License 1.0.0. See `LICENSE`.

For commercial use, please contact the repository owner.

## Acknowledgments

- Transfer-matrix calculations use the [`tmm`](https://github.com/sbyrnes321/tmm) library by Steven Byrnes.
- Dispersion modeling uses [`pyElli`](https://github.com/PyEllips/pyElli).
