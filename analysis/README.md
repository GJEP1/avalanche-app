# Rock Avalanche Runout Explorer

Interactive visualization tool for exploring AVAFRAME rock avalanche simulation parameter sweeps. Compares modeled runouts against empirical relationships from the scientific literature.

![Layout Example](docs/layout_example.png)

## Features

- **Interactive parameter exploration**: Click or use sliders to navigate through μ-ξ parameter combinations
- **Map visualization**: Hillshade + flow thickness overlay with Hmax/Hmin markers
- **Strom's relation** (2019): Compare V×Hmax vs total affected area
- **Brideau's relation** (2021): Compare H/L mobility ratio vs volume
- **Automatic metrics**: Computes H, L, H/L, Atotal, and deviation from empirical relations

## Quick Start

### 1. Install dependencies

```bash
cd avalanche_viz
pip install -r requirements.txt
```

### 2. Run the visualization tool

```bash
streamlit run runout_explorer.py
```

### 3. Load your simulations

Enter the path to either:
- A **single simulation** folder (with `Inputs/`, `Outputs/`, and config file)
- A **parameter sweep** folder containing multiple `mu_X_turb_Y/` subfolders

## Running Parameter Sweeps

If you want to run new simulations across a grid of Voellmy parameters:

```bash
python run_parameter_sweep.py \
    --dem /path/to/your/DEM.asc \
    --rel /path/to/your/release_thickness.asc \
    --mu-min 0.025 --mu-max 0.4 --mu-steps 16 \
    --xi-min 250 --xi-max 2000 --xi-steps 8 \
    --output /mnt/data/my_parameter_sweep \
    --avaframe /app/AvaFrame
```

This will create 128 simulations (16 × 8) with the following structure:

```
my_parameter_sweep/
├── sweep_config.json              # Sweep metadata
├── mu_0.025_turb_250/
│   ├── Inputs/
│   │   ├── DEM.asc
│   │   └── REL/relTh.asc
│   ├── Outputs/com1DFA/peakFiles/
│   └── local_com6RockAvalancheCfg.ini
├── mu_0.025_turb_500/
├── mu_0.050_turb_250/
└── ...
```

## Empirical Relations

### Strom et al. (2019)

**Reference:** Strom, A., L. Li, and H. Lan. 2019. "Rock Avalanche Mobility: Optimal Characterization and the Effects of Confinement." *Landslides* 16(8): 1437-52.

The relation predicts total affected area from volume and drop height:

```
log₁₀(A_total) = 1.0884 + 0.5497 × log₁₀(V × H_max)
```

Where:
- `A_total` = Total affected area (km²)
- `V` = Volume (km³)
- `H_max` = Maximum drop height (km)

The tool shows three regression lines for different confinement types:
- **Frontally confined** (R² = 0.9258)
- **Laterally confined** (R² = 0.9267)
- **Unconfined** (R² = 0.9361)

### Brideau et al. (2021)

**Reference:** Brideau, M.-A., et al. 2021. Global landslide database with H/L vs Volume relationship.

Two populations identified:

**Small/medium events:**
```
log₁₀(H/L) = -0.033 × log₁₀(V) + 0.0315
R² = 0.113, N = 144
```

**Large events:**
```
log₁₀(H/L) = -0.137 × log₁₀(V) + 0.469
R² = 0.431, N = 288
```

Where:
- `H/L` = Mobility ratio (drop height / runout length)
- `V` = Volume (m³)

## Understanding the Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| **μ** | Voellmy friction coefficient | - |
| **ξ** | Voellmy turbulence coefficient | m/s² |
| **V** | Release volume | m³ |
| **H_max** | Highest elevation with significant flow | m |
| **H_min** | Lowest elevation with flow (runout toe) | m |
| **H** | Drop height (H_max - H_min) | m |
| **L** | Planimetric runout length | m |
| **H/L** | Mobility ratio (lower = more mobile) | - |
| **A_total** | Total area affected by flow | km² |
| **AE_Strom** | Absolute error from Strom's relation | log₁₀(km²) |

## Interpreting Results

### AE_Strom (Strom Absolute Error)

This metric tells you how well a simulation matches observed rock avalanche behavior:

| AE_Strom | Interpretation |
|----------|---------------|
| < 0.2 | ✅ Good match - parameters are realistic |
| 0.2 - 0.5 | ⚠️ Moderate match - consider alternatives |
| > 0.5 | ❌ Poor match - parameters likely unrealistic |

### Parameter Selection Guidance

- **Lower μ, higher ξ**: More mobile flow, longer runout
- **Higher μ, lower ξ**: Less mobile, shorter runout
- Rock avalanches typically: μ ≈ 0.05-0.2, ξ ≈ 500-2000

### Confinement Effects

If your simulation shows more area than expected for frontal confinement but less than unconfined, your site is likely laterally confined.

## Integration with Existing App

To integrate with your existing `main_map_new.py`:

```python
# In your main app, add a new page
import streamlit as st

st.sidebar.page_link("main_map_new.py", label="Run Simulation")
st.sidebar.page_link("runout_explorer.py", label="Explore Results")
```

Or import the visualization components directly:

```python
from runout_explorer import (
    scan_simulation_batch,
    create_map_figure,
    create_strom_plot,
    create_brideau_plot,
)

# Load batch
batch = scan_simulation_batch(Path("/mnt/data/my_sweep"))

# Get a specific simulation
sim = batch.simulations[(0.1, 1500)]  # μ=0.1, ξ=1500

# Create plots
map_fig = create_map_figure(batch, sim)
strom_fig = create_strom_plot(batch, sim)
```

## Troubleshooting

### "No simulation folders found"

Check that folder names follow the pattern: `mu_X.XXX_turb_XXXX`

Examples:
- ✅ `mu_0.100_turb_1500`
- ✅ `mu_0.05_turb_500`
- ❌ `simulation_1`
- ❌ `mu0.1_xi1500`

### "No peak thickness file found"

Ensure simulations completed successfully and have:
```
Outputs/com1DFA/peakFiles/*_pft.asc
```

### Memory issues with large sweeps

For very large DEMs or many simulations, you may need to:
1. Reduce DEM resolution
2. Use a machine with more RAM
3. Process in batches

## File Formats

### Input: ESRI ASCII Grid (.asc)

```
ncols         844
nrows         869
xllcorner     123456.0
yllcorner     7891234.0
cellsize      10.0
NODATA_value  -9999
<space-separated elevation data>
```

### Output: sweep_config.json

```json
{
  "dem": "/path/to/DEM.asc",
  "release": "/path/to/relTh.asc",
  "mu_values": [0.025, 0.05, ...],
  "xi_values": [250, 500, ...],
  "total_simulations": 128,
  "created": "2026-01-26T12:00:00"
}
```

## Contributing

Feel free to extend this tool! Some ideas:
- Add center-of-mass trajectory visualization
- Include energy line contours
- Export results to GeoJSON/GeoPackage
- Add comparison with field observations
- Implement uncertainty quantification

## License

Internal use - MTLab

## References

1. Strom, A., L. Li, and H. Lan. 2019. "Rock Avalanche Mobility: Optimal Characterization and the Effects of Confinement." *Landslides* 16(8): 1437-52. https://doi.org/10.1007/s10346-019-01181-z

2. Brideau, M.-A., et al. 2021. Version 2.2, 28 Jan 2020. Global landslide database.

3. AVAFRAME Documentation: https://docs.avaframe.org/
