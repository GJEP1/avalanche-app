# Avalanche Modeling Application

A web-based GUI for rock avalanche hazard assessment, integrating SLBL failure surface generation with AvaFrame runout simulations and probabilistic analysis.

## Features

- **Scenario Editor**: Interactive map for drawing and managing scenario polygons and cross-section lines
- **SLBL Generator**: Generate potential failure surfaces using the Sloping Local Base Level method
- **Simulation**: Run AvaFrame rock avalanche simulations with Voellmy friction model
- **Results Explorer**: Interactive visualization with empirical relationship validation (Strom 2019, Brideau 2021)
- **Probability Ensemble**: Generate probabilistic runout maps from ensemble simulations
- **File Manager**: Manage project files and monitor disk usage

## Architecture

```
avalanche-app/
├── app/
│   ├── Home.py                    # Main entry point
│   ├── pages/
│   │   ├── 0_Scenario_Editor.py   # Interactive map editor
│   │   ├── 1_SLBL_Generator.py    # SLBL failure surface generation
│   │   ├── 2_Simulation.py        # AvaFrame simulation runner
│   │   ├── 3_Results_Explorer.py  # Parameter exploration & validation
│   │   ├── 4_Probability_Ensemble.py  # Probabilistic analysis
│   │   └── 5_File_Manager.py      # Project file management
│   ├── core/
│   │   ├── auth.py                # Authentication
│   │   ├── project_manager.py     # Project & data management
│   │   ├── slbl_engine.py         # SLBL algorithm implementation
│   │   ├── avaframe_runner.py     # AvaFrame integration
│   │   ├── job_queue.py           # Background task processing
│   │   └── probability/           # Ensemble & weighting methods
│   └── components/
│       └── ol_map/                # OpenLayers map component
├── analysis/                       # Standalone analysis scripts
├── .streamlit/
│   ├── config.yaml                # Streamlit configuration
│   └── secrets.toml               # Authentication secrets (not in repo)
└── AvaFrame/                       # AvaFrame simulation engine (submodule)
```

## Installation

### Prerequisites

- Python 3.10+
- GDAL libraries
- Redis server (for background tasks)

### Setup

```bash
# Clone the repository
git clone https://github.com/GJEP1/avalanche-app.git
cd avalanche-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install AvaFrame
git clone https://github.com/avaframe/AvaFrame.git
cd AvaFrame && pip install -e . && cd ..

# Configure authentication
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your credentials
```

### Configuration

1. **Authentication**: Edit `.streamlit/secrets.toml` with your admin credentials
2. **Data directory**: Create a data directory and symlink: `ln -s /path/to/data data`
3. **Redis**: Ensure Redis is running for background job processing

## Running the Application

```bash
cd avalanche-app
source venv/bin/activate
streamlit run app/Home.py
```

Access at `http://localhost:8501`

## Workflow

1. **Create Project**: Set up a new project with DEM and coordinate system
2. **Define Scenarios**: Draw release area polygons in the Scenario Editor
3. **Generate SLBL**: Create failure thickness surfaces with various e-ratios
4. **Run Simulations**: Execute AvaFrame simulations with parameter sweeps
5. **Analyze Results**: Explore results against empirical relationships
6. **Probability Maps**: Generate ensemble probability maps

## Scientific References

### SLBL Method
- Jaboyedoff, M., et al. (2019). "Testing a failure surface prediction and deposit reconstruction method for a landslide cluster that occurred during Typhoon Talas (Japan)". *Earth Surf. Dynam.*, 7, 439-458.

### Empirical Relationships
- Aaron, J., et al. (2022). "Probabilistic prediction of rock avalanche runout using a numerical model". *Landslides*, 19, 2853-2869.
- Strom, A., Li, L., & Lan, H. (2019). "Rock avalanche mobility: optimal characterization and the effects of confinement". *Landslides*, 16, 1437-1452.
- Brideau, M.-A., et al. (2021). "Empirical Relationships to Estimate the Probability of Runout Exceedance for Various Landslide Types". *WLF5 Proceedings*, pp. 321-327.

### Simulation Engine
- AvaFrame: Open source avalanche simulation framework - https://avaframe.org

## Default Parameters

### SLBL E-ratios
`[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]`

### Voellmy Parameters (typical ranges)
- μ (friction): 0.05 - 0.30
- ξ (turbulence): 100 - 1500 m/s²

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
