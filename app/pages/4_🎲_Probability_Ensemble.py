"""
Probability Map Generator - Streamlit Page
===========================================

Generate probabilistic runout maps from ensemble AvaFrame simulations.

This page allows users to:
1. Select SLBL surfaces to include in the ensemble
2. Configure Voellmy parameter ranges (mu, xi)
3. Choose weighting method (equal vs empirically-calibrated)
4. Submit ensemble job to background queue
5. View completed probability maps

Scientific Basis:
----------------
- Parameter ranges informed by Aaron et al. (2022) back-analyses: mu ~ 0.05-0.25
- Empirical weighting constraints from Strom et al. (2019): Area vs V*H relationships
- Mobility constraints from Brideau et al. (2021): H/L exceedance curves

Outputs:
--------
- Impact probability map
- Threshold exceedance maps (depth, velocity, pressure)
- Percentile maps (P10, P50, P90) for depth, velocity, and pressure
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import io
import zipfile
import plotly.graph_objects as go

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import rasterio for GeoTIFF reading
try:
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Folium for interactive maps with terrain basemap
try:
    import folium
    from folium.raster_layers import ImageOverlay
    from streamlit_folium import st_folium
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import project_selector_sidebar, get_current_project
from core.job_queue import get_job_queue

# =============================================================================
# Visualization Helper Functions
# =============================================================================

def create_folium_probability_map(
    raster_path: Path,
    map_type: str = "probability",
    title: str = ""
) -> folium.Map:
    """
    Create an interactive Folium map with terrain basemap and probability overlay.

    Args:
        raster_path: Path to the GeoTIFF file
        map_type: Type of map for colorscale selection
        title: Map title

    Returns:
        folium.Map object
    """
    if not FOLIUM_AVAILABLE or not RASTERIO_AVAILABLE:
        return None

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata

        # Transform bounds to WGS84 (EPSG:4326) for Folium
        if crs and crs.to_epsg() != 4326:
            bounds_wgs84 = transform_bounds(crs, 'EPSG:4326',
                                            bounds.left, bounds.bottom,
                                            bounds.right, bounds.top)
        else:
            bounds_wgs84 = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Mask nodata and zeros
    masked_data = data.astype(float)
    if nodata is not None:
        masked_data = np.where(data == nodata, np.nan, masked_data)
    masked_data = np.where(masked_data == 0, np.nan, masked_data)

    # Select colormap based on map type
    if "probability" in map_type.lower() or "impact" in map_type.lower():
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'prob', ['blue', 'cyan', 'green', 'yellow', 'red'])
        vmin, vmax = 0, 1
    elif "runout" in map_type.lower():
        # Runout envelope: binary mask (0=outside, 1=inside envelope)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'runout', ['white', 'orangered'])
        vmin, vmax = 0, 1
    elif "depth" in map_type.lower():
        cmap = plt.cm.Blues
        vmin = 0
        vmax = np.nanpercentile(masked_data, 99) if np.any(~np.isnan(masked_data)) else 10
    elif "velocity" in map_type.lower():
        cmap = plt.cm.Purples
        vmin = 0
        vmax = np.nanpercentile(masked_data, 99) if np.any(~np.isnan(masked_data)) else 50
    elif "pressure" in map_type.lower():
        cmap = plt.cm.Oranges
        vmin = 0
        vmax = np.nanpercentile(masked_data, 99) if np.any(~np.isnan(masked_data)) else 100
    elif "exceed" in map_type.lower():
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'exceed', ['darkgreen', 'limegreen', 'yellow', 'orange', 'red'])
        vmin, vmax = 0, 1
    else:
        cmap = plt.cm.viridis
        vmin = np.nanmin(masked_data) if np.any(~np.isnan(masked_data)) else 0
        vmax = np.nanmax(masked_data) if np.any(~np.isnan(masked_data)) else 1

    # Normalize data and apply colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colored = cmap(norm(masked_data))

    # Set alpha channel: transparent where NaN, semi-transparent where data
    alpha = np.where(np.isnan(masked_data), 0, 0.7)
    colored[:, :, 3] = alpha

    # Convert to PIL Image
    img_array = (colored * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='RGBA')

    # Save to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # Create base64 encoded image
    import base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    img_data = f"data:image/png;base64,{img_base64}"

    # Calculate center
    center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
    center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2

    # Create Folium map with terrain basemap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None  # We'll add custom tiles
    )

    # Add terrain basemap (Stamen Terrain or OpenTopoMap)
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
        name='Terrain',
        max_zoom=17
    ).add_to(m)

    # Add satellite option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)

    # Add the probability overlay
    ImageOverlay(
        image=img_data,
        bounds=[[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]],
        opacity=1.0,  # Alpha is already in the image
        name=title or map_type.replace('_', ' ').title()
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Fit bounds
    m.fit_bounds([[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]])

    return m

def compute_hillshade(dem: np.ndarray, cellsize: float,
                      azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """Compute hillshade from DEM using Horn's method."""
    azimuth_rad = np.radians(360 - azimuth + 90)
    altitude_rad = np.radians(altitude)

    dy, dx = np.gradient(dem, cellsize)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    hillshade = (np.cos(altitude_rad) * np.cos(slope) +
                 np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))

    return np.clip(hillshade * 255, 0, 255).astype(np.uint8)


def load_probability_raster(raster_path: Path) -> tuple:
    """
    Load a probability raster (GeoTIFF) and return data with metadata.

    Returns:
        tuple: (data_array, bounds, cellsize, nodata) or (None, None, None, None) on error
    """
    if not RASTERIO_AVAILABLE:
        return None, None, None, None

    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds  # (left, bottom, right, top)
            transform = src.transform
            cellsize = transform.a  # Assuming square pixels
            nodata = src.nodata

            # Convert bounds to (xmin, ymin, xmax, ymax)
            bounds_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)

            return data, bounds_tuple, cellsize, nodata
    except Exception as e:
        st.error(f"Error loading raster: {e}")
        return None, None, None, None


def load_dem_for_ensemble(ensemble_dir: Path, project) -> tuple:
    """
    Load DEM for visualization background.

    Tries to find DEM from the ensemble's simulations or the project.

    Returns:
        tuple: (dem_array, bounds, cellsize) or (None, None, None)
    """
    # First, try to find DEM from simulation outputs
    sim_dir = ensemble_dir / "simulations"
    if sim_dir.exists():
        # Look for any simulation directory with a DEM
        for d in sim_dir.iterdir():
            if d.is_dir():
                dem_path = d / "Inputs" / "DEM.asc"
                if dem_path.exists():
                    try:
                        # Read ASC header
                        header = {}
                        with open(dem_path, 'r') as f:
                            for _ in range(6):
                                line = f.readline().strip().split()
                                key = line[0].lower()
                                try:
                                    value = float(line[1]) if '.' in line[1] else int(line[1])
                                except:
                                    value = line[1]
                                header[key] = value

                        # Read data
                        dem_array = np.loadtxt(dem_path, skiprows=6)

                        cellsize = header.get('cellsize', 10)
                        if 'xllcorner' in header:
                            xmin = header['xllcorner']
                            ymin = header['yllcorner']
                        else:
                            xmin = header.get('xllcenter', 0) - cellsize / 2
                            ymin = header.get('yllcenter', 0) - cellsize / 2

                        xmax = xmin + header.get('ncols', 0) * cellsize
                        ymax = ymin + header.get('nrows', 0) * cellsize
                        bounds = (xmin, ymin, xmax, ymax)

                        return dem_array, bounds, cellsize
                    except Exception:
                        continue

    # Fallback: try project DEM directory
    if hasattr(project, 'dem_dir') and project.dem_dir.exists():
        for dem_file in project.dem_dir.glob("*.tif"):
            try:
                with rasterio.open(dem_file) as src:
                    dem_array = src.read(1)
                    bounds = src.bounds
                    cellsize = src.transform.a
                    return dem_array, (bounds.left, bounds.bottom, bounds.right, bounds.top), cellsize
            except Exception:
                continue

    return None, None, None


def create_probability_map_figure(
    data: np.ndarray,
    bounds: tuple,
    cellsize: float,
    nodata: float,
    dem_array: np.ndarray = None,
    dem_bounds: tuple = None,
    dem_cellsize: float = None,
    map_type: str = "probability",
    title: str = ""
) -> go.Figure:
    """
    Create a plotly figure for probability map visualization.

    Args:
        data: 2D numpy array of map values
        bounds: (xmin, ymin, xmax, ymax)
        cellsize: Pixel size in map units
        nodata: Nodata value to mask
        dem_array: Optional DEM for hillshade background
        dem_bounds: DEM bounds if different from data
        dem_cellsize: DEM cellsize
        map_type: Type of map for colorscale selection
        title: Figure title

    Returns:
        plotly Figure object
    """
    fig = go.Figure()

    xmin, ymin, xmax, ymax = bounds
    nrows, ncols = data.shape

    # Add hillshade background if DEM available
    if dem_array is not None and dem_cellsize is not None:
        hillshade = compute_hillshade(dem_array, dem_cellsize)
        hs_bounds = dem_bounds if dem_bounds else bounds
        hs_xmin, hs_ymin, hs_xmax, hs_ymax = hs_bounds

        fig.add_trace(go.Heatmap(
            z=hillshade,
            x0=hs_xmin, dx=dem_cellsize,
            y0=hs_ymax, dy=-dem_cellsize,
            colorscale='Greys',
            showscale=False,
            hoverinfo='skip'
        ))

    # Mask nodata values
    masked_data = data.copy().astype(float)
    if nodata is not None:
        masked_data = np.where(data == nodata, np.nan, masked_data)

    # Also mask zeros for better visualization
    masked_data = np.where(masked_data == 0, np.nan, masked_data)

    # Select colorscale based on map type
    # Use solid rgb colors with single opacity (same style as Results Explorer)
    if "probability" in map_type.lower() or "impact" in map_type.lower():
        # Probability: 0-1 scale, blue to red
        colorscale = [
            [0.0, 'rgb(0,0,255)'],
            [0.25, 'rgb(0,255,255)'],
            [0.5, 'rgb(0,255,0)'],
            [0.75, 'rgb(255,255,0)'],
            [1.0, 'rgb(255,0,0)']
        ]
        zmin, zmax = 0, 1
        colorbar_title = "Probability"
    elif "runout" in map_type.lower():
        # Runout envelope: binary mask (0=outside, 1=inside envelope)
        # Use solid color for impacted area
        colorscale = [
            [0.0, 'rgba(0,0,0,0)'],
            [0.5, 'rgba(0,0,0,0)'],
            [0.5, 'rgb(255,100,50)'],
            [1.0, 'rgb(255,100,50)']
        ]
        zmin = 0
        zmax = 1
        colorbar_title = "Runout Envelope"
    elif "depth" in map_type.lower():
        # Depth: meters, blue scale
        colorscale = [
            [0.0, 'rgb(200,230,255)'],
            [0.25, 'rgb(100,180,255)'],
            [0.5, 'rgb(50,130,220)'],
            [0.75, 'rgb(30,80,180)'],
            [1.0, 'rgb(10,40,120)']
        ]
        zmin = 0
        zmax = np.nanpercentile(masked_data[~np.isnan(masked_data)], 99) if np.any(~np.isnan(masked_data)) else 10
        colorbar_title = "Depth (m)"
    elif "velocity" in map_type.lower():
        # Velocity: m/s, purple scale
        colorscale = [
            [0.0, 'rgb(230,200,255)'],
            [0.25, 'rgb(180,130,255)'],
            [0.5, 'rgb(140,80,220)'],
            [0.75, 'rgb(100,40,180)'],
            [1.0, 'rgb(60,0,120)']
        ]
        zmin = 0
        zmax = np.nanpercentile(masked_data[~np.isnan(masked_data)], 99) if np.any(~np.isnan(masked_data)) else 50
        colorbar_title = "Velocity (m/s)"
    elif "pressure" in map_type.lower():
        # Pressure: kPa, orange-red scale
        colorscale = [
            [0.0, 'rgb(255,230,200)'],
            [0.25, 'rgb(255,180,130)'],
            [0.5, 'rgb(255,130,80)'],
            [0.75, 'rgb(220,80,40)'],
            [1.0, 'rgb(180,30,0)']
        ]
        zmin = 0
        zmax = np.nanpercentile(masked_data[~np.isnan(masked_data)], 99) if np.any(~np.isnan(masked_data)) else 100
        colorbar_title = "Pressure (kPa)"
    elif "exceed" in map_type.lower():
        # Exceedance probability: green to red
        colorscale = [
            [0.0, 'rgb(0,100,0)'],
            [0.25, 'rgb(100,200,0)'],
            [0.5, 'rgb(255,255,0)'],
            [0.75, 'rgb(255,150,0)'],
            [1.0, 'rgb(255,0,0)']
        ]
        zmin, zmax = 0, 1
        colorbar_title = "P(Exceedance)"
    else:
        # Default
        colorscale = 'Viridis'
        zmin = np.nanmin(masked_data) if np.any(~np.isnan(masked_data)) else 0
        zmax = np.nanmax(masked_data) if np.any(~np.isnan(masked_data)) else 1
        colorbar_title = "Value"

    # Add data layer with single opacity (same as Results Explorer)
    fig.add_trace(go.Heatmap(
        z=masked_data,
        x0=xmin, dx=cellsize,
        y0=ymax, dy=-cellsize,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        opacity=0.7,
        colorbar=dict(title=colorbar_title, x=1.02, len=0.5),
        hovertemplate='Value: %{z:.3f}<extra></extra>'
    ))

    # Calculate view extent (zoom to data)
    valid_mask = ~np.isnan(masked_data)
    if np.any(valid_mask):
        rows_with_data = np.any(valid_mask, axis=1)
        cols_with_data = np.any(valid_mask, axis=0)
        row_indices = np.where(rows_with_data)[0]
        col_indices = np.where(cols_with_data)[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            data_xmin = xmin + col_indices[0] * cellsize
            data_xmax = xmin + (col_indices[-1] + 1) * cellsize
            data_ymax = ymax - row_indices[0] * cellsize
            data_ymin = ymax - (row_indices[-1] + 1) * cellsize

            # Add 20% padding
            pad_x = (data_xmax - data_xmin) * 0.2
            pad_y = (data_ymax - data_ymin) * 0.2

            view_xmin = data_xmin - pad_x
            view_xmax = data_xmax + pad_x
            view_ymin = data_ymin - pad_y
            view_ymax = data_ymax + pad_y
        else:
            view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax
    else:
        view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax

    fig.update_layout(
        title=title,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            title='Easting (m)',
            range=[view_xmin, view_xmax]
        ),
        yaxis=dict(
            title='Northing (m)',
            range=[view_ymin, view_ymax]
        ),
        height=600,
        margin=dict(l=60, r=80, t=60, b=60),
        uirevision='prob_map'
    )

    return fig


def _generate_export_readme(ensemble_path: Path, ensemble_name: str) -> str:
    """
    Generate README documentation for the export package.

    Includes all information needed to understand and reproduce the results.
    """
    # Try to load config and summary for detailed documentation
    config_data = {}
    summary_data = {}

    config_files = list((ensemble_path / "config").glob("config_*.json"))
    if config_files:
        try:
            with open(config_files[0]) as f:
                config_data = json.load(f)
        except:
            pass

    summary_path = ensemble_path / "results" / "summary_report.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary_data = json.load(f)
        except:
            pass

    # Build README content
    readme = f"""# Probability Ensemble Export: {ensemble_name}

## Overview

This archive contains the complete data and configuration for a probabilistic
rock avalanche runout ensemble analysis. All files needed to understand,
verify, and reproduce the results are included.

**Export Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Ensemble Configuration

"""

    if config_data:
        readme += f"""- **Total Simulations:** {config_data.get('total_simulations', 'N/A')}
- **Simulations per SLBL:** {config_data.get('sims_per_slbl', 'N/A')}
- **Sampling Method:** {config_data.get('sampling_method', 'N/A')}
- **Weighting Method:** {config_data.get('weighting_method', 'N/A')}
- **Confinement Type:** {config_data.get('confinement_type', 'N/A')}

### Voellmy Parameter Ranges

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| μ (friction) | {config_data.get('mu_min', 'N/A')} | {config_data.get('mu_max', 'N/A')} | dimensionless |
| ξ (turbulence) | {config_data.get('xi_min', 'N/A')} | {config_data.get('xi_max', 'N/A')} | m/s² |

### Simulation Performance Settings

- **Mass per particle:** {config_data.get('mass_per_part', 'N/A')} kg
- **Release thickness per particle:** {config_data.get('delta_th', 'N/A')} m

### SLBL Surfaces Included

"""
        slbl_surfaces = config_data.get('slbl_surfaces', [])
        if slbl_surfaces:
            readme += "| SLBL ID | Volume (m³) | Height Drop (m) | Scenario |\n"
            readme += "|---------|-------------|-----------------|----------|\n"
            for slbl in slbl_surfaces:
                readme += f"| {slbl.get('slbl_id', 'N/A')} | {slbl.get('volume_m3', 0):,.0f} | {slbl.get('height_drop_m', 0):,.0f} | {slbl.get('scenario_name', 'N/A')} |\n"
            readme += "\n"

    if summary_data:
        readme += f"""## Execution Summary

- **Ensemble ID:** {summary_data.get('ensemble_id', 'N/A')}
- **Status:** {summary_data.get('status', 'N/A')}
- **Created:** {config_data.get('created_at', 'N/A')}
- **Completed:** {summary_data.get('completed_at', 'N/A')}

### Results

"""
        config_summary = summary_data.get('configuration', {})
        readme += f"""- **Successful simulations:** {config_summary.get('successful', 'N/A')} / {config_summary.get('total_simulations', 'N/A')}
- **Success rate:** {config_summary.get('success_rate', 0) * 100:.1f}%

"""
        sim_stats = summary_data.get('simulation_statistics', {})
        if sim_stats:
            runout = sim_stats.get('runout_m', {})
            readme += f"""### Runout Statistics

| Percentile | Runout (m) |
|------------|------------|
| Minimum | {runout.get('min', 0):,.0f} |
| P10 | {runout.get('p10', 0):,.0f} |
| P50 (Median) | {runout.get('p50', 0):,.0f} |
| P90 | {runout.get('p90', 0):,.0f} |
| Maximum | {runout.get('max', 0):,.0f} |

"""

    readme += """## Directory Structure

```
{ensemble_name}/
├── config/
│   ├── config_*.json      # Full ensemble configuration
│   └── samples.json       # All parameter combinations (mu, xi, volume, etc.)
├── weights/
│   └── weights_*.json     # Computed weights for each simulation
├── results/
│   ├── summary_report.json    # Execution summary and statistics
│   ├── impact_probability.tif # P(impact) at each location
│   ├── depth_p*.tif          # Percentile depth maps
│   ├── velocity_p*.tif       # Percentile velocity maps
│   ├── pressure_p*.tif       # Percentile pressure maps
│   └── exceed_*.tif          # Threshold exceedance probability maps
├── simulations/              # (Optional) Individual simulation outputs
│   └── mu_*_xi_*/
│       ├── simulation_config.json
│       └── peakFiles/*.asc
├── README.md                 # This file
└── export_info.json          # Export metadata
```

## Output Maps Description

### Probability Maps
- **impact_probability.tif**: Probability that flow reaches each cell (0-1)

### Percentile Maps
- **depth_p10/p50/p90.tif**: 10th/50th/90th percentile flow depth (meters)
- **velocity_p10/p50/p90.tif**: 10th/50th/90th percentile velocity (m/s)
- **pressure_p10/p50/p90.tif**: 10th/50th/90th percentile impact pressure (kPa)

### Threshold Exceedance Maps
- **exceed_depth_1m.tif**: P(depth > 1 m)
- **exceed_velocity_5ms.tif**: P(velocity > 5 m/s)
- **exceed_pressure_10kpa.tif**: P(pressure > 10 kPa)

## Reproducing Results

To reproduce these results:

1. **Install dependencies:**
   - AvaFrame (https://avaframe.org)
   - Python with numpy, rasterio, scipy

2. **Restore configuration:**
   - Use `config/config_*.json` for ensemble settings
   - Use `config/samples.json` for exact parameter combinations

3. **Run simulations:**
   - For each sample in samples.json, run AvaFrame com1DFA
   - Use the SLBL thickness raster as release area
   - Apply mu and xi values from the sample

4. **Apply weights:**
   - Use `weights/weights_*.json` to weight each simulation
   - Aggregate using weighted percentiles

## Scientific References

- Aaron, J., et al. (2022). Probabilistic prediction of rock avalanche runout.
  *Landslides*, 19, 2853-2869. DOI: 10.1007/s10346-022-01939-y

- Strom, A., Li, L., & Lan, H. (2019). Rock avalanche mobility.
  *Landslides*, 16, 1437-1452. DOI: 10.1007/s10346-019-01181-z

- Brideau, M.-A., et al. (2021). Empirical Relationships for Runout Exceedance.
  *WLF5 Proceedings*, pp. 321-327.

## License and Attribution

This data was generated using the Avalanche Simulation Application.
Please cite the relevant scientific references when using these results.
"""

    return readme.replace("{ensemble_name}", ensemble_name)


# Page configuration
st.set_page_config(
    page_title="Probability Ensemble",
    page_icon="🎲",
    layout="wide"
)

# Authentication
is_authenticated, username, name = require_authentication()
if not is_authenticated:
    st.stop()

show_user_info_sidebar()
project = project_selector_sidebar()

if project is None:
    st.warning("Please select or create a project first.")
    st.stop()

# Import probability core module
try:
    from core.probability import (
        EnsembleConfig,
        EnsembleManager,
        SLBLSurface,
        WeightingMethod,
        ConfinementType,
        SamplingMethod,
        get_weighting_method_description,
        STROM_ATOTAL_VHMAX_COEFFICIENTS,
        BRIDEAU_HL_COEFFICIENTS,
        VOELLMY_REFERENCE_VALUES
    )
    PROBABILITY_CORE_AVAILABLE = True
except ImportError as e:
    PROBABILITY_CORE_AVAILABLE = False
    st.error(f"Could not import probability_core: {e}")

# Page title
st.title("🎲 Probability Ensemble")
st.caption(f"Project: {project.name}")

# About section
with st.expander("About Probabilistic Runout Mapping", expanded=False):
    st.markdown("""
    ### Scientific Foundation

    This tool generates probabilistic runout maps using ensemble simulations with
    variable failure volumes (derived from SLBL surfaces) and Voellmy friction
    parameters (μ, ξ).

    #### Core Methodology

    The approach follows the ensemble probabilistic framework informed by:

    1. **Aaron et al. (2022)** - *Landslides* 19:2853-2869 [(DOI)](https://doi.org/10.1007/s10346-022-01939-y)
       - Bayesian-inspired framework for probabilistic weighting and combination of ensemble simulation results
       - Empirical relationships used as probabilistic constraints
       - Parameter ranges: μ = 0.05-0.25, ξ = 100-2000 m/s²

    2. **Strom et al. (2019)** - *Landslides* 16:1437-1452 [(DOI)](https://doi.org/10.1007/s10346-019-01181-z)
       - Empirical A_total vs V×H_max relationships from 595 rock avalanches
       - High predictive power (R² > 0.92) across confinement types

    3. **Brideau et al. (2021)** - *WLF5 Proceedings* [(ResearchGate)](https://www.researchgate.net/publication/347828445_Empirical_Relationships_to_Estimate_the_Probability_of_Runout_Exceedance_for_Various_Landslide_Types)
       - H/L vs Volume probability of exceedance curves
       - Regression based on 288 rock avalanche cases

    #### Outputs

    - **Impact probability map**: Probability that flow reaches each location
    - **Threshold exceedance maps**: P(depth > 1m), P(velocity > 5 m/s), etc.
    - **Percentile maps**: 10th/50th/90th percentile depth, velocity, and impact pressure

    #### Weighting Methods

    - **Equal weights**: conservative baseline assuming no prior information - all simulations treated equally
    - **Strom (2019)**: weighting based on consistency with A_total vs V×H_max empirical relationships
    - **Brideau (2021)**: weighting based on consistency with H/L vs volume empirical relationships
    - **Combined**: geometric mean of Strom and Brideau weights (conceptually similar to Bayesian model averaging)
    """)

# Tabs
tab_config, tab_results, tab_references = st.tabs([
    "Configure Ensemble",
    "View Results",
    "References"
])

# TAB 1: CONFIGURE ENSEMBLE
with tab_config:
    if not PROBABILITY_CORE_AVAILABLE:
        st.error("Probability core module not available. Please check installation.")
        st.stop()

    # ----- 1. SELECT SLBL SURFACES -----
    st.subheader("1. Select SLBL Surfaces")

    st.markdown("""
    Select the SLBL-generated thickness rasters to include in the ensemble.
    Each surface represents a different potential failure volume.
    """)

    # Find available SLBL results
    slbl_dir = project.slbl_dir
    available_slbl = []

    if slbl_dir.exists():
        # First try to load from slbl_results.json (preferred method)
        results_json = slbl_dir / "slbl_results.json"
        if results_json.exists():
            try:
                with open(results_json) as f:
                    slbl_results = json.load(f)
                for result in slbl_results:
                    if result.get("success", False) and result.get("thickness_path"):
                        thickness_path = Path(result["thickness_path"])
                        if thickness_path.exists():
                            # Use max_depth as proxy for height drop if not available
                            # In practice, height_drop should be calculated from DEM
                            height_drop = result.get("height_drop_m", result.get("max_depth_m", 0) * 10)
                            available_slbl.append({
                                "path": str(thickness_path),
                                "scenario": result.get("scenario_name", "Unknown"),
                                "cross_section": result.get("label", thickness_path.stem),
                                "filename": thickness_path.name,
                                "volume_m3": result.get("volume_m3", 0),
                                "height_drop_m": height_drop,
                                "slbl_id": result.get("label", thickness_path.stem),
                                "e_ratio": result.get("e_ratio", 0),
                                "max_depth_m": result.get("max_depth_m", 0),
                                "area_m2": result.get("area_m2", 0)
                            })
            except Exception as e:
                st.warning(f"Could not load slbl_results.json: {e}")

        # Fallback: scan for thickness files directly
        if not available_slbl:
            for thickness_file in slbl_dir.glob("*_thickness.tif"):
                available_slbl.append({
                    "path": str(thickness_file),
                    "scenario": thickness_file.stem.split("_")[0],
                    "cross_section": thickness_file.stem,
                    "filename": thickness_file.name,
                    "volume_m3": 0,
                    "height_drop_m": 500,  # Default estimate
                    "slbl_id": thickness_file.stem
                })

    if not available_slbl:
        st.warning("""
        No SLBL results found. Please run SLBL analysis first using the SLBL Generator page.

        The SLBL Generator creates thickness rasters representing potential failure surfaces
        at different probability levels (e.g., P10, P50, P90 volumes).
        """)
        st.stop()

    # Create dataframe for display
    slbl_df = pd.DataFrame(available_slbl)
    slbl_df["volume_Mm3"] = slbl_df["volume_m3"] / 1e6

    # Multi-select
    st.markdown("**Available SLBL surfaces:**")

    # Display table with relevant columns
    display_cols = ["scenario", "slbl_id", "volume_Mm3"]
    display_names = ["Scenario", "SLBL Label", "Volume (Mm^3)"]

    # Add e_ratio if available
    if "e_ratio" in slbl_df.columns:
        display_cols.append("e_ratio")
        display_names.append("E-ratio")

    # Add max depth if available
    if "max_depth_m" in slbl_df.columns:
        display_cols.append("max_depth_m")
        display_names.append("Max Depth (m)")

    display_df = slbl_df[display_cols].copy()
    display_df.columns = display_names

    # Use data editor for selection
    display_df.insert(0, "Select", True)

    col_config = {
        "Select": st.column_config.CheckboxColumn(default=True),
        "Volume (Mm^3)": st.column_config.NumberColumn(format="%.2f"),
    }
    if "E-ratio" in display_df.columns:
        col_config["E-ratio"] = st.column_config.NumberColumn(format="%.2f")
    if "Max Depth (m)" in display_df.columns:
        col_config["Max Depth (m)"] = st.column_config.NumberColumn(format="%.1f")

    disabled_cols = [c for c in display_df.columns if c != "Select"]

    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        column_config=col_config,
        disabled=disabled_cols
    )

    selected_indices = edited_df[edited_df["Select"]].index.tolist()

    if not selected_indices:
        st.warning("Please select at least one SLBL surface.")
    else:
        st.success(f"Selected {len(selected_indices)} SLBL surface(s)")

    st.divider()

    # ----- 2. SIMULATION PARAMETERS -----
    st.subheader("2. Simulation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        sims_per_slbl = st.radio(
            "Simulations per SLBL surface:",
            options=[32, 64, 128],
            index=1,
            horizontal=True,
            help="""
            Number of mu-xi parameter combinations to test for each SLBL surface.

            - **32**: Quick assessment
            - **64**: Standard analysis (Recommended)
            - **128**: High resolution

            Based on Quan Luna et al. (2012) Monte Carlo methodology.
            """
        )

        total_sims = len(selected_indices) * sims_per_slbl

        # Runtime estimate based on performance preset (24 parallel workers)
        # Fast: ~15 min/sim, Standard: ~35 min/sim, Accurate: ~80 min/sim
        perf_preset_key = "ensemble_perf_preset"
        if perf_preset_key in st.session_state:
            preset = st.session_state[perf_preset_key]
            if preset == "Fast":
                time_per_sim = 15
            elif preset == "Standard":
                time_per_sim = 35
            else:  # Accurate
                time_per_sim = 80
        else:
            time_per_sim = 35  # Default to Standard estimate
        n_workers = 24
        est_hours = (total_sims * time_per_sim) / n_workers / 60

        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Total Simulations", total_sims)
        with col1b:
            st.metric("Est. Runtime", f"{est_hours:.1f} hrs")
        st.caption(f"Based on {time_per_sim} min/sim with {n_workers} parallel workers")

    with col2:
        sampling_method = st.selectbox(
            "Parameter sampling method:",
            options=["grid", "latin_hypercube"],
            index=0,
            format_func=lambda x: {
                "grid": "Regular Grid",
                "latin_hypercube": "Latin Hypercube Sampling"
            }[x],
            help="""
            How to distribute mu-xi parameter combinations:

            - **Regular Grid**: Even spacing, ensures coverage of extremes
            - **Latin Hypercube**: Space-filling, statistically efficient

            Reference: McKay et al. (1979)
            """
        )

    st.divider()

    # ----- 3. VOELLMY PARAMETERS -----
    st.subheader("3. Voellmy Parameters")

    st.markdown("""
    The **Voellmy-Salm model** describes basal resistance with two parameters:
    - **mu** (Coulomb friction): Dominates at low velocity, controls runout distance
    - **xi** (Turbulence): Dominates at high velocity, controls flow speed

    Ranges based on **Aaron et al. (2022)** analysis of 31 rock avalanches.
    """)

    col1, col2 = st.columns(2)

    with col1:
        mu_range = st.slider(
            "mu (friction coefficient):",
            min_value=0.01,
            max_value=0.50,
            value=(0.05, 0.30),
            step=0.01,
            help="""
            Coulomb friction coefficient (dimensionless).

            - Lower mu -> longer runout
            - Rock avalanches: typically 0.05-0.25
            - All 31 cases in Aaron et al. had mu < 0.25
            """
        )

        # Show equivalent friction angle
        mu_mid = (mu_range[0] + mu_range[1]) / 2
        angle_mid = np.degrees(np.arctan(mu_mid))
        st.caption(f"Equivalent friction angle: {angle_mid:.1f} deg")

    with col2:
        xi_range = st.slider(
            "xi (turbulence, m/s^2):",
            min_value=50,
            max_value=2000,
            value=(100, 1500),
            step=50,
            help="""
            Turbulence coefficient in m/s^2.

            - Higher xi -> faster flow
            - Rock avalanches: typically 200-1000 m/s^2
            - On glaciers: can exceed 1500 m/s^2
            """
        )

    # Show reference values
    with st.expander("Typical parameter ranges from literature and modeling practice"):
        st.caption("These are practical modeling ranges synthesized from back-analyses, sensitivity studies, and runout code conventions.")
        ref_data = []
        for name, values in VOELLMY_REFERENCE_VALUES.items():
            ref_data.append({
                "Type": name.replace("_", " ").title(),
                "μ min": values["mu"]["min"],
                "μ typical": values["mu"]["typical"],
                "μ max": values["mu"]["max"],
                "ξ min": values["xi"]["min"],
                "ξ typical": values["xi"]["typical"],
                "ξ max": values["xi"]["max"],
                "Basis": values["basis"]
            })
        st.dataframe(pd.DataFrame(ref_data), hide_index=True)

    st.divider()

    # ----- 4. WEIGHTING METHOD -----
    st.subheader("4. Weighting Method")

    st.markdown("""
    Choose how to weight simulation results when computing probability maps.
    Empirical weighting down-weights physically implausible simulations.
    """)

    weighting_method = st.radio(
        "Select weighting approach:",
        options=[
            WeightingMethod.EQUAL,
            WeightingMethod.EMPIRICAL_STROM,
            WeightingMethod.EMPIRICAL_BRIDEAU,
            WeightingMethod.COMBINED
        ],
        format_func=lambda x: {
            WeightingMethod.EQUAL: "Equal weights (conservative baseline)",
            WeightingMethod.EMPIRICAL_STROM: "Strom et al. (2019) - Area vs V*H",
            WeightingMethod.EMPIRICAL_BRIDEAU: "Brideau et al. (2021) - H/L mobility",
            WeightingMethod.COMBINED: "Combined (geometric mean)"
        }[x],
        index=0,
        help="See 'References' tab for details on each method"
    )

    # Show description
    st.info(get_weighting_method_description(weighting_method))

    # Confinement type (for empirical methods)
    if weighting_method != WeightingMethod.EQUAL:
        confinement = st.selectbox(
            "Confinement type:",
            options=[
                ConfinementType.LATERAL,
                ConfinementType.FRONTAL,
                ConfinementType.UNCONFINED
            ],
            format_func=lambda x: {
                ConfinementType.LATERAL: "Laterally confined (valley/channel)",
                ConfinementType.FRONTAL: "Frontally confined (blocked by slope)",
                ConfinementType.UNCONFINED: "Unconfined (open slope/fan)"
            }[x],
            index=0,
            help="""
            Confinement type affects empirical mobility relationships.

            From Strom et al. (2019):
            - **Laterally confined**: Flow channelized by valley walls
            - **Frontally confined**: Flow blocked by opposing slope (dam-forming)
            - **Unconfined**: Open slope/fan, can spread laterally
            """
        )

        # Show Strom coefficients for selected confinement
        coef = STROM_ATOTAL_VHMAX_COEFFICIENTS[confinement]
        st.caption(
            f"Strom (2019): log₁₀(A_total) = {coef['a']:.4f}(±{coef.get('a_se', 0):.4f}) + "
            f"{coef['b']:.4f}(±{coef.get('b_se', 0):.4f}) × log₁₀(V×H_max), "
            f"R² = {coef['R2']:.4f}, N = {coef['N']}"
        )
    else:
        confinement = ConfinementType.LATERAL

    st.divider()

    # ----- 5. SIMULATION SETTINGS -----
    st.subheader("5. Simulation Settings")

    with st.expander("Performance Settings", expanded=False):
        perf_preset = st.radio(
            "Performance preset",
            ["Fast", "Standard", "Accurate"],
            index=1,  # Default to Standard
            horizontal=True,
            key="ensemble_perf_preset",
            help="Fast: ~15 min/sim. Standard: ~35 min/sim. Accurate: ~80 min/sim."
        )

        if perf_preset == "Accurate":
            default_mass = 280000.0
            default_delta = 2.0
            default_timeout = 9000  # 2.5 hours for accurate mode
        elif perf_preset == "Standard":
            default_mass = 500000.0
            default_delta = 4.0
            default_timeout = 7200  # 2 hours
        else:  # Fast
            default_mass = 700000.0
            default_delta = 6.0
            default_timeout = 3600  # 1 hour

        col_mass, col_delta = st.columns(2)
        with col_mass:
            mass_per_part = st.number_input(
                "Mass per particle [kg]",
                value=default_mass,
                step=10000.0,
                help="Default: 700,000 (Fast) / 500,000 (Standard) / 280,000 (Accurate)"
            )
        with col_delta:
            delta_th = st.number_input(
                "Release thickness per particle [m]",
                value=default_delta,
                step=0.5,
                help="Default: 6.0 (Fast) / 4.0 (Standard) / 2.0 (Accurate)"
            )

    # ----- 6. SUBMIT -----
    st.subheader("6. Submit Ensemble")

    notes = st.text_area(
        "Notes (optional):",
        placeholder="E.g., Emergency assessment for potential failure scenario",
        help="These notes will be saved with the ensemble for future reference"
    )

    # Summary before submit
    with st.expander("Configuration Summary", expanded=True):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown(f"""
            **SLBL Surfaces:** {len(selected_indices)} selected
            **Simulations per surface:** {sims_per_slbl}
            **Total simulations:** {total_sims}
            **Estimated runtime:** {est_hours:.1f} hours
            """)

        with summary_col2:
            st.markdown(f"""
            **mu range:** {mu_range[0]:.2f} - {mu_range[1]:.2f}
            **xi range:** {xi_range[0]:.0f} - {xi_range[1]:.0f} m/s²
            **Weighting:** {weighting_method.value}
            **Performance:** {perf_preset}
            """)

    # Submit button
    submit_disabled = len(selected_indices) == 0

    # Check if another heavy job is running
    job_queue = get_job_queue()
    queue_status = job_queue.get_queue_status()
    if queue_status.get('exclusive_job_running'):
        st.info("⏳ Note: Another heavy job (sweep or ensemble) is currently running. "
                "Your job will be queued and start automatically when the current job completes.")

    if st.button(
        "Submit Probability Ensemble",
        type="primary",
        disabled=submit_disabled,
        use_container_width=True
    ):
        # Build SLBL surface objects
        slbl_surfaces = []
        for i in selected_indices:
            row = slbl_df.iloc[i]
            # Use height_drop_m if available, otherwise estimate from volume
            # Typical H/L ratio ~0.3-0.5 for rock avalanches
            height_drop = row.get("height_drop_m", 0)
            if height_drop == 0:
                # Rough estimate: H ~ 10 * max_depth for steep terrain
                height_drop = row.get("max_depth_m", 100) * 8

            slbl_surfaces.append(SLBLSurface(
                slbl_id=row["slbl_id"],
                thickness_raster_path=Path(row["path"]),
                volume_m3=row["volume_m3"],
                height_drop_m=height_drop,
                scenario_name=row["scenario"],
                cross_section_name=row.get("cross_section", row["slbl_id"])
            ))

        # Build configuration
        config = EnsembleConfig(
            slbl_surfaces=slbl_surfaces,
            sims_per_slbl=sims_per_slbl,
            sampling_method=SamplingMethod(sampling_method),
            mu_min=mu_range[0],
            mu_max=mu_range[1],
            xi_min=float(xi_range[0]),
            xi_max=float(xi_range[1]),
            weighting_method=weighting_method,
            confinement_type=confinement,
            mass_per_part=mass_per_part,
            delta_th=delta_th,
            sim_timeout=default_timeout,
            notes=notes
        )

        # Submit to job queue (job_queue already obtained above)
        job = job_queue.submit(
            job_type="probability_ensemble",
            project_name=project.name,
            created_by=username,
            params={"config": config.to_dict()}
        )

        # Check if job was queued behind another
        queued_msg = ""
        if queue_status.get('exclusive_job_running'):
            queued_msg = "\n\n⏳ **Status:** Queued — will start when current heavy job completes."

        st.success(f"""
        Ensemble job submitted!

        **Job ID:** {job.id}
        **Total simulations:** {total_sims}{queued_msg}

        Monitor progress in the Simulation page Jobs tab.
        Results will appear in the 'View Results' tab when complete.
        """)

# TAB 2: VIEW RESULTS
with tab_results:
    st.subheader("Completed Probability Ensembles")

    prob_dir = project.probability_dir

    if not prob_dir.exists():
        st.info("No probability ensembles have been run yet.")
        st.markdown("""
        After running an ensemble from the 'Configure Ensemble' tab,
        results will appear here with:

        - **Impact probability map**
        - **Threshold exceedance maps**
        - **Percentile maps** (depth, velocity, pressure)
        - **Summary statistics**
        """)
    else:
        # List completed ensembles
        ensembles = sorted([d for d in prob_dir.iterdir() if d.is_dir()], reverse=True)

        if not ensembles:
            st.info("No completed ensembles found.")
        else:
            selected_ensemble = st.selectbox(
                "Select ensemble to view:",
                options=ensembles,
                format_func=lambda x: x.name
            )

            if selected_ensemble:
                summary_path = selected_ensemble / "results" / "summary_report.json"

                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)

                    # Display summary metrics
                    st.markdown("### Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    config_data = summary.get("configuration", {})

                    with col1:
                        st.metric("Total Simulations", config_data.get("total_simulations", "N/A"))
                    with col2:
                        st.metric("Successful", config_data.get("successful", "N/A"))
                    with col3:
                        success_rate = config_data.get("success_rate", 0)
                        st.metric("Success Rate", f"{success_rate*100:.1f}%")
                    with col4:
                        st.metric("Weighting", config_data.get("weighting_method", "N/A"))

                    # Runout statistics
                    st.markdown("### Runout Length")

                    runout_stats = summary.get("simulation_statistics", {}).get("runout_m", {})

                    if runout_stats:
                        # Display P10, P50, P90 as prominent metrics
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            p10_val = runout_stats.get("p10", 0)
                            st.metric("P10 Runout", f"{p10_val:,.0f} m")
                        with col_r2:
                            p50_val = runout_stats.get("p50", 0)
                            st.metric("P50 Runout (Median)", f"{p50_val:,.0f} m")
                        with col_r3:
                            p90_val = runout_stats.get("p90", 0)
                            st.metric("P90 Runout", f"{p90_val:,.0f} m")

                        # Full statistics table
                        with st.expander("Full runout statistics"):
                            stats_df = pd.DataFrame({
                                "Statistic": ["Minimum", "P10", "Median (P50)", "P90", "Maximum", "Mean"],
                                "Runout (m)": [
                                    runout_stats.get("min", 0),
                                    runout_stats.get("p10", 0),
                                    runout_stats.get("p50", 0),
                                    runout_stats.get("p90", 0),
                                    runout_stats.get("max", 0),
                                    runout_stats.get("mean", 0)
                                ]
                            })
                            st.dataframe(stats_df, hide_index=True)

                    # Output maps
                    st.markdown("### Probability Maps")

                    # Find available map files
                    results_path = selected_ensemble / "results"
                    if results_path.exists():
                        available_tifs = sorted([f.stem for f in results_path.glob("*.tif")])
                    else:
                        available_tifs = summary.get("output_maps", [])

                    if available_tifs:
                        # Organize maps by category for better UX
                        map_categories = {
                            "Impact Probability": [m for m in available_tifs if "impact" in m.lower() or m == "impact_probability"],
                            "Runout Envelope": [m for m in available_tifs if "runout" in m.lower()],
                            "Depth": [m for m in available_tifs if "depth" in m.lower()],
                            "Velocity": [m for m in available_tifs if "velocity" in m.lower()],
                            "Pressure": [m for m in available_tifs if "pressure" in m.lower()],
                            "Exceedance": [m for m in available_tifs if "exceed" in m.lower()],
                        }

                        # Flatten to single list with category info
                        organized_maps = []
                        for cat, maps in map_categories.items():
                            if maps:
                                organized_maps.extend(maps)

                        # Add any uncategorized
                        categorized = set(m for maps in map_categories.values() for m in maps)
                        uncategorized = [m for m in available_tifs if m not in categorized]
                        organized_maps.extend(uncategorized)

                        if not organized_maps:
                            organized_maps = available_tifs

                        map_type = st.selectbox(
                            "Select map to view:",
                            options=organized_maps,
                            format_func=lambda x: x.replace("_", " ").title()
                        )

                        map_path = results_path / f"{map_type}.tif"

                        col_map, col_download = st.columns([4, 1])

                        with col_download:
                            if map_path.exists():
                                with open(map_path, "rb") as f:
                                    st.download_button(
                                        "Download GeoTIFF",
                                        f.read(),
                                        file_name=f"{map_type}.tif",
                                        mime="image/tiff",
                                        use_container_width=True
                                    )

                        with col_map:
                            if map_path.exists() and RASTERIO_AVAILABLE:
                                # Use Folium map with terrain basemap if available
                                if FOLIUM_AVAILABLE:
                                    folium_map = create_folium_probability_map(
                                        raster_path=map_path,
                                        map_type=map_type,
                                        title=map_type.replace('_', ' ').title()
                                    )

                                    if folium_map is not None:
                                        st_folium(
                                            folium_map,
                                            width=None,
                                            height=600,
                                            key=f"prob_map_{map_type}"
                                        )
                                        st.caption("Use layer control (top right) to switch between Terrain and Satellite basemaps")
                                    else:
                                        st.warning("Could not create map visualization.")
                                else:
                                    # Fallback to Plotly visualization
                                    data, bounds, cellsize, nodata = load_probability_raster(map_path)

                                    if data is not None:
                                        dem_array, dem_bounds, dem_cellsize = load_dem_for_ensemble(
                                            selected_ensemble, project
                                        )

                                        fig = create_probability_map_figure(
                                            data=data,
                                            bounds=bounds,
                                            cellsize=cellsize,
                                            nodata=nodata,
                                            dem_array=dem_array,
                                            dem_bounds=dem_bounds,
                                            dem_cellsize=dem_cellsize,
                                            map_type=map_type,
                                            title=f"{map_type.replace('_', ' ').title()}"
                                        )

                                        fig.update_layout(
                                            dragmode='pan',
                                            xaxis=dict(fixedrange=False),
                                            yaxis=dict(fixedrange=False),
                                        )

                                        st.plotly_chart(
                                            fig,
                                            use_container_width=True,
                                            key='prob_map_view',
                                            config={
                                                'scrollZoom': True,
                                                'displayModeBar': True,
                                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                                'displaylogo': False,
                                            }
                                        )
                                        st.caption("Mouse wheel to zoom, drag to pan")
                                    else:
                                        st.warning("Could not load raster data.")
                            elif not RASTERIO_AVAILABLE:
                                st.warning("Install rasterio for map visualization: `pip install rasterio`")
                            else:
                                st.warning(f"Map file not found: {map_path}")
                    else:
                        st.info("No probability maps found in this ensemble.")
                else:
                    # No summary report, but check for result files anyway
                    results_path = selected_ensemble / "results"
                    if results_path.exists():
                        available_tifs = sorted([f.stem for f in results_path.glob("*.tif")])
                        if available_tifs:
                            st.warning("Summary report not found, but result maps are available.")

                            map_type = st.selectbox(
                                "Select map to view:",
                                options=available_tifs,
                                format_func=lambda x: x.replace("_", " ").title()
                            )

                            map_path = results_path / f"{map_type}.tif"

                            col_map, col_download = st.columns([4, 1])

                            with col_download:
                                if map_path.exists():
                                    with open(map_path, "rb") as f:
                                        st.download_button(
                                            "Download GeoTIFF",
                                            f.read(),
                                            file_name=f"{map_type}.tif",
                                            mime="image/tiff",
                                            use_container_width=True
                                        )

                            with col_map:
                                if map_path.exists() and RASTERIO_AVAILABLE:
                                    # Use Folium map with terrain basemap if available
                                    if FOLIUM_AVAILABLE:
                                        folium_map = create_folium_probability_map(
                                            raster_path=map_path,
                                            map_type=map_type,
                                            title=map_type.replace('_', ' ').title()
                                        )

                                        if folium_map is not None:
                                            st_folium(
                                                folium_map,
                                                width=None,
                                                height=600,
                                                key=f"prob_map_nosummary_{map_type}"
                                            )
                                            st.caption("Use layer control (top right) to switch between Terrain and Satellite basemaps")
                                        else:
                                            st.warning("Could not create map visualization.")
                                    else:
                                        # Fallback to Plotly
                                        data, bounds, cellsize, nodata = load_probability_raster(map_path)

                                        if data is not None:
                                            dem_array, dem_bounds, dem_cellsize = load_dem_for_ensemble(
                                                selected_ensemble, project
                                            )

                                            fig = create_probability_map_figure(
                                                data=data,
                                                bounds=bounds,
                                                cellsize=cellsize,
                                                nodata=nodata,
                                                dem_array=dem_array,
                                                dem_bounds=dem_bounds,
                                                dem_cellsize=dem_cellsize,
                                                map_type=map_type,
                                                title=f"{map_type.replace('_', ' ').title()}"
                                            )

                                            fig.update_layout(
                                                dragmode='pan',
                                                xaxis=dict(fixedrange=False),
                                                yaxis=dict(fixedrange=False),
                                            )

                                            st.plotly_chart(
                                                fig,
                                                use_container_width=True,
                                                key='prob_map_view_nosummary',
                                                config={
                                                    'scrollZoom': True,
                                                    'displayModeBar': True,
                                                    'displaylogo': False,
                                                }
                                            )
                                            st.caption("Mouse wheel to zoom, drag to pan")
                        else:
                            st.warning("Summary report not found for this ensemble.")
                    else:
                        st.warning("Summary report not found for this ensemble.")

            # ----- EXPORT SECTION -----
            st.divider()
            st.markdown("### Export Ensemble Data")

            st.markdown("""
            Export all ensemble data for reproducibility and archival purposes.
            The export includes configuration, parameters, weights, and result maps.
            """)

            # Export options
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                include_result_maps = st.checkbox(
                    "Include result maps (GeoTIFFs)",
                    value=True,
                    help="Include all probability and percentile maps"
                )
            with col_opt2:
                include_sim_outputs = st.checkbox(
                    "Include individual simulation outputs",
                    value=False,
                    help="Include peak files from each simulation (larger file size)"
                )

            if st.button("Prepare Export Package", type="primary", use_container_width=True):
                with st.spinner("Preparing export package..."):
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        ensemble_name = selected_ensemble.name

                        # 1. Configuration files
                        config_dir = selected_ensemble / "config"
                        if config_dir.exists():
                            for config_file in config_dir.glob("*.json"):
                                zf.write(config_file, f"{ensemble_name}/config/{config_file.name}")

                        # 2. Weights files
                        weights_dir = selected_ensemble / "weights"
                        if weights_dir.exists():
                            for weights_file in weights_dir.glob("*.json"):
                                zf.write(weights_file, f"{ensemble_name}/weights/{weights_file.name}")

                        # 3. Summary report
                        summary_file = selected_ensemble / "results" / "summary_report.json"
                        if summary_file.exists():
                            zf.write(summary_file, f"{ensemble_name}/results/summary_report.json")

                        # 4. Result maps (GeoTIFFs)
                        if include_result_maps:
                            results_dir = selected_ensemble / "results"
                            if results_dir.exists():
                                for tif_file in results_dir.glob("*.tif"):
                                    zf.write(tif_file, f"{ensemble_name}/results/{tif_file.name}")

                        # 5. Individual simulation outputs (optional)
                        if include_sim_outputs:
                            sim_dir = selected_ensemble / "simulations"
                            if sim_dir.exists():
                                for sim_folder in sim_dir.iterdir():
                                    if sim_folder.is_dir():
                                        # Config files
                                        for cfg in ["simulation_config.json", "metadata.json",
                                                    "local_com6RockAvalancheCfg.ini"]:
                                            cfg_path = sim_folder / cfg
                                            if cfg_path.exists():
                                                zf.write(cfg_path,
                                                        f"{ensemble_name}/simulations/{sim_folder.name}/{cfg}")

                                        # Peak output files
                                        peak_dir = sim_folder / "Outputs" / "com1DFA" / "peakFiles"
                                        if peak_dir.exists():
                                            for asc_file in peak_dir.glob("*.asc"):
                                                zf.write(asc_file,
                                                        f"{ensemble_name}/simulations/{sim_folder.name}/peakFiles/{asc_file.name}")

                        # 6. Create README for reproducibility
                        readme_content = _generate_export_readme(selected_ensemble, ensemble_name)
                        zf.writestr(f"{ensemble_name}/README.md", readme_content)

                        # 7. Export metadata
                        export_meta = {
                            "exported_at": datetime.now().isoformat(),
                            "ensemble_id": ensemble_name,
                            "includes_result_maps": include_result_maps,
                            "includes_simulation_outputs": include_sim_outputs,
                            "export_version": "1.0"
                        }
                        zf.writestr(f"{ensemble_name}/export_info.json",
                                   json.dumps(export_meta, indent=2))

                    zip_buffer.seek(0)
                    zip_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)

                st.success(f"Export package ready ({zip_size_mb:.1f} MB)")

                st.download_button(
                    label=f"Download {ensemble_name}.zip",
                    data=zip_buffer.getvalue(),
                    file_name=f"{ensemble_name}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                # Show what's included
                with st.expander("Package Contents"):
                    st.markdown(f"""
                    **{ensemble_name}.zip** contains:

                    - `config/` - Ensemble configuration and parameter samples
                    - `weights/` - Computed simulation weights
                    - `results/summary_report.json` - Execution summary and statistics
                    {"- `results/*.tif` - Probability and percentile maps" if include_result_maps else ""}
                    {"- `simulations/` - Individual simulation configs and outputs" if include_sim_outputs else ""}
                    - `README.md` - Documentation for reproducing results
                    - `export_info.json` - Export metadata
                    """)

# TAB 3: REFERENCES
with tab_references:
    st.subheader("Scientific References")

    st.markdown("""
    ### Primary Methodological References

    #### Aaron et al. (2022)
    **Probabilistic prediction of rock avalanche runout using a numerical model**
    Aaron, J., McDougall, S., Kowalski, J. et al. *Landslides* 19, 2853-2869. [DOI (Open Access)](https://doi.org/10.1007/s10346-022-01939-y)

    Key contributions:
    - A Bayesian-inspired framework for deriving and probabilistically combining case-specific parameter distributions via similarity weighting
    - Empirical relationships used as probabilistic constraints on simulation ensembles
    - Explicit separation of calibration uncertainty from mechanistic uncertainty, with mechanistic variability producing ~4–5× greater runout uncertainty
    - Effective basal friction values inferred from 31 global rock avalanche back-analyses typically span μ ≈ 0.05–0.25

    ---

    #### Strom, Li & Lan (2019)
    **Rock avalanche mobility: optimal characterization and the effects of confinement**
    *Landslides*, 16, 1437-1452. [DOI (Open Access)](https://doi.org/10.1007/s10346-019-01181-z)

    Key contributions:
    - Area vs V*H relationships from 595 Central Asian rock avalanches
    - Stratified by confinement type (R^2 > 0.92 for all types)
    - V*H product provides optimal mobility characterization

    """)

    # Show Strom coefficients table
    if PROBABILITY_CORE_AVAILABLE:
        st.markdown("**Strom et al. (2019) Regression Coefficients (Table 1, A_total vs V×H_max):**")

        strom_table = []
        for conf_type, coef in STROM_ATOTAL_VHMAX_COEFFICIENTS.items():
            strom_table.append({
                "Confinement": conf_type.value.title(),
                "a (intercept)": f"{coef['a']:.4f} (±{coef.get('a_se', 0):.4f})",
                "b (slope)": f"{coef['b']:.4f} (±{coef.get('b_se', 0):.4f})",
                "R²": f"{coef['R2']:.4f}",
                "N": coef["N"]
            })

        st.dataframe(pd.DataFrame(strom_table), hide_index=True)
        st.caption("""
        **Equation:** log₁₀(A_total) = a + b × log₁₀(V × H_max)

        **Variable definitions:**
        - A_total = total affected area (km²)
        - V = volume (10⁶ m³)
        - H_max = maximum drop height (km)
        - log₁₀ = base-10 logarithm

        Coefficients from Table 1 (A_total vs V × H_max; log₁₀ units as defined in Strom et al., 2019).
        Standard errors shown in parentheses.
        """)

    st.markdown("""
    ---

    #### Brideau et al. (2021)
    **Empirical Relationships to Estimate the Probability of Runout Exceedance for Various Landslide Types**
    *WLF5 Proceedings*, pp. 321-327. [ResearchGate](https://www.researchgate.net/publication/347828445_Empirical_Relationships_to_Estimate_the_Probability_of_Runout_Exceedance_for_Various_Landslide_Types)

    Key contributions:
    - Empirical ΔH/L (Fahrböschung) vs volume relationships with probability of runout exceedance curves
    - Rock avalanche (dry, unglaciated terrain) regression: log(ΔH/L) = −0.137 · log(V) + 0.469
    - Regression based on N = 288 dry / earthquake-triggered rock avalanche cases

    ---

    #### Hungr (1995)
    **A model for the runout analysis of rapid flow slides, debris flows, and avalanches**
    *Canadian Geotechnical Journal*, 32(4), 610-623.

    Key contributions:
    - Development of a depth-averaged dynamic runout model for rapid landslides
    - Frictional–collisional rheology incorporating Coulomb friction and a velocity-squared resistance term
    - Engineering formulation for impact forces consistent with dynamic pressure (∼ ½ ρ v²)
    - Conceptual foundation for later Voellmy-type and DAN/DAN3D runout models

    ---

    ### Additional References

    - **McKay, M. D., Beckman, R. J., & Conover, W. J. (1979)**: A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2), 239–245. [DOI](https://doi.org/10.2307/1268522) | [JSTOR](https://www.jstor.org/stable/1268522)
    - **Mitchell, A., McDougall, S., Nolde, N. et al. (2020)**: Rock avalanche runout prediction using stochastic analysis of a regional dataset. *Landslides* 17, 777–792. [DOI](https://doi.org/10.1007/s10346-019-01331-3)

    """)