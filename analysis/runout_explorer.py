"""
Rock Avalanche Runout Explorer
==============================
Interactive visualization tool for exploring AVAFRAME rock avalanche simulation 
parameter sweeps. Compares modeled runouts against empirical relationships 
(Strom et al. 2019, Brideau et al. 2021).

Usage:
    streamlit run runout_explorer.py

Author: Based on MATLAB tool by colleague, Python port for web accessibility
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import configparser
from functools import lru_cache
import colorsys
import shutil
import tarfile
import tempfile

# Check Streamlit version for on_select support (requires Streamlit 1.29+)
try:
    STREAMLIT_VERSION = tuple(map(int, st.__version__.split('.')[:2]))
    HAS_ON_SELECT = STREAMLIT_VERSION >= (1, 29)
except:
    HAS_ON_SELECT = False


# Optional authentication integration
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.main_map_new import load_authentication
    HAS_AUTH = True
except:
    HAS_AUTH = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimulationResult:
    """Container for a single simulation's results and metadata."""
    folder: Path
    mu: float  # Friction coefficient
    xi: float  # Turbulence coefficient (ξ)
    
    # Computed values (populated by processing)
    volume_m3: float = 0.0  # Release volume in m³
    hmax_m: float = 0.0     # Maximum elevation with flow (m)
    hmin_m: float = 0.0     # Minimum elevation with flow (m)
    delta_h_m: float = 0.0  # H = hmax - hmin (m)
    runout_l_m: float = 0.0 # Runout length L (m)
    h_over_l: float = 0.0   # H/L ratio (dimensionless)
    atot_km2: float = 0.0   # Total affected area (km²)
    v_times_h_km4: float = 0.0  # V × Hmax product (km⁴)
    
    # Coordinates of key points
    hmax_coords: Tuple[float, float] = (0.0, 0.0)  # (x, y) of highest flow point
    hmin_coords: Tuple[float, float] = (0.0, 0.0)  # (x, y) of lowest flow point
    
    # Strom's relation error
    ae_strom: float = 0.0  # Absolute error from Strom's relation (log10 scale)
    
    # Raster data (loaded on demand)
    thickness_array: Optional[np.ndarray] = None
    dem_array: Optional[np.ndarray] = None
    hillshade_array: Optional[np.ndarray] = None
    
    # Georeferencing
    bounds: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    cellsize: float = 10.0


@dataclass 
class SimulationBatch:
    """Container for a batch of simulations (parameter sweep)."""
    base_folder: Path
    simulations: Dict[Tuple[float, float], SimulationResult] = field(default_factory=dict)
    
    # Unique parameter values found
    mu_values: List[float] = field(default_factory=list)
    xi_values: List[float] = field(default_factory=list)
    
    # Shared data
    dem_array: Optional[np.ndarray] = None
    hillshade_array: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    cellsize: float = 10.0
    
    # Volume (same for all simulations in a batch)
    volume_m3: float = 0.0
    volume_km3: float = 0.0


# =============================================================================
# Empirical Relations Data
# =============================================================================

def get_strom_relation_data():
    """
    Strom et al. (2019) - Rock Avalanche Mobility: Optimal Characterization 
    and the Effects of Confinement. Landslides 16(8): 1437-52.
    
    Relation: log10(A_total) = 1.0884 + 0.5497 * log10(V * H_max)
    
    Returns data for three confinement types with their regression lines.
    """
    # Generate regression lines
    x_range = np.logspace(-5, 2, 100)  # V × Hmax from 10^-5 to 10^2 km⁴
    
    # Combined relation (all data)
    y_combined = 10 ** (1.0884 + 0.5497 * np.log10(x_range))
    
    # Approximate relations for different confinement types (from figure)
    # These are approximations based on the visual in the paper
    y_frontally = 10 ** (1.05 + 0.55 * np.log10(x_range))
    y_laterally = 10 ** (1.10 + 0.55 * np.log10(x_range))
    y_unconfined = 10 ** (1.15 + 0.55 * np.log10(x_range))
    
    return {
        'x_range': x_range,
        'combined': y_combined,
        'frontally_confined': y_frontally,
        'laterally_confined': y_laterally,
        'unconfined': y_unconfined,
        'r2_frontally': 0.9258,
        'r2_laterally': 0.9267,
        'r2_unconfined': 0.9361,
    }


def get_brideau_relation_data():
    """
    Brideau et al. (2021) - H/L vs Volume relationship from global landslide database.
    
    Two populations identified:
    1. Small/medium: Log(H/L) = -0.033 * Log(V) + 0.0315, R² = 0.113, N = 144
    2. Large: Log(H/L) = -0.137 * Log(V) + 0.469, R² = 0.431, N = 288
    
    Returns regression lines and confidence intervals.
    """
    # Volume range in m³
    v_range = np.logspace(4, 11, 100)  # 10^4 to 10^11 m³
    
    # Small/medium events regression
    log_hl_small = -0.033 * np.log10(v_range) + 0.0315
    hl_small = 10 ** log_hl_small
    
    # Large events regression  
    log_hl_large = -0.137 * np.log10(v_range) + 0.469
    hl_large = 10 ** log_hl_large
    
    # Approximate confidence bounds (±1 order of magnitude spread from figure)
    hl_small_upper = hl_small * 3
    hl_small_lower = hl_small / 3
    hl_large_upper = hl_large * 3
    hl_large_lower = hl_large / 3
    
    return {
        'v_range': v_range,
        'hl_small': hl_small,
        'hl_large': hl_large,
        'hl_small_upper': hl_small_upper,
        'hl_small_lower': hl_small_lower,
        'hl_large_upper': hl_large_upper,
        'hl_large_lower': hl_large_lower,
        'r2_small': 0.113,
        'r2_large': 0.431,
        'n_small': 144,
        'n_large': 288,
    }


# =============================================================================
# Raster Processing Functions
# =============================================================================

def compute_hillshade(dem: np.ndarray, cellsize: float, 
                      azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """
    Compute hillshade from DEM using Horn's method.
    
    Parameters:
        dem: 2D numpy array of elevations
        cellsize: Cell size in map units
        azimuth: Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)
        altitude: Sun altitude angle in degrees above horizon
    
    Returns:
        Hillshade array (0-255)
    """
    # Convert angles to radians
    azimuth_rad = np.radians(360 - azimuth + 90)  # Convert to math convention
    altitude_rad = np.radians(altitude)
    
    # Compute gradients
    dy, dx = np.gradient(dem, cellsize)
    
    # Compute slope and aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    
    # Compute hillshade
    hillshade = (np.cos(altitude_rad) * np.cos(slope) + 
                 np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    
    # Scale to 0-255
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
    
    return hillshade


def read_asc_file(filepath: Path) -> Tuple[np.ndarray, dict]:
    """
    Read ESRI ASCII Grid file.
    
    Returns:
        Tuple of (data array, metadata dict)
    """
    with open(filepath, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            # Handle "nan" as a special case
            if line[1].lower() == 'nan':
                value = np.nan
            else:
                value = float(line[1]) if '.' in line[1] or 'e' in line[1].lower() else int(line[1])
            header[key] = value
        
        # Read data
        data = np.loadtxt(f)
        
    # Handle nodata
    nodata = header.get('nodata_value', -9999)
    data = np.where(data == nodata, np.nan, data)
    
    # Compute bounds - handle both xllcorner and xllcenter variants
    cellsize = header['cellsize']
    if 'xllcorner' in header:
        xmin = header['xllcorner']
        ymin = header['yllcorner']
    else:
        # Convert from center to corner
        xmin = header['xllcenter'] - cellsize / 2
        ymin = header['yllcenter'] - cellsize / 2
    
    xmax = xmin + header['ncols'] * cellsize
    ymax = ymin + header['nrows'] * cellsize
    header['bounds'] = (xmin, ymin, xmax, ymax)
    
    return data, header


def thickness_to_rgba(thickness: np.ndarray, max_thick: float = 25.0) -> np.ndarray:
    """
    Convert thickness array to RGBA image using HSV colormap (like MATLAB code).
    
    Uses rainbow colormap: thick = red (H=0), thin = blue (H=0.833)
    Transparency based on whether there's flow or not.
    """
    # Normalize thickness
    norm_thick = np.clip(thickness / max_thick, 0, 1)
    
    # Create HSV arrays
    # Hue: 0.833 (blue) for thin, 0 (red) for thick
    hue = 0.833 * (1 - norm_thick)
    saturation = np.where(thickness > 0, 0.8, 0)
    value = np.where(thickness > 0, 0.9, 0)
    
    # Convert to RGB
    h, s, v = hue.flatten(), saturation.flatten(), value.flatten()
    rgb = np.array([colorsys.hsv_to_rgb(hi, si, vi) for hi, si, vi in zip(h, s, v)])
    rgb = rgb.reshape((*thickness.shape, 3))
    
    # Add alpha channel (transparent where no flow)
    alpha = np.where(thickness > 0, 0.7, 0)
    
    rgba = np.dstack([rgb, alpha])
    return (rgba * 255).astype(np.uint8)


# =============================================================================
# Simulation Loading and Processing
# =============================================================================

def parse_mu_xi_from_folder(folder_name: str) -> Optional[Tuple[float, float]]:
    """
    Extract μ and ξ values from folder name.
    Expected formats: 'mu_0.100_turb_1500' or 'mu_0.1_xi_1500'
    """
    # Try pattern: mu_X.XXX_turb_XXXX
    match = re.search(r'mu_(\d+\.?\d*)_(?:turb|xi)_(\d+\.?\d*)', folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def parse_mu_xi_from_config(config_path: Path) -> Optional[Tuple[float, float]]:
    """Extract μ and ξ from configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    mu = None
    xi = None
    
    # Check various possible section names
    for section in config.sections():
        if config.has_option(section, 'muvoellmy'):
            mu = config.getfloat(section, 'muvoellmy')
        if config.has_option(section, 'xsivoellmy'):
            xi = config.getfloat(section, 'xsivoellmy')
    
    if mu is not None and xi is not None:
        return mu, xi
    return None


def compute_simulation_metrics(sim: SimulationResult, dem: np.ndarray, 
                               thickness: np.ndarray, header: dict,
                               thickness_threshold: float = 2.5) -> SimulationResult:
    """
    Compute all metrics for a simulation result.
    
    Parameters:
        sim: SimulationResult to populate
        dem: DEM array
        thickness: Peak flow thickness array
        header: Raster metadata
        thickness_threshold: Minimum thickness to consider (default 10% of 25m = 2.5m)
    """
    cellsize = header['cellsize']
    xmin, ymin, xmax, ymax = header['bounds']
    nrows, ncols = thickness.shape
    
    # Create coordinate grids
    x_coords = np.linspace(xmin + cellsize/2, xmax - cellsize/2, ncols)
    y_coords = np.linspace(ymax - cellsize/2, ymin + cellsize/2, nrows)  # Note: flipped
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Mask for significant flow
    flow_mask = thickness >= thickness_threshold
    
    if not np.any(flow_mask):
        return sim
    
    # Total affected area
    sim.atot_km2 = np.sum(flow_mask) * (cellsize ** 2) / 1e6  # km²
    
    # Elevation of flow areas
    dem_masked = np.where(flow_mask, dem, np.nan)
    
    # Find Hmax (highest elevation with flow)
    hmax_idx = np.unravel_index(np.nanargmax(dem_masked), dem_masked.shape)
    sim.hmax_m = dem_masked[hmax_idx]
    sim.hmax_coords = (xx[hmax_idx], yy[hmax_idx])
    
    # Find Hmin (lowest elevation with flow)
    hmin_idx = np.unravel_index(np.nanargmin(dem_masked), dem_masked.shape)
    sim.hmin_m = dem_masked[hmin_idx]
    sim.hmin_coords = (xx[hmin_idx], yy[hmin_idx])
    
    # Compute H and L
    sim.delta_h_m = sim.hmax_m - sim.hmin_m
    dx = sim.hmax_coords[0] - sim.hmin_coords[0]
    dy = sim.hmax_coords[1] - sim.hmin_coords[1]
    sim.runout_l_m = np.sqrt(dx**2 + dy**2)
    
    # H/L ratio
    if sim.runout_l_m > 0:
        sim.h_over_l = sim.delta_h_m / sim.runout_l_m
    
    # Strom's relation: V × Hmax product (in km⁴)
    volume_km3 = sim.volume_m3 / 1e9
    hmax_km = sim.delta_h_m / 1000  # Using delta H, not absolute Hmax
    sim.v_times_h_km4 = volume_km3 * hmax_km
    
    # Error from Strom's relation
    if sim.v_times_h_km4 > 0 and sim.atot_km2 > 0:
        predicted_log_a = 1.0884 + 0.5497 * np.log10(sim.v_times_h_km4)
        actual_log_a = np.log10(sim.atot_km2)
        sim.ae_strom = abs(actual_log_a - predicted_log_a)
    
    return sim


def load_single_simulation(folder: Path, mu: float, xi: float,
                           shared_dem: Optional[np.ndarray] = None,
                           dem_header: Optional[dict] = None) -> SimulationResult:
    """Load and process a single simulation."""
    sim = SimulationResult(folder=folder, mu=mu, xi=xi)
    
    # Find peak thickness file
    peak_files = list(folder.glob('Outputs/com1DFA/peakFiles/*_pft.asc'))
    if not peak_files:
        peak_files = list(folder.glob('Outputs/com1DFA/peakFiles/*_pft.tif'))
    
    if not peak_files:
        st.warning(f"No peak thickness file found in {folder}")
        return sim
    
    # Read thickness
    thickness, header = read_asc_file(peak_files[0])
    sim.thickness_array = thickness
    sim.bounds = header['bounds']
    sim.cellsize = header['cellsize']
    
    # Read or use shared DEM
    if shared_dem is not None:
        dem = shared_dem
    else:
        dem_path = folder / 'Inputs' / 'DEM.asc'
        if dem_path.exists():
            dem, dem_header = read_asc_file(dem_path)
        else:
            st.warning(f"No DEM found for {folder}")
            return sim
    
    sim.dem_array = dem
    
    # Get volume from release area
    rel_path = folder / 'Inputs' / 'REL' / 'relTh.asc'
    if rel_path.exists():
        rel_thickness, rel_header = read_asc_file(rel_path)
        sim.volume_m3 = np.nansum(rel_thickness) * (rel_header['cellsize'] ** 2)
    
    # Compute metrics
    sim = compute_simulation_metrics(sim, dem, thickness, header)
    
    return sim


def scan_simulation_batch(base_folder: Path, 
                          progress_callback=None) -> SimulationBatch:
    """
    Scan a folder containing multiple simulation runs (parameter sweep).
    
    Expected structure:
        base_folder/
        ├── mu_0.025_turb_250/
        ├── mu_0.025_turb_500/
        ├── mu_0.050_turb_250/
        └── ...
    
    OR for single simulations:
        base_folder/  (a single simulation folder)
    """
    batch = SimulationBatch(base_folder=base_folder)
    
    # Check if this is a single simulation or a batch
    if (base_folder / 'Outputs').exists():
        # Single simulation - check for config
        config_path = base_folder / 'local_com6RockAvalancheCfg.ini'
        if config_path.exists():
            params = parse_mu_xi_from_config(config_path)
            if params:
                mu, xi = params
                sim = load_single_simulation(base_folder, mu, xi)
                batch.simulations[(mu, xi)] = sim
                batch.mu_values = [mu]
                batch.xi_values = [xi]
                batch.volume_m3 = sim.volume_m3
                batch.volume_km3 = sim.volume_m3 / 1e9
                batch.dem_array = sim.dem_array
                batch.bounds = sim.bounds
                batch.cellsize = sim.cellsize
                
                if sim.dem_array is not None:
                    batch.hillshade_array = compute_hillshade(sim.dem_array, sim.cellsize)
        return batch
    
    # Batch of simulations
    sim_folders = [f for f in base_folder.iterdir() 
                   if f.is_dir() and parse_mu_xi_from_folder(f.name)]
    
    if not sim_folders:
        st.error("No simulation folders found matching pattern 'mu_X_turb_Y'")
        return batch
    
    # Load shared DEM from first simulation
    first_folder = sim_folders[0]
    dem_path = first_folder / 'Inputs' / 'DEM.asc'
    shared_dem = None
    dem_header = None
    
    if dem_path.exists():
        shared_dem, dem_header = read_asc_file(dem_path)
        batch.dem_array = shared_dem
        batch.bounds = dem_header['bounds']
        batch.cellsize = dem_header['cellsize']
        batch.hillshade_array = compute_hillshade(shared_dem, dem_header['cellsize'])
    
    # Get volume from first simulation
    rel_path = first_folder / 'Inputs' / 'REL'
    rel_files = list(rel_path.glob('*.asc')) if rel_path.exists() else []
    if rel_files:
        rel_data, rel_header = read_asc_file(rel_files[0])
        batch.volume_m3 = np.nansum(rel_data) * (rel_header['cellsize'] ** 2)
        batch.volume_km3 = batch.volume_m3 / 1e9
    
    # Load all simulations
    total = len(sim_folders)
    for i, folder in enumerate(sim_folders):
        if progress_callback:
            progress_callback((i + 1) / total)
        
        params = parse_mu_xi_from_folder(folder.name)
        if params:
            mu, xi = params
            sim = load_single_simulation(folder, mu, xi, shared_dem, dem_header)
            sim.volume_m3 = batch.volume_m3
            batch.simulations[(mu, xi)] = sim
    
    # Extract unique parameter values
    batch.mu_values = sorted(set(mu for mu, xi in batch.simulations.keys()))
    batch.xi_values = sorted(set(xi for mu, xi in batch.simulations.keys()))
    
    return batch


# =============================================================================
# Visualization Functions
# =============================================================================

def create_parameter_heatmap(batch: SimulationBatch, 
                             selected_mu: float, 
                             selected_xi: float,
                             clickable: bool = False) -> go.Figure:
    """Create interactive heatmap of μ-ξ parameter space colored by Strom error."""
    
    # Create grid of AE values
    ae_grid = np.full((len(batch.xi_values), len(batch.mu_values)), np.nan)
    
    for (mu, xi), sim in batch.simulations.items():
        try:
            i = batch.mu_values.index(mu)
            j = batch.xi_values.index(xi)
            ae_grid[j, i] = sim.ae_strom
        except ValueError:
            continue
    
    fig = go.Figure()
    
    # Add heatmap with explicit x and y values
    fig.add_trace(go.Heatmap(
        x=batch.mu_values,
        y=batch.xi_values,
        z=ae_grid,
        colorscale='Turbo',
        colorbar=dict(title='AE Strom<br>(log₁₀)', len=0.8),
        hovertemplate='μ: %{x:.3f}<br>ξ: %{y:.0f}<br>AE: %{z:.3f}<extra></extra>',
    ))
    
    # Add marker for selected simulation
    fig.add_trace(go.Scatter(
        x=[selected_mu],
        y=[selected_xi],
        mode='markers',
        marker=dict(
            size=18,
            color='white',
            line=dict(color='black', width=3),
            symbol='circle',
        ),
        name='Selected',
        hovertemplate=f'μ: {selected_mu:.3f}<br>ξ: {selected_xi:.0f}<extra></extra>',
    ))
    
    title_text = 'Parameter Space — Click to Select' if clickable else 'Parameter Space'
    
    # Calculate axis ranges with padding
    mu_min, mu_max = min(batch.mu_values), max(batch.mu_values)
    xi_min, xi_max = min(batch.xi_values), max(batch.xi_values)
    mu_pad = (mu_max - mu_min) * 0.1 if mu_max > mu_min else 0.01
    xi_pad = (xi_max - xi_min) * 0.1 if xi_max > xi_min else 50
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=dict(
            title='Friction (μ)',
            tickmode='array',
            tickvals=batch.mu_values,
            ticktext=[f'{m:.2f}' for m in batch.mu_values],
            range=[mu_min - mu_pad, mu_max + mu_pad],
        ),
        yaxis=dict(
            title='Turbulence (ξ)',
            tickmode='array',
            tickvals=batch.xi_values,
            ticktext=[f'{x:.0f}' for x in batch.xi_values],
            range=[xi_min - xi_pad, xi_max + xi_pad],
        ),
        height=400,
        margin=dict(l=60, r=20, t=40, b=60),
        showlegend=False,
    )
    
    return fig


def create_strom_plot(batch: SimulationBatch,
                      selected_sim: SimulationResult) -> go.Figure:
    """Create Strom's relation plot with selected simulation point."""
    
    strom_data = get_strom_relation_data()
    
    fig = go.Figure()
    
    # Add regression lines for different confinement types
    fig.add_trace(go.Scatter(
        x=strom_data['x_range'],
        y=strom_data['frontally_confined'],
        mode='lines',
        name=f"Frontally confined (R²={strom_data['r2_frontally']:.3f})",
        line=dict(color='gray', dash='dash'),
    ))
    
    fig.add_trace(go.Scatter(
        x=strom_data['x_range'],
        y=strom_data['laterally_confined'],
        mode='lines',
        name=f"Laterally confined (R²={strom_data['r2_laterally']:.3f})",
        line=dict(color='magenta', dash='dash'),
    ))
    
    fig.add_trace(go.Scatter(
        x=strom_data['x_range'],
        y=strom_data['unconfined'],
        mode='lines',
        name=f"Unconfined (R²={strom_data['r2_unconfined']:.3f})",
        line=dict(color='green', dash='dash'),
    ))
    
    # Add all simulation points (light gray)
    all_x = [sim.v_times_h_km4 for sim in batch.simulations.values() if sim.v_times_h_km4 > 0]
    all_y = [sim.atot_km2 for sim in batch.simulations.values() if sim.atot_km2 > 0]
    
    fig.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode='markers',
        marker=dict(size=6, color='lightgray', opacity=0.5),
        name='All simulations',
        hoverinfo='skip',
    ))
    
    # Add selected simulation point
    if selected_sim.v_times_h_km4 > 0:
        fig.add_trace(go.Scatter(
            x=[selected_sim.v_times_h_km4],
            y=[selected_sim.atot_km2],
            mode='markers',
            marker=dict(
                size=15,
                color='cyan',
                line=dict(color='black', width=2),
            ),
            name='Selected',
            hovertemplate=(
                f'V×H: {selected_sim.v_times_h_km4:.4f} km⁴<br>'
                f'A_tot: {selected_sim.atot_km2:.2f} km²<extra></extra>'
            ),
        ))
    
    fig.update_layout(
        title=dict(text='Strom et al. (2019)', font=dict(size=14)),
        xaxis_title='V × H_max (km⁴)',
        yaxis_title='Total Area (km²)',
        xaxis_type='log',
        yaxis_type='log',
        xaxis=dict(range=[-5, 2]),
        yaxis=dict(range=[-2, 3]),
        height=300,
        margin=dict(l=60, r=20, t=40, b=60),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.4,
            xanchor='center',
            x=0.5,
            font=dict(size=9),
        ),
        showlegend=True,
    )
    
    return fig


def create_brideau_plot(batch: SimulationBatch,
                        selected_sim: SimulationResult) -> go.Figure:
    """Create Brideau's H/L vs Volume plot."""
    
    brideau_data = get_brideau_relation_data()
    
    fig = go.Figure()
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([brideau_data['v_range'], brideau_data['v_range'][::-1]]),
        y=np.concatenate([brideau_data['hl_large_upper'], brideau_data['hl_large_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Large events range',
        showlegend=False,
    ))
    
    # Add regression lines
    fig.add_trace(go.Scatter(
        x=brideau_data['v_range'],
        y=brideau_data['hl_small'],
        mode='lines',
        name=f"Small/medium (R²={brideau_data['r2_small']:.3f}, N={brideau_data['n_small']})",
        line=dict(color='blue', dash='dash'),
    ))
    
    fig.add_trace(go.Scatter(
        x=brideau_data['v_range'],
        y=brideau_data['hl_large'],
        mode='lines',
        name=f"Large events (R²={brideau_data['r2_large']:.3f}, N={brideau_data['n_large']})",
        line=dict(color='red'),
    ))
    
    # Add all simulation points
    all_x = [sim.volume_m3 for sim in batch.simulations.values() if sim.h_over_l > 0]
    all_y = [sim.h_over_l for sim in batch.simulations.values() if sim.h_over_l > 0]
    
    fig.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode='markers',
        marker=dict(size=6, color='lightgray', opacity=0.5),
        name='All simulations',
        hoverinfo='skip',
    ))
    
    # Add selected point
    if selected_sim.h_over_l > 0:
        fig.add_trace(go.Scatter(
            x=[selected_sim.volume_m3],
            y=[selected_sim.h_over_l],
            mode='markers',
            marker=dict(
                size=15,
                color='cyan',
                line=dict(color='black', width=2),
            ),
            name='Selected',
            hovertemplate=(
                f'V: {selected_sim.volume_m3:.2e} m³<br>'
                f'H/L: {selected_sim.h_over_l:.3f}<extra></extra>'
            ),
        ))
    
    fig.update_layout(
        title=dict(text='Brideau et al. (2021)', font=dict(size=14)),
        xaxis_title='Volume (m³)',
        yaxis_title='H/L',
        xaxis_type='log',
        yaxis_type='log',
        xaxis=dict(range=[4, 11]),
        yaxis=dict(range=[-2, 0.7]),
        height=300,
        margin=dict(l=60, r=20, t=40, b=60),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.4,
            xanchor='center',
            x=0.5,
            font=dict(size=9),
        ),
    )
    
    return fig


def create_map_figure(batch: SimulationBatch,
                      selected_sim: SimulationResult,
                      max_thickness: float = 25.0) -> go.Figure:
    """Create map visualization with hillshade and flow thickness overlay."""
    
    if batch.hillshade_array is None or selected_sim.thickness_array is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    xmin, ymin, xmax, ymax = batch.bounds
    
    fig = go.Figure()
    
    # Hillshade as base layer
    fig.add_trace(go.Heatmap(
        z=batch.hillshade_array,
        x0=xmin,
        dx=batch.cellsize,
        y0=ymax,  # Note: y starts from top
        dy=-batch.cellsize,
        colorscale='Greys',
        showscale=False,
        hoverinfo='skip',
    ))
    
    # Flow thickness overlay (only show where > 0)
    thickness = selected_sim.thickness_array.copy()
    thickness = np.where(thickness < 0.1, np.nan, thickness)  # Hide very thin flow
    
    # Calculate extent of flow for better initial zoom
    flow_mask = ~np.isnan(thickness)
    if flow_mask.any():
        rows, cols = np.where(flow_mask)
        # Add buffer around flow extent
        buffer = 50  # cells
        row_min = max(0, rows.min() - buffer)
        row_max = min(thickness.shape[0], rows.max() + buffer)
        col_min = max(0, cols.min() - buffer)
        col_max = min(thickness.shape[1], cols.max() + buffer)
        
        # Convert to coordinates
        flow_xmin = xmin + col_min * batch.cellsize
        flow_xmax = xmin + col_max * batch.cellsize
        flow_ymax = ymax - row_min * batch.cellsize
        flow_ymin = ymax - row_max * batch.cellsize
    else:
        # Fallback to full extent
        flow_xmin, flow_ymin, flow_xmax, flow_ymax = xmin, ymin, xmax, ymax
    
    # Custom colorscale (blue to red like MATLAB)
    colorscale = [
        [0.0, 'rgb(0, 0, 255)'],      # Blue (thin)
        [0.25, 'rgb(0, 255, 255)'],   # Cyan
        [0.5, 'rgb(0, 255, 0)'],      # Green
        [0.75, 'rgb(255, 255, 0)'],   # Yellow
        [1.0, 'rgb(255, 0, 0)'],      # Red (thick)
    ]
    
    fig.add_trace(go.Heatmap(
        z=thickness,
        x0=xmin,
        dx=batch.cellsize,
        y0=ymax,
        dy=-batch.cellsize,
        colorscale=colorscale,
        zmin=0,
        zmax=max_thickness,
        opacity=0.7,
        colorbar=dict(
            title='Thickness (m)',
            x=1.02,
            len=0.5,
        ),
        hovertemplate='Thickness: %{z:.1f} m<extra></extra>',
    ))
    
    # Add Hmax point (red)
    fig.add_trace(go.Scatter(
        x=[selected_sim.hmax_coords[0]],
        y=[selected_sim.hmax_coords[1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'H_max ({selected_sim.hmax_m:.0f} m)',
        hovertemplate=f'H_max: {selected_sim.hmax_m:.0f} m<extra></extra>',
    ))
    
    # Add Hmin point (blue)
    fig.add_trace(go.Scatter(
        x=[selected_sim.hmin_coords[0]],
        y=[selected_sim.hmin_coords[1]],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'H_min ({selected_sim.hmin_m:.0f} m)',
        hovertemplate=f'H_min: {selected_sim.hmin_m:.0f} m<extra></extra>',
    ))
    
    # Add line from Hmax to Hmin
    fig.add_trace(go.Scatter(
        x=[selected_sim.hmax_coords[0], selected_sim.hmin_coords[0]],
        y=[selected_sim.hmax_coords[1], selected_sim.hmin_coords[1]],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name=f'L = {selected_sim.runout_l_m:.0f} m',
        hoverinfo='skip',
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=(f'μ: {selected_sim.mu:.3f}, ξ: {selected_sim.xi:.0f} | '
                  f'V: {selected_sim.volume_m3/1e6:.2f}×10⁶ m³ | '
                  f'H: {selected_sim.delta_h_m:.0f} m | '
                  f'L: {selected_sim.runout_l_m:.0f} m | '
                  f'A: {selected_sim.atot_km2:.2f} km² | '
                  f'AE_Strom: {selected_sim.ae_strom:.2f}'),
            font=dict(size=12),
        ),
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            title='Easting (m)',
            range=[flow_xmin, flow_xmax],  # Set initial zoom to flow extent
        ),
        yaxis=dict(
            title='Northing (m)',
            range=[flow_ymin, flow_ymax],  # Set initial zoom to flow extent
        ),
        height=700,  # Increased height for better map view
        margin=dict(l=60, r=80, t=60, b=60),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        uirevision='constant',  # CRITICAL: Preserves zoom/pan when switching parameters
    )
    
    return fig


# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Rock Avalanche Runout Explorer",
        layout="wide",
    )
    
    # Optional authentication check
    if HAS_AUTH:
        if not st.session_state.get('authentication_status'):
            st.error("Please log in through the main application first")
            st.info("Run: `streamlit run app/main_map_new.py`")
            st.stop()
        st.sidebar.success(f"Logged in as {st.session_state.get('name')}")
    
    # Custom CSS for compact parameter grid and better appearance
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    /* Compact parameter grid styling */
    .param-grid-container {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px;
        background: rgba(50, 50, 50, 0.3);
        margin-bottom: 1rem;
    }
    .param-grid {
        display: grid;
        gap: 2px;
        font-family: monospace;
        font-size: 11px;
    }
    .param-cell {
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 3px;
        cursor: pointer;
        transition: transform 0.1s;
    }
    .param-cell:hover {
        transform: scale(1.2);
        z-index: 10;
    }
    .param-cell.selected {
        outline: 2px solid white;
        outline-offset: 1px;
    }
    .param-label {
        font-size: 10px;
        color: #888;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Rock Avalanche Runout Explorer")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("Load Simulations")
        
        load_method = st.radio(
            "Load method:",
            ["Select sweep", "Enter folder path", "Upload results (coming soon)"],
            index=0,
        )
        
        if load_method == "Select sweep":
            # Scan for available sweeps
            sweep_dir = Path("/mnt/data/simulations")
            sweep_folders = []
            if sweep_dir.exists():
                sweep_folders = sorted(
                    [f for f in sweep_dir.iterdir() if f.is_dir() and f.name.startswith('sweep_')],
                    key=lambda x: x.name,
                    reverse=True  # Most recent first
                )
            
            if sweep_folders:
                sweep_options = [f.name for f in sweep_folders]
                selected_sweep = st.selectbox(
                    "Available parameter sweeps:",
                    sweep_options,
                    help="Select a parameter sweep to visualize"
                )
                folder_path = str(sweep_dir / selected_sweep)
                
                # Show sweep info
                sweep_path = sweep_dir / selected_sweep
                if sweep_path.exists():
                    # Calculate sweep size
                    total_size = sum(f.stat().st_size for f in sweep_path.rglob('*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    st.caption(f"Size: {size_mb:.1f} MB")
                    
                    # Show disk space
                    import shutil
                    disk_usage = shutil.disk_usage(sweep_dir)
                    free_gb = disk_usage.free / (1024**3)
                    total_gb = disk_usage.total / (1024**3)
                    used_percent = (disk_usage.used / disk_usage.total) * 100
                    st.caption(f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB ({used_percent:.0f}% used)")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", type="primary", use_container_width=True):
                        with st.spinner("Scanning simulations..."):
                            progress = st.progress(0)
                            batch = scan_simulation_batch(
                                Path(folder_path),
                                progress_callback=lambda p: progress.progress(p)
                            )
                            st.session_state['batch'] = batch
                            progress.empty()
                            
                            if batch.simulations:
                                st.success(f"Loaded {len(batch.simulations)} simulation(s)")
                                # Set default selection
                                first_key = list(batch.simulations.keys())[0]
                                st.session_state['selected_mu'] = first_key[0]
                                st.session_state['selected_xi'] = first_key[1]
                            else:
                                st.error("No valid simulations found in this sweep")
                
                with col2:
                    if st.button("Export", use_container_width=True):
                        with st.spinner("Creating archive..."):
                            sweep_path = sweep_dir / selected_sweep
                            # Create tar.gz in temp directory
                            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                                with tarfile.open(tmp.name, 'w:gz') as tar:
                                    tar.add(sweep_path, arcname=selected_sweep)
                                
                                # Read the archive for download
                                with open(tmp.name, 'rb') as f:
                                    archive_data = f.read()
                                
                                # Clean up temp file
                                Path(tmp.name).unlink()
                            
                            st.download_button(
                                label="Download Archive",
                                data=archive_data,
                                file_name=f"{selected_sweep}.tar.gz",
                                mime="application/gzip",
                                use_container_width=True
                            )
                
                st.divider()
                
                # Delete sweep section
                with st.expander("Delete Sweep", expanded=False):
                    st.warning(f"This will permanently delete: **{selected_sweep}**")
                    confirm_delete = st.checkbox("I understand this cannot be undone")
                    
                    if st.button("Delete Sweep", type="secondary", disabled=not confirm_delete):
                        sweep_path = sweep_dir / selected_sweep
                        try:
                            shutil.rmtree(sweep_path)
                            st.success(f"Deleted {selected_sweep}")
                            # Clear loaded batch if it was the deleted sweep
                            if 'batch' in st.session_state:
                                del st.session_state['batch']
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting sweep: {e}")
            else:
                st.warning("No parameter sweeps found in /mnt/data/simulations/")
                st.info("Run a parameter sweep from the main app first")
        
        elif load_method == "Enter folder path":
            folder_path = st.text_input(
                "Simulation folder path:",
                value="/mnt/data/simulations/",
                help="Path to a single simulation or a folder containing multiple simulations"
            )
            
            if st.button("Load Simulations", type="primary"):
                with st.spinner("Scanning simulations..."):
                    progress = st.progress(0)
                    batch = scan_simulation_batch(
                        Path(folder_path),
                        progress_callback=lambda p: progress.progress(p)
                    )
                    st.session_state['batch'] = batch
                    progress.empty()
                    
                    if batch.simulations:
                        st.success(f"Loaded {len(batch.simulations)} simulation(s)")
                        # Set default selection
                        first_key = list(batch.simulations.keys())[0]
                        st.session_state['selected_mu'] = first_key[0]
                        st.session_state['selected_xi'] = first_key[1]
                    else:
                        st.error("No valid simulations found")
        
        st.divider()
        
        # Settings
        st.header("Settings")
        
        # Show Streamlit version info
        st.caption(f"Streamlit v{st.__version__}")
        if not HAS_ON_SELECT:
            st.warning("Upgrade to Streamlit 1.29+ for clickable heatmap")
        
        max_thickness = st.slider(
            "Max thickness for colorscale (m):",
            min_value=5.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
        )
        st.session_state['max_thickness'] = max_thickness
        
        thickness_threshold_pct = st.slider(
            "Min thickness threshold (% of max):",
            min_value=1,
            max_value=50,
            value=10,
            help="Minimum thickness to consider for area/runout calculations"
        )
        st.session_state['thickness_threshold'] = max_thickness * thickness_threshold_pct / 100
    
    # Main content
    if 'batch' not in st.session_state or not st.session_state['batch'].simulations:
        st.info("Enter a simulation folder path in the sidebar to begin.")
        
        # Show example usage
        with st.expander("How to use this tool"):
            st.markdown("""
            ### Expected folder structure
            
            **For parameter sweeps:**
            ```
            parent_folder/
            ├── mu_0.025_turb_250/
            │   ├── Inputs/
            │   │   ├── DEM.asc
            │   │   └── REL/relTh.asc
            │   └── Outputs/com1DFA/peakFiles/*_pft.asc
            ├── mu_0.025_turb_500/
            └── ...
            ```
            
            **For single simulations:**
            ```
            simulation_folder/
            ├── Inputs/
            ├── Outputs/
            └── local_com6RockAvalancheCfg.ini
            ```
            
            ### What gets computed
            
            - **H_max**: Highest elevation with significant flow
            - **H_min**: Lowest elevation with flow (runout toe)
            - **L**: Planimetric runout distance
            - **H/L**: Mobility ratio (lower = more mobile)
            - **A_total**: Total affected area
            - **AE_Strom**: Deviation from Strom's empirical relation
            """)
        return
    
    batch = st.session_state['batch']
    
    # Get current selection
    selected_mu = st.session_state.get('selected_mu', batch.mu_values[0])
    selected_xi = st.session_state.get('selected_xi', batch.xi_values[0])
    
    # Parameter selection - only show if multiple simulations
    if len(batch.simulations) > 1:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Parameter Selection")
            
            # ========================================
            # CLICKABLE HEATMAP using native Streamlit
            # ========================================
            fig = create_parameter_heatmap(batch, selected_mu, selected_xi, clickable=HAS_ON_SELECT)
            
            if HAS_ON_SELECT:
                # Use native Streamlit on_select callback (Streamlit 1.29+)
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="param_heatmap_click",
                    on_select="rerun",
                    selection_mode="points",
                )
                
                # Handle selection from heatmap click
                if event and event.selection and event.selection.points:
                    point = event.selection.points[0]
                    click_x = point.get('x')
                    click_y = point.get('y')
                    
                    if click_x is not None and click_y is not None:
                        # Find nearest mu and xi values
                        closest_mu = min(batch.mu_values, key=lambda m: abs(m - click_x))
                        closest_xi = min(batch.xi_values, key=lambda x: abs(x - click_y))
                        
                        # Check if this combination exists and is different
                        if (closest_mu, closest_xi) in batch.simulations:
                            if closest_mu != selected_mu or closest_xi != selected_xi:
                                st.session_state['selected_mu'] = closest_mu
                                st.session_state['selected_xi'] = closest_xi
                                st.rerun()
            else:
                # Fallback for older Streamlit versions
                st.plotly_chart(fig, use_container_width=True, key="param_heatmap")
                st.caption(f"Streamlit {st.__version__} - upgrade to 1.29+ for click-to-select")
            
            st.divider()
            
            # ========================================
            # SLIDER CONTROLS (Backup/precise control)
            # ========================================
            st.markdown("**Fine Control**")
            
            new_mu = st.select_slider(
                "Friction (μ)",
                options=batch.mu_values,
                value=selected_mu,
                format_func=lambda x: f"{x:.3f}",
                key="slider_mu",
            )
            
            new_xi = st.select_slider(
                "Turbulence (ξ)",
                options=batch.xi_values,
                value=selected_xi,
                format_func=lambda x: f"{x:.0f}",
                key="slider_xi",
            )
            
            # Update from sliders
            if new_mu != selected_mu or new_xi != selected_xi:
                if (new_mu, new_xi) in batch.simulations:
                    st.session_state['selected_mu'] = new_mu
                    st.session_state['selected_xi'] = new_xi
                    st.rerun()
            
            # Show current selection info
            st.markdown(f"**Selected:** μ = {selected_mu:.3f}, ξ = {selected_xi:.0f}")
            
    else:
        # Single simulation
        selected_mu, selected_xi = list(batch.simulations.keys())[0]
        st.session_state['selected_mu'] = selected_mu
        st.session_state['selected_xi'] = selected_xi
        col2 = st.container()
    
    # Get selected simulation
    selected_key = (st.session_state['selected_mu'], st.session_state['selected_xi'])
    if selected_key not in batch.simulations:
        st.warning("Selected parameter combination not found")
        return
    
    selected_sim = batch.simulations[selected_key]
    
    # Main visualization area
    if len(batch.simulations) > 1:
        with col2:
            st.plotly_chart(
                create_map_figure(batch, selected_sim, st.session_state.get('max_thickness', 25)),
                use_container_width=True,
                key='map_view',
            )
    else:
        st.plotly_chart(
            create_map_figure(batch, selected_sim, st.session_state.get('max_thickness', 25)),
            use_container_width=True,
            key='map_view',
        )
    
    # Bottom row: empirical plots and stats
    col_strom, col_brideau, col_stats = st.columns([1, 1, 1])
    
    with col_strom:
        st.plotly_chart(
            create_strom_plot(batch, selected_sim),
            use_container_width=True,
            key='strom_plot',
        )
    
    with col_brideau:
        st.plotly_chart(
            create_brideau_plot(batch, selected_sim),
            use_container_width=True,
            key='brideau_plot',
        )
    
    with col_stats:
        st.subheader("Simulation Statistics")
        
        st.metric("Volume", f"{selected_sim.volume_m3/1e6:.2f} × 10⁶ m³")
        st.metric("Drop Height (H)", f"{selected_sim.delta_h_m:.0f} m")
        st.metric("Runout Length (L)", f"{selected_sim.runout_l_m:.0f} m")
        st.metric("H/L Ratio", f"{selected_sim.h_over_l:.3f}")
        st.metric("Total Area", f"{selected_sim.atot_km2:.2f} km²")
        st.metric("V × H_max", f"{selected_sim.v_times_h_km4:.4f} km⁴")
        st.metric("AE Strom", f"{selected_sim.ae_strom:.3f} log₁₀(km²)")
        
        # Quality indicator
        if selected_sim.ae_strom < 0.2:
            st.success("Good match with Strom's relation")
        elif selected_sim.ae_strom < 0.5:
            st.warning("Moderate match with Strom's relation")
        else:
            st.error("Poor match with Strom's relation")


if __name__ == "__main__":
    main()