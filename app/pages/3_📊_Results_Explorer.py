"""
Results Explorer Page
=====================
Interactive visualization of simulation results with empirical relation comparisons.
Adapted from runout_explorer.py for project integration.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json
import rasterio
import zipfile
import io
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import colorsys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import project_selector_sidebar, get_current_project

# Page config
st.set_page_config(
    page_title="Results Explorer",
    page_icon="📊",
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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimulationData:
    """Container for a single simulation's data."""
    sim_id: str
    mu: float
    xi: float
    
    # Metrics
    volume_m3: float = 0.0
    hmax_m: float = 0.0
    hmin_m: float = 0.0
    delta_h_m: float = 0.0
    runout_l_m: float = 0.0
    h_over_l: float = 0.0
    atot_km2: float = 0.0
    v_times_h_km4: float = 0.0
    ae_strom: float = 0.0
    
    max_velocity: float = 0.0
    max_thickness: float = 0.0
    max_pressure: float = 0.0
    
    # Coordinates
    hmax_coords: Tuple[float, float] = (0.0, 0.0)
    hmin_coords: Tuple[float, float] = (0.0, 0.0)
    
    # Raster data (loaded on demand)
    thickness_array: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    cellsize: float = 10.0
    
    # Paths
    thickness_path: str = ""
    velocity_path: str = ""


@dataclass
class SweepData:
    """Container for a parameter sweep."""
    name: str
    path: Path
    simulations: Dict[Tuple[float, float], SimulationData] = field(default_factory=dict)
    
    mu_values: List[float] = field(default_factory=list)
    xi_values: List[float] = field(default_factory=list)
    
    # Shared data
    dem_array: Optional[np.ndarray] = None
    hillshade_array: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    cellsize: float = 10.0
    volume_m3: float = 0.0


# =============================================================================
# Empirical Relations
# =============================================================================

def get_strom_relation():
    """
    Strom et al. (2019) - Rock Avalanche Mobility.
    Relation: log10(A_total) = 1.0884 + 0.5497 * log10(V × H_max)
    """
    x_range = np.logspace(-5, 2, 100)
    y_combined = 10 ** (1.0884 + 0.5497 * np.log10(x_range))
    
    return {
        'x_range': x_range,
        'y_combined': y_combined,
        'equation': 'log₁₀(A) = 1.0884 + 0.5497 × log₁₀(V×H)',
        'r2': 0.93
    }


def get_brideau_relation():
    """
    Brideau et al. (2021) - H/L vs Volume relationship.
    Two populations: small/medium and large events.
    """
    v_range = np.logspace(4, 11, 100)
    
    # Small/medium: Log(H/L) = -0.033 * Log(V) + 0.0315
    hl_small = 10 ** (-0.033 * np.log10(v_range) + 0.0315)
    
    # Large: Log(H/L) = -0.137 * Log(V) + 0.469
    hl_large = 10 ** (-0.137 * np.log10(v_range) + 0.469)
    
    return {
        'v_range': v_range,
        'hl_small': hl_small,
        'hl_large': hl_large,
        'r2_small': 0.113,
        'r2_large': 0.431
    }


# =============================================================================
# Data Loading
# =============================================================================

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


def read_asc_header(filepath: Path) -> dict:
    """Read header from ASC file."""
    header = {}
    with open(filepath, 'r') as f:
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            try:
                value = float(line[1]) if '.' in line[1] else int(line[1])
            except:
                value = line[1]
            header[key] = value
    
    cellsize = header.get('cellsize', 10)
    if 'xllcorner' in header:
        xmin = header['xllcorner']
        ymin = header['yllcorner']
    else:
        xmin = header.get('xllcenter', 0) - cellsize / 2
        ymin = header.get('yllcenter', 0) - cellsize / 2
    
    xmax = xmin + header.get('ncols', 0) * cellsize
    ymax = ymin + header.get('nrows', 0) * cellsize
    header['bounds'] = (xmin, ymin, xmax, ymax)
    
    return header


def compute_simulation_metrics(sim: SimulationData, dem: np.ndarray,
                               thickness: np.ndarray, header: dict,
                               thickness_threshold: float = 0.5) -> SimulationData:
    """Compute runout metrics for a simulation."""
    cellsize = header.get('cellsize', 10)
    bounds = header['bounds']
    nrows, ncols = thickness.shape
    
    # Create coordinate grids
    x_coords = np.linspace(bounds[0] + cellsize/2, bounds[2] - cellsize/2, ncols)
    y_coords = np.linspace(bounds[3] - cellsize/2, bounds[1] + cellsize/2, nrows)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flow mask
    flow_mask = thickness >= thickness_threshold
    
    if not np.any(flow_mask):
        return sim
    
    # Total area
    sim.atot_km2 = np.sum(flow_mask) * (cellsize ** 2) / 1e6
    
    # Elevation analysis
    dem_masked = np.where(flow_mask, dem, np.nan)
    
    # Hmax (highest with flow)
    hmax_idx = np.unravel_index(np.nanargmax(dem_masked), dem_masked.shape)
    sim.hmax_m = float(dem_masked[hmax_idx])
    sim.hmax_coords = (float(xx[hmax_idx]), float(yy[hmax_idx]))
    
    # Hmin (lowest with flow)
    hmin_idx = np.unravel_index(np.nanargmin(dem_masked), dem_masked.shape)
    sim.hmin_m = float(dem_masked[hmin_idx])
    sim.hmin_coords = (float(xx[hmin_idx]), float(yy[hmin_idx]))
    
    # H and L
    sim.delta_h_m = sim.hmax_m - sim.hmin_m
    dx = sim.hmax_coords[0] - sim.hmin_coords[0]
    dy = sim.hmax_coords[1] - sim.hmin_coords[1]
    sim.runout_l_m = np.sqrt(dx**2 + dy**2)
    
    # H/L ratio
    if sim.runout_l_m > 0:
        sim.h_over_l = sim.delta_h_m / sim.runout_l_m
    
    # Strom's relation
    volume_km3 = sim.volume_m3 / 1e9
    hmax_km = sim.delta_h_m / 1000
    sim.v_times_h_km4 = volume_km3 * hmax_km
    
    # AE Strom error
    if sim.v_times_h_km4 > 0 and sim.atot_km2 > 0:
        predicted = 1.0884 + 0.5497 * np.log10(sim.v_times_h_km4)
        actual = np.log10(sim.atot_km2)
        sim.ae_strom = abs(actual - predicted)
    
    return sim


def load_single_simulation(sim_path: Path) -> Optional[SimulationData]:
    """Load a single simulation from disk."""
    # Parse mu/xi from folder name if possible
    try:
        parts = sim_path.name.split('_')
        if 'mu' in sim_path.name:
            mu_idx = parts.index('mu') if 'mu' in parts else None
            xi_idx = parts.index('xi') if 'xi' in parts else None
            mu = float(parts[mu_idx + 1]) if mu_idx is not None else 0.1
            xi = float(parts[xi_idx + 1]) if xi_idx is not None else 1000
        else:
            mu, xi = 0.1, 1000
    except:
        mu, xi = 0.1, 1000

    sim = SimulationData(sim_id=sim_path.name, mu=mu, xi=xi)

    # Load DEM
    dem_path = sim_path / 'Inputs' / 'DEM.asc'
    dem_array = None
    dem_header = None

    if dem_path.exists():
        with open(dem_path, 'r') as f:
            for _ in range(6):
                f.readline()
            dem_array = np.loadtxt(f)
        dem_header = read_asc_header(dem_path)
        sim.bounds = dem_header['bounds']
        sim.cellsize = dem_header.get('cellsize', 10)

    # Get volume from release area
    rel_path = sim_path / 'Inputs' / 'REL'
    rel_files = list(rel_path.glob('*.asc')) if rel_path.exists() else []
    if rel_files:
        with open(rel_files[0], 'r') as f:
            for _ in range(6):
                f.readline()
            rel_data = np.loadtxt(f)
        rel_header = read_asc_header(rel_files[0])
        sim.volume_m3 = float(np.nansum(rel_data) * (rel_header.get('cellsize', 10) ** 2))

    # Find peak files
    peak_dir = sim_path / 'Outputs' / 'com1DFA' / 'peakFiles'
    thickness_files = list(peak_dir.glob('*_pft.asc')) if peak_dir.exists() else []
    velocity_files = list(peak_dir.glob('*_pfv.asc')) if peak_dir.exists() else []

    if thickness_files:
        sim.thickness_path = str(thickness_files[0])

        with open(thickness_files[0], 'r') as f:
            for _ in range(6):
                f.readline()
            thickness_data = np.loadtxt(f)

        thickness_data = np.where(thickness_data < 0, 0, thickness_data)
        sim.thickness_array = thickness_data
        sim.max_thickness = float(np.nanmax(thickness_data))

        # Compute metrics if we have DEM
        if dem_array is not None:
            header = read_asc_header(thickness_files[0])
            sim = compute_simulation_metrics(sim, dem_array, thickness_data, header)

    if velocity_files:
        sim.velocity_path = str(velocity_files[0])
        with open(velocity_files[0], 'r') as f:
            for _ in range(6):
                f.readline()
            vel_data = np.loadtxt(f)
        sim.max_velocity = float(np.nanmax(vel_data[vel_data > 0])) if np.any(vel_data > 0) else 0

    return sim, dem_array, dem_header


def load_sweep_data(sweep_path: Path, progress_callback=None) -> Optional[SweepData]:
    """Load a parameter sweep from disk."""
    sweep = SweepData(name=sweep_path.name, path=sweep_path)
    
    # Check for summary CSV
    summary_path = sweep_path / 'sweep_summary.csv'
    if not summary_path.exists():
        return None
    
    df = pd.read_csv(summary_path)
    
    # Find simulation directories
    sim_dirs = [d for d in sweep_path.iterdir()
                if d.is_dir() and d.name.startswith('mu_')]
    
    if not sim_dirs:
        return None
    
    # Load shared DEM from first simulation
    first_dir = sim_dirs[0]
    dem_path = first_dir / 'Inputs' / 'DEM.asc'
    
    if dem_path.exists():
        with open(dem_path, 'r') as f:
            # Skip header, read data
            for _ in range(6):
                f.readline()
            dem_data = np.loadtxt(f)
        
        header = read_asc_header(dem_path)
        sweep.dem_array = dem_data
        sweep.bounds = header['bounds']
        sweep.cellsize = header.get('cellsize', 10)
        sweep.hillshade_array = compute_hillshade(dem_data, sweep.cellsize)
    
    # Get volume from release area
    rel_path = first_dir / 'Inputs' / 'REL'
    rel_files = list(rel_path.glob('*.asc')) if rel_path.exists() else []
    if rel_files:
        with open(rel_files[0], 'r') as f:
            for _ in range(6):
                f.readline()
            rel_data = np.loadtxt(f)
        rel_header = read_asc_header(rel_files[0])
        sweep.volume_m3 = float(np.nansum(rel_data) * (rel_header.get('cellsize', 10) ** 2))
    
    # Load each simulation
    total = len(sim_dirs)
    for i, sim_dir in enumerate(sim_dirs):
        if progress_callback:
            progress_callback((i + 1) / total)
        
        # Parse mu/xi from folder name
        try:
            parts = sim_dir.name.split('_')
            mu = float(parts[1])
            xi = float(parts[3]) if len(parts) > 3 else float(parts[2])
        except:
            continue
        
        sim = SimulationData(sim_id=sim_dir.name, mu=mu, xi=xi, volume_m3=sweep.volume_m3)
        
        # Find peak thickness file
        peak_dir = sim_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
        thickness_files = list(peak_dir.glob('*_pft.asc')) if peak_dir.exists() else []
        velocity_files = list(peak_dir.glob('*_pfv.asc')) if peak_dir.exists() else []
        
        if thickness_files:
            sim.thickness_path = str(thickness_files[0])
            
            # Read thickness data
            with open(thickness_files[0], 'r') as f:
                for _ in range(6):
                    f.readline()
                thickness_data = np.loadtxt(f)
            
            thickness_data = np.where(thickness_data < 0, 0, thickness_data)
            sim.thickness_array = thickness_data
            sim.max_thickness = float(np.nanmax(thickness_data))
            
            # Compute metrics
            if sweep.dem_array is not None:
                header = read_asc_header(thickness_files[0])
                sim = compute_simulation_metrics(sim, sweep.dem_array, thickness_data, header)
        
        if velocity_files:
            sim.velocity_path = str(velocity_files[0])
            with open(velocity_files[0], 'r') as f:
                for _ in range(6):
                    f.readline()
                vel_data = np.loadtxt(f)
            sim.max_velocity = float(np.nanmax(vel_data[vel_data > 0])) if np.any(vel_data > 0) else 0
        
        sweep.simulations[(mu, xi)] = sim
    
    # Extract unique values
    sweep.mu_values = sorted(set(mu for mu, xi in sweep.simulations.keys()))
    sweep.xi_values = sorted(set(xi for mu, xi in sweep.simulations.keys()))
    
    return sweep


# =============================================================================
# Visualization Functions
# =============================================================================

def create_parameter_heatmap(sweep: SweepData, selected_mu: float,
                             selected_xi: float, metric: str = 'ae_strom') -> go.Figure:
    """Create heatmap of parameter space."""
    
    # Build grid
    grid = np.full((len(sweep.xi_values), len(sweep.mu_values)), np.nan)
    
    for (mu, xi), sim in sweep.simulations.items():
        try:
            i = sweep.mu_values.index(mu)
            j = sweep.xi_values.index(xi)
            
            if metric == 'ae_strom':
                grid[j, i] = sim.ae_strom
            elif metric == 'max_velocity':
                grid[j, i] = sim.max_velocity
            elif metric == 'atot_km2':
                grid[j, i] = sim.atot_km2
            elif metric == 'h_over_l':
                grid[j, i] = sim.h_over_l
        except ValueError:
            continue
    
    fig = go.Figure()
    
    # Heatmap
    fig.add_trace(go.Heatmap(
        x=sweep.mu_values,
        y=sweep.xi_values,
        z=grid,
        colorscale='Viridis',
        colorbar=dict(title=metric),
        hovertemplate='μ: %{x:.3f}<br>ξ: %{y:.0f}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    # Selection marker
    fig.add_trace(go.Scatter(
        x=[selected_mu],
        y=[selected_xi],
        mode='markers',
        marker=dict(size=20, color='white', line=dict(color='black', width=3)),
        name='Selected',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'Parameter Space — {metric}',
        xaxis_title='Friction (μ)',
        yaxis_title='Turbulence (ξ)',
        height=400,
        margin=dict(l=60, r=20, t=50, b=60)
    )
    
    return fig


def create_strom_plot(sweep: SweepData, selected_sim: SimulationData) -> go.Figure:
    """Create Strom's relation plot."""
    strom = get_strom_relation()
    
    fig = go.Figure()
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=strom['x_range'],
        y=strom['y_combined'],
        mode='lines',
        name=f"Strom (R²={strom['r2']:.2f})",
        line=dict(color='gray', dash='dash')
    ))
    
    # All simulations
    all_x = [s.v_times_h_km4 for s in sweep.simulations.values() if s.v_times_h_km4 > 0]
    all_y = [s.atot_km2 for s in sweep.simulations.values() if s.atot_km2 > 0]
    
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode='markers',
        marker=dict(size=8, color='lightgray', opacity=0.6),
        name='All simulations'
    ))
    
    # Selected point
    if selected_sim.v_times_h_km4 > 0:
        fig.add_trace(go.Scatter(
            x=[selected_sim.v_times_h_km4],
            y=[selected_sim.atot_km2],
            mode='markers',
            marker=dict(size=15, color='red', line=dict(color='black', width=2)),
            name='Selected'
        ))
    
    fig.update_layout(
        title='Strom et al. (2019)',
        xaxis_title='V × H (km⁴)',
        yaxis_title='Total Area (km²)',
        xaxis_type='log',
        yaxis_type='log',
        height=350,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(orientation='h', y=-0.2)
    )
    
    return fig


def create_brideau_plot(sweep: SweepData, selected_sim: SimulationData) -> go.Figure:
    """Create Brideau's H/L vs Volume plot."""
    brideau = get_brideau_relation()
    
    fig = go.Figure()
    
    # Regression lines
    fig.add_trace(go.Scatter(
        x=brideau['v_range'], y=brideau['hl_small'],
        mode='lines',
        name=f"Small/medium (R²={brideau['r2_small']:.2f})",
        line=dict(color='blue', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=brideau['v_range'], y=brideau['hl_large'],
        mode='lines',
        name=f"Large (R²={brideau['r2_large']:.2f})",
        line=dict(color='red')
    ))
    
    # All simulations
    all_x = [s.volume_m3 for s in sweep.simulations.values() if s.h_over_l > 0]
    all_y = [s.h_over_l for s in sweep.simulations.values() if s.h_over_l > 0]
    
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode='markers',
        marker=dict(size=8, color='lightgray', opacity=0.6),
        name='All simulations'
    ))
    
    # Selected point
    if selected_sim.h_over_l > 0:
        fig.add_trace(go.Scatter(
            x=[selected_sim.volume_m3],
            y=[selected_sim.h_over_l],
            mode='markers',
            marker=dict(size=15, color='red', line=dict(color='black', width=2)),
            name='Selected'
        ))
    
    fig.update_layout(
        title='Brideau et al. (2021)',
        xaxis_title='Volume (m³)',
        yaxis_title='H/L',
        xaxis_type='log',
        yaxis_type='log',
        height=350,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(orientation='h', y=-0.2)
    )
    
    return fig


def create_single_map_figure(sim: SimulationData, dem_array: np.ndarray,
                             bounds: tuple, cellsize: float,
                             max_thickness: float = 25.0) -> go.Figure:
    """Create map visualization for a single simulation."""
    if dem_array is None or sim.thickness_array is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False)
        return fig

    xmin, ymin, xmax, ymax = bounds
    hillshade = compute_hillshade(dem_array, cellsize)

    fig = go.Figure()

    # Hillshade
    fig.add_trace(go.Heatmap(
        z=hillshade,
        x0=xmin, dx=cellsize,
        y0=ymax, dy=-cellsize,
        colorscale='Greys',
        showscale=False,
        hoverinfo='skip'
    ))

    # Flow thickness
    thickness = sim.thickness_array.copy()
    thickness = np.where(thickness < 0.1, np.nan, thickness)

    colorscale = [
        [0.0, 'rgb(0,0,255)'],
        [0.25, 'rgb(0,255,255)'],
        [0.5, 'rgb(0,255,0)'],
        [0.75, 'rgb(255,255,0)'],
        [1.0, 'rgb(255,0,0)']
    ]

    fig.add_trace(go.Heatmap(
        z=thickness,
        x0=xmin, dx=cellsize,
        y0=ymax, dy=-cellsize,
        colorscale=colorscale,
        zmin=0, zmax=max_thickness,
        opacity=0.7,
        colorbar=dict(title='Thickness (m)', x=1.02, len=0.5),
        hovertemplate='Thickness: %{z:.1f} m<extra></extra>'
    ))

    # Hmax/Hmin markers
    fig.add_trace(go.Scatter(
        x=[sim.hmax_coords[0]], y=[sim.hmax_coords[1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'Hmax ({sim.hmax_m:.0f} m)'
    ))

    fig.add_trace(go.Scatter(
        x=[sim.hmin_coords[0]], y=[sim.hmin_coords[1]],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'Hmin ({sim.hmin_m:.0f} m)'
    ))

    # Runout line
    fig.add_trace(go.Scatter(
        x=[sim.hmax_coords[0], sim.hmin_coords[0]],
        y=[sim.hmax_coords[1], sim.hmin_coords[1]],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name=f'L = {sim.runout_l_m:.0f} m'
    ))

    # Calculate zoom extent
    flow_mask = sim.thickness_array >= 0.1
    if np.any(flow_mask):
        rows_with_flow = np.any(flow_mask, axis=1)
        cols_with_flow = np.any(flow_mask, axis=0)
        row_indices = np.where(rows_with_flow)[0]
        col_indices = np.where(cols_with_flow)[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            flow_xmin = xmin + col_indices[0] * cellsize
            flow_xmax = xmin + (col_indices[-1] + 1) * cellsize
            flow_ymax = ymax - row_indices[0] * cellsize
            flow_ymin = ymax - (row_indices[-1] + 1) * cellsize

            pad_x = (flow_xmax - flow_xmin) * 0.2
            pad_y = (flow_ymax - flow_ymin) * 0.2

            view_xmin = flow_xmin - pad_x
            view_xmax = flow_xmax + pad_x
            view_ymin = flow_ymin - pad_y
            view_ymax = flow_ymax + pad_y
        else:
            view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax
    else:
        view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax

    fig.update_layout(
        title=f'{sim.sim_id} | V={sim.max_velocity:.1f} m/s | A={sim.atot_km2:.2f} km²',
        xaxis=dict(scaleanchor='y', scaleratio=1, title='Easting (m)',
                  range=[view_xmin, view_xmax]),
        yaxis=dict(title='Northing (m)', range=[view_ymin, view_ymax]),
        height=600,
        margin=dict(l=60, r=80, t=60, b=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                   bgcolor='rgba(255,255,255,0.8)'),
        uirevision='map'
    )

    return fig


def create_map_figure(sweep: SweepData, selected_sim: SimulationData,
                      max_thickness: float = 25.0) -> go.Figure:
    """Create map visualization with hillshade and flow overlay."""

    if sweep.hillshade_array is None or selected_sim.thickness_array is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False)
        return fig

    xmin, ymin, xmax, ymax = sweep.bounds
    nrows, ncols = selected_sim.thickness_array.shape

    fig = go.Figure()

    # Hillshade
    fig.add_trace(go.Heatmap(
        z=sweep.hillshade_array,
        x0=xmin, dx=sweep.cellsize,
        y0=ymax, dy=-sweep.cellsize,
        colorscale='Greys',
        showscale=False,
        hoverinfo='skip'
    ))

    # Flow thickness
    thickness = selected_sim.thickness_array.copy()
    thickness = np.where(thickness < 0.1, np.nan, thickness)

    colorscale = [
        [0.0, 'rgb(0,0,255)'],
        [0.25, 'rgb(0,255,255)'],
        [0.5, 'rgb(0,255,0)'],
        [0.75, 'rgb(255,255,0)'],
        [1.0, 'rgb(255,0,0)']
    ]

    fig.add_trace(go.Heatmap(
        z=thickness,
        x0=xmin, dx=sweep.cellsize,
        y0=ymax, dy=-sweep.cellsize,
        colorscale=colorscale,
        zmin=0, zmax=max_thickness,
        opacity=0.7,
        colorbar=dict(title='Thickness (m)', x=1.02, len=0.5),
        hovertemplate='Thickness: %{z:.1f} m<extra></extra>'
    ))

    # Hmax/Hmin markers
    fig.add_trace(go.Scatter(
        x=[selected_sim.hmax_coords[0]],
        y=[selected_sim.hmax_coords[1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'Hmax ({selected_sim.hmax_m:.0f} m)'
    ))

    fig.add_trace(go.Scatter(
        x=[selected_sim.hmin_coords[0]],
        y=[selected_sim.hmin_coords[1]],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle',
                   line=dict(color='black', width=2)),
        name=f'Hmin ({selected_sim.hmin_m:.0f} m)'
    ))

    # Runout line
    fig.add_trace(go.Scatter(
        x=[selected_sim.hmax_coords[0], selected_sim.hmin_coords[0]],
        y=[selected_sim.hmax_coords[1], selected_sim.hmin_coords[1]],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name=f'L = {selected_sim.runout_l_m:.0f} m'
    ))

    # Calculate zoom extent based on flow area with padding
    flow_mask = selected_sim.thickness_array >= 0.1
    if np.any(flow_mask):
        # Find rows and cols with flow
        rows_with_flow = np.any(flow_mask, axis=1)
        cols_with_flow = np.any(flow_mask, axis=0)

        row_indices = np.where(rows_with_flow)[0]
        col_indices = np.where(cols_with_flow)[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            # Convert to coordinates
            flow_xmin = xmin + col_indices[0] * sweep.cellsize
            flow_xmax = xmin + (col_indices[-1] + 1) * sweep.cellsize
            flow_ymax = ymax - row_indices[0] * sweep.cellsize
            flow_ymin = ymax - (row_indices[-1] + 1) * sweep.cellsize

            # Add 20% padding
            pad_x = (flow_xmax - flow_xmin) * 0.2
            pad_y = (flow_ymax - flow_ymin) * 0.2

            view_xmin = flow_xmin - pad_x
            view_xmax = flow_xmax + pad_x
            view_ymin = flow_ymin - pad_y
            view_ymax = flow_ymax + pad_y
        else:
            view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax
    else:
        view_xmin, view_ymin, view_xmax, view_ymax = xmin, ymin, xmax, ymax

    fig.update_layout(
        title=f'μ={selected_sim.mu:.3f}, ξ={selected_sim.xi:.0f} | '
              f'V={selected_sim.max_velocity:.1f} m/s | '
              f'A={selected_sim.atot_km2:.2f} km²',
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
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                   bgcolor='rgba(255,255,255,0.8)'),
        uirevision='map'
    )

    return fig


# =============================================================================
# Main Page
# =============================================================================

st.title("📊 Results Explorer")
st.markdown(f"**Project:** {project.name}")

# Find available sweeps and single simulations
sweep_dirs = []
single_dirs = []
if project.simulations_dir.exists():
    for d in project.simulations_dir.iterdir():
        if d.is_dir():
            if (d / 'sweep_summary.csv').exists():
                sweep_dirs.append(d)
            elif (d / 'Outputs' / 'com1DFA' / 'peakFiles').exists():
                # It's a single simulation with results
                single_dirs.append(d)

# Check for pre-selected sweep from simulation page
preselected_path = st.session_state.get('explorer_sweep_path')

# Sidebar controls
with st.sidebar:
    st.header("Data Selection")

    has_sweeps = len(sweep_dirs) > 0
    has_singles = len(single_dirs) > 0

    if not has_sweeps and not has_singles:
        st.warning("No simulations found. Run simulations first.")
        st.stop()

    # Mode selection
    mode_options = []
    if has_sweeps:
        mode_options.append("Parameter Sweeps")
    if has_singles:
        mode_options.append("Single Simulations")

    data_mode = st.radio(
        "View:",
        mode_options,
        index=0,
        horizontal=True
    )

    if data_mode == "Parameter Sweeps":
        sweep_options = {d.name: d for d in sweep_dirs}

        # Pre-select if coming from simulation page
        default_idx = 0
        if preselected_path:
            preselected_name = Path(preselected_path).name
            if preselected_name in sweep_options:
                default_idx = list(sweep_options.keys()).index(preselected_name)

        selected_sweep_name = st.selectbox(
            "Select sweep:",
            options=list(sweep_options.keys()),
            index=default_idx
        )

        if st.button("Load Sweep", type="primary"):
            with st.spinner("Loading sweep data..."):
                progress = st.progress(0)
                sweep_data = load_sweep_data(
                    sweep_options[selected_sweep_name],
                    progress_callback=lambda p: progress.progress(p)
                )
                progress.empty()

                if sweep_data and sweep_data.simulations:
                    st.session_state['sweep_data'] = sweep_data
                    st.session_state['single_sim_data'] = None  # Clear single data
                    first_key = list(sweep_data.simulations.keys())[0]
                    st.session_state['selected_mu'] = first_key[0]
                    st.session_state['selected_xi'] = first_key[1]
                    st.success(f"Loaded {len(sweep_data.simulations)} simulations")
                else:
                    st.error("Failed to load sweep data")

    else:  # Single Simulations
        single_options = {d.name: d for d in single_dirs}

        selected_single_name = st.selectbox(
            "Select simulation:",
            options=list(single_options.keys())
        )

        if st.button("Load Simulation", type="primary"):
            with st.spinner("Loading simulation..."):
                result = load_single_simulation(single_options[selected_single_name])

                if result and result[0]:
                    sim_data, dem_array, dem_header = result
                    st.session_state['single_sim_data'] = {
                        'sim': sim_data,
                        'dem': dem_array,
                        'header': dem_header
                    }
                    st.session_state['sweep_data'] = None  # Clear sweep data
                    st.success(f"Loaded: {selected_single_name}")
                else:
                    st.error("Failed to load simulation")

    st.divider()

    # Visualization settings
    st.header("Settings")

    # Heatmap metric only relevant for sweeps
    if data_mode == "Parameter Sweeps":
        heatmap_metric = st.selectbox(
            "Heatmap metric:",
            options=['ae_strom', 'max_velocity', 'atot_km2', 'h_over_l'],
            format_func=lambda x: {
                'ae_strom': 'AE Strom (log₁₀)',
                'max_velocity': 'Max Velocity (m/s)',
                'atot_km2': 'Total Area (km²)',
                'h_over_l': 'H/L Ratio'
            }.get(x, x)
        )
    else:
        heatmap_metric = 'ae_strom'  # Default value when not used

    max_thickness = st.slider(
        "Max thickness colorscale (m):",
        min_value=5.0, max_value=100.0, value=25.0, step=5.0
    )

    st.divider()

    # Export section
    st.header("Export")

    has_sweep = 'sweep_data' in st.session_state and st.session_state['sweep_data'] is not None
    has_single = 'single_sim_data' in st.session_state and st.session_state['single_sim_data'] is not None

    if has_sweep or has_single:
        export_include_rasters = st.checkbox(
            "Include raster outputs",
            value=False,
            help="Include peak thickness/velocity ASC files (larger download)"
        )

        if has_sweep and st.button("📦 Export Sweep Data"):
            sweep_path = sweep_options[selected_sweep_name]

            with st.spinner("Preparing export..."):
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Add sweep summary CSV
                    summary_csv = sweep_path / 'sweep_summary.csv'
                    if summary_csv.exists():
                        zf.write(summary_csv, f"{sweep_path.name}/sweep_summary.csv")

                    # Add sweep results JSON
                    results_json = sweep_path / 'sweep_results.json'
                    if results_json.exists():
                        zf.write(results_json, f"{sweep_path.name}/sweep_results.json")

                    # Add each simulation's config and metadata
                    sim_dirs = [d for d in sweep_path.iterdir()
                               if d.is_dir() and d.name.startswith('mu_')]

                    for sim_dir in sim_dirs:
                        sim_name = sim_dir.name

                        for config_file in ['simulation_config.json', 'metadata.json',
                                           'local_com6RockAvalancheCfg.ini', 'avaframe_log.txt']:
                            config_path = sim_dir / config_file
                            if config_path.exists():
                                zf.write(config_path, f"{sweep_path.name}/{sim_name}/{config_file}")

                        if export_include_rasters:
                            peak_dir = sim_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
                            if peak_dir.exists():
                                for asc_file in peak_dir.glob('*.asc'):
                                    zf.write(asc_file,
                                            f"{sweep_path.name}/{sim_name}/peakFiles/{asc_file.name}")

                    export_meta = {
                        'exported_at': datetime.now().isoformat(),
                        'sweep_name': sweep_path.name,
                        'num_simulations': len(sim_dirs),
                        'includes_rasters': export_include_rasters
                    }
                    zf.writestr(f"{sweep_path.name}/export_info.json",
                               json.dumps(export_meta, indent=2))

                zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{sweep_path.name}_export.zip",
                mime="application/zip"
            )

        if has_single and st.button("📦 Export Simulation"):
            sim_path = single_options[selected_single_name]

            with st.spinner("Preparing export..."):
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    sim_name = sim_path.name

                    # Config files
                    for config_file in ['simulation_config.json', 'metadata.json',
                                       'local_com6RockAvalancheCfg.ini', 'avaframe_log.txt']:
                        config_path = sim_path / config_file
                        if config_path.exists():
                            zf.write(config_path, f"{sim_name}/{config_file}")

                    # Input files
                    inputs_dir = sim_path / 'Inputs'
                    if inputs_dir.exists():
                        # DEM
                        dem_file = inputs_dir / 'DEM.asc'
                        if dem_file.exists():
                            zf.write(dem_file, f"{sim_name}/Inputs/DEM.asc")

                        # Release area
                        rel_dir = inputs_dir / 'REL'
                        if rel_dir.exists():
                            for rel_file in rel_dir.glob('*.asc'):
                                zf.write(rel_file, f"{sim_name}/Inputs/REL/{rel_file.name}")

                    # Peak output files
                    if export_include_rasters:
                        peak_dir = sim_path / 'Outputs' / 'com1DFA' / 'peakFiles'
                        if peak_dir.exists():
                            for asc_file in peak_dir.glob('*.asc'):
                                zf.write(asc_file, f"{sim_name}/peakFiles/{asc_file.name}")

                    # Add computed statistics
                    single_data = st.session_state['single_sim_data']
                    sim = single_data['sim']
                    stats = {
                        'simulation_id': sim.sim_id,
                        'mu': sim.mu,
                        'xi': sim.xi,
                        'volume_m3': sim.volume_m3,
                        'delta_h_m': sim.delta_h_m,
                        'runout_l_m': sim.runout_l_m,
                        'h_over_l': sim.h_over_l,
                        'total_area_km2': sim.atot_km2,
                        'max_velocity_ms': sim.max_velocity,
                        'max_thickness_m': sim.max_thickness,
                        'hmax_m': sim.hmax_m,
                        'hmin_m': sim.hmin_m,
                        'hmax_coords': sim.hmax_coords,
                        'hmin_coords': sim.hmin_coords,
                        'exported_at': datetime.now().isoformat()
                    }
                    zf.writestr(f"{sim_name}/statistics.json", json.dumps(stats, indent=2))

                zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{sim_path.name}_export.zip",
                mime="application/zip"
            )
    else:
        st.info("Load data first to enable export")

# Main content
has_sweep_data = 'sweep_data' in st.session_state and st.session_state['sweep_data'] is not None
has_single_data = 'single_sim_data' in st.session_state and st.session_state['single_sim_data'] is not None

if not has_sweep_data and not has_single_data:
    st.info("Select a simulation and click 'Load' to begin.")
    st.stop()

# Handle single simulation view
if has_single_data:
    single_data = st.session_state['single_sim_data']
    sim = single_data['sim']
    dem_array = single_data['dem']
    dem_header = single_data['header']

    # Map visualization (full width for single sim)
    fig = create_single_map_figure(
        sim, dem_array,
        dem_header['bounds'] if dem_header else (0, 0, 100, 100),
        dem_header.get('cellsize', 10) if dem_header else 10,
        max_thickness
    )

    # Enable scroll zoom and configure for interactivity
    fig.update_layout(
        dragmode='pan',  # Default to pan mode
        xaxis=dict(fixedrange=False),  # Allow zoom on x-axis
        yaxis=dict(fixedrange=False),  # Allow zoom on y-axis
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key='single_map_view',
        config={
            'scrollZoom': True,  # Enable mouse wheel zoom
            'displayModeBar': True,  # Always show toolbar
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{sim.sim_id}_map',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

    st.caption("💡 **Tip:** Use mouse wheel to zoom, click and drag to pan. Toolbar: 🔍 zoom, 🏠 reset view, 📷 download image")

    # Stats row
    col_stats1, col_stats2 = st.columns(2)

    with col_stats1:
        st.subheader("Simulation Statistics")
        stats_html = f"""
        <style>
            .stats-table {{ font-size: 14px; width: 100%; }}
            .stats-table td {{ padding: 4px 8px; }}
            .stats-table .label {{ color: #666; }}
            .stats-table .value {{ font-weight: 600; text-align: right; }}
        </style>
        <table class="stats-table">
            <tr><td class="label">Simulation</td><td class="value">{sim.sim_id}</td></tr>
            <tr><td class="label">Friction (μ)</td><td class="value">{sim.mu:.3f}</td></tr>
            <tr><td class="label">Turbulence (ξ)</td><td class="value">{sim.xi:.0f}</td></tr>
            <tr><td class="label">Volume</td><td class="value">{sim.volume_m3/1e6:.2f} M m³</td></tr>
            <tr><td class="label">Drop Height (H)</td><td class="value">{sim.delta_h_m:.0f} m</td></tr>
            <tr><td class="label">Runout Length (L)</td><td class="value">{sim.runout_l_m:.0f} m</td></tr>
            <tr><td class="label">H/L Ratio</td><td class="value">{sim.h_over_l:.3f}</td></tr>
            <tr><td class="label">Total Area</td><td class="value">{sim.atot_km2:.2f} km²</td></tr>
            <tr><td class="label">Max Velocity</td><td class="value">{sim.max_velocity:.1f} m/s</td></tr>
            <tr><td class="label">Max Thickness</td><td class="value">{sim.max_thickness:.1f} m</td></tr>
        </table>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

    with col_stats2:
        st.subheader("Empirical Comparison")

        # Create simple comparison with empirical relations
        strom = get_strom_relation()
        brideau = get_brideau_relation()

        if sim.v_times_h_km4 > 0:
            predicted_strom = 10 ** (1.0884 + 0.5497 * np.log10(sim.v_times_h_km4))
            st.metric("Strom Predicted Area", f"{predicted_strom:.2f} km²",
                     delta=f"{sim.atot_km2 - predicted_strom:.2f} km²")

        if sim.volume_m3 > 0:
            predicted_hl_small = 10 ** (-0.033 * np.log10(sim.volume_m3) + 0.0315)
            predicted_hl_large = 10 ** (-0.137 * np.log10(sim.volume_m3) + 0.469)
            st.metric("Brideau H/L (small/med)", f"{predicted_hl_small:.3f}",
                     delta=f"{sim.h_over_l - predicted_hl_small:.3f}")
            st.metric("Brideau H/L (large)", f"{predicted_hl_large:.3f}",
                     delta=f"{sim.h_over_l - predicted_hl_large:.3f}")

    st.stop()  # Don't continue to sweep visualization

# Sweep data view
sweep_data = st.session_state['sweep_data']
selected_mu = st.session_state.get('selected_mu', sweep_data.mu_values[0])
selected_xi = st.session_state.get('selected_xi', sweep_data.xi_values[0])

# Parameter selection
if len(sweep_data.simulations) > 1:
    col_params, col_map = st.columns([1, 3])
    
    with col_params:
        st.subheader("Parameter Selection")
        
        # Heatmap
        heatmap_fig = create_parameter_heatmap(
            sweep_data, selected_mu, selected_xi, heatmap_metric
        )
        
        event = st.plotly_chart(
            heatmap_fig,
            width="stretch",
            key="param_heatmap",
            on_select="rerun",
            selection_mode="points"
        )
        
        # Handle click selection
        if event and event.selection and event.selection.points:
            point = event.selection.points[0]
            click_x, click_y = point.get('x'), point.get('y')
            
            if click_x is not None and click_y is not None:
                closest_mu = min(sweep_data.mu_values, key=lambda m: abs(m - click_x))
                closest_xi = min(sweep_data.xi_values, key=lambda x: abs(x - click_y))
                
                if (closest_mu, closest_xi) in sweep_data.simulations:
                    if closest_mu != selected_mu or closest_xi != selected_xi:
                        st.session_state['selected_mu'] = closest_mu
                        st.session_state['selected_xi'] = closest_xi
                        st.rerun()
        
        st.divider()
        
        # Sliders for fine control
        st.markdown("**Fine Control**")
        
        new_mu = st.select_slider(
            "Friction (μ)",
            options=sweep_data.mu_values,
            value=selected_mu,
            format_func=lambda x: f"{x:.3f}"
        )
        
        new_xi = st.select_slider(
            "Turbulence (ξ)",
            options=sweep_data.xi_values,
            value=selected_xi,
            format_func=lambda x: f"{x:.0f}"
        )
        
        if new_mu != selected_mu or new_xi != selected_xi:
            if (new_mu, new_xi) in sweep_data.simulations:
                st.session_state['selected_mu'] = new_mu
                st.session_state['selected_xi'] = new_xi
                st.rerun()
else:
    col_map = st.container()

# Get selected simulation
selected_key = (st.session_state['selected_mu'], st.session_state['selected_xi'])
if selected_key not in sweep_data.simulations:
    st.warning("Selected parameters not found")
    st.stop()

selected_sim = sweep_data.simulations[selected_key]

# Map visualization
map_fig = create_map_figure(sweep_data, selected_sim, max_thickness)
map_fig.update_layout(
    dragmode='pan',
    xaxis=dict(fixedrange=False),
    yaxis=dict(fixedrange=False),
)

map_config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': f'sweep_mu{selected_sim.mu:.3f}_xi{selected_sim.xi:.0f}',
        'height': 800,
        'width': 1200,
        'scale': 2
    }
}

if len(sweep_data.simulations) > 1:
    with col_map:
        st.plotly_chart(map_fig, use_container_width=True, key='map_view', config=map_config)
        st.caption("💡 Mouse wheel to zoom, drag to pan")
else:
    st.plotly_chart(map_fig, use_container_width=True, key='map_view', config=map_config)
    st.caption("💡 Mouse wheel to zoom, drag to pan")

# Bottom row: empirical plots and stats
col_strom, col_brideau, col_stats = st.columns([1, 1, 1])

with col_strom:
    st.plotly_chart(
        create_strom_plot(sweep_data, selected_sim),
        width="stretch"
    )

with col_brideau:
    st.plotly_chart(
        create_brideau_plot(sweep_data, selected_sim),
        width="stretch"
    )

with col_stats:
    st.subheader("Statistics")

    # Compact stats table
    stats_html = f"""
    <style>
        .stats-table {{ font-size: 14px; width: 100%; }}
        .stats-table td {{ padding: 4px 8px; }}
        .stats-table .label {{ color: #666; }}
        .stats-table .value {{ font-weight: 600; text-align: right; }}
    </style>
    <table class="stats-table">
        <tr><td class="label">Volume</td><td class="value">{selected_sim.volume_m3/1e6:.2f} M m³</td></tr>
        <tr><td class="label">Drop Height (H)</td><td class="value">{selected_sim.delta_h_m:.0f} m</td></tr>
        <tr><td class="label">Runout Length (L)</td><td class="value">{selected_sim.runout_l_m:.0f} m</td></tr>
        <tr><td class="label">H/L Ratio</td><td class="value">{selected_sim.h_over_l:.3f}</td></tr>
        <tr><td class="label">Total Area</td><td class="value">{selected_sim.atot_km2:.2f} km²</td></tr>
        <tr><td class="label">Max Velocity</td><td class="value">{selected_sim.max_velocity:.1f} m/s</td></tr>
        <tr><td class="label">AE Strom</td><td class="value">{selected_sim.ae_strom:.3f}</td></tr>
    </table>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown("")  # Spacer

    # Quality indicator
    if selected_sim.ae_strom < 0.2:
        st.success("Good match with Strom's relation")
    elif selected_sim.ae_strom < 0.5:
        st.warning("Moderate match")
    else:
        st.error("Poor match")

# References section
st.divider()
st.markdown("**References**")
st.markdown("""
- Brideau, M.-A., de Vilder, S., Massey, C., Mitchell, A., McDougall, S., & Aaron, J. (2021).
  *Empirical Relationships to Estimate the Probability of Runout Exceedance for Various Landslide Types.*
  DOI: [10.1007/978-3-030-60227-7_36](https://doi.org/10.1007/978-3-030-60227-7_36)
  — [Open Access PDF](https://www.researchgate.net/publication/347828445_Empirical_Relationships_to_Estimate_the_Probability_of_Runout_Exceedance_for_Various_Landslide_Types)

- Strom, A., Li, L., & Lan, H. (2019).
  *Rock avalanche mobility: optimal characterization and the effects of confinement.*
  Landslides, 16. DOI: [10.1007/s10346-019-01181-z](https://doi.org/10.1007/s10346-019-01181-z)
  — [Open Access PDF](https://link.springer.com/content/pdf/10.1007/s10346-019-01181-z.pdf)
""")
