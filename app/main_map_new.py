import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
import json
import zipfile
import os
from pathlib import Path
import sys
from pyproj import Transformer
import numpy as np
from PIL import Image
import io
import base64
import shutil
import multiprocessing as mp
from datetime import timedelta

# Add components to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

from ol_map import ol_map

# Add AvaFrame to path
avaframe_path = os.path.join(os.path.dirname(__file__), '..', 'AvaFrame')
sys.path.insert(0, avaframe_path)

# Target CRS for Norway
TARGET_CRS = "EPSG:25833"  # UTM33N - standard for Norway

def create_results_zip(ava_dir):
    """
    Create a ZIP file containing all simulation results.
    
    Args:
        ava_dir: Path to the AVAFRAME simulation directory
    
    Returns:
        BytesIO object containing the ZIP file
    """
    ava_dir = Path(ava_dir)
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all peak files (velocity, thickness, pressure, timeInfo)
        peak_dir = ava_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
        if peak_dir.exists():
            for asc_file in peak_dir.glob('*.asc'):
                zip_file.write(asc_file, f'peakFiles/{asc_file.name}')
        
        # Add statistics JSON
        stats_file = ava_dir / 'simulation_stats.json'
        if stats_file.exists():
            zip_file.write(stats_file, 'simulation_stats.json')
        
        # Add results CSV
        results_dir = ava_dir / 'Outputs' / 'com1DFA'
        if results_dir.exists():
            for csv_file in results_dir.glob('resultsDF_*.csv'):
                zip_file.write(csv_file, f'results/{csv_file.name}')
        
        # Add input files for reference
        inputs_dir = ava_dir / 'Inputs'
        if inputs_dir.exists():
            dem_file = inputs_dir / 'DEM.asc'
            if dem_file.exists():
                zip_file.write(dem_file, 'Inputs/DEM.asc')
            
            rel_dir = inputs_dir / 'REL'
            if rel_dir.exists():
                for rel_file in rel_dir.glob('*.asc'):
                    zip_file.write(rel_file, f'Inputs/REL/{rel_file.name}')
        
        # Add configuration files
        for config_name in ['local_com6RockAvalancheCfg.ini', 'local_com1DFACfg.ini']:
            config_file = ava_dir / config_name
            if config_file.exists():
                zip_file.write(config_file, f'configuration/{config_name}')
        
        # Add AVAFRAME configuration files (the ones actually used in simulation)
        config_dir = ava_dir / 'Outputs' / 'com1DFA' / 'configurationFiles'
        if config_dir.exists():
            for cfg_file in config_dir.glob('*.ini'):
                zip_file.write(cfg_file, f'configuration/avaframe/{cfg_file.name}')
        
        # Create a detailed parameter log
        log_content = _create_parameter_log(ava_dir)
        zip_file.writestr('SIMULATION_LOG.txt', log_content)
    
    zip_buffer.seek(0)
    return zip_buffer


def _create_parameter_log(ava_dir):
    """Create a detailed log of all simulation parameters."""
    from datetime import datetime
    import configparser
    
    ava_dir = Path(ava_dir)
    log_lines = []
    
    log_lines.append("=" * 80)
    log_lines.append("AVAFRAME ROCK AVALANCHE SIMULATION LOG")
    log_lines.append("=" * 80)
    log_lines.append(f"Simulation Directory: {ava_dir.name}")
    log_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("")
    
    # Read DEM information
    dem_file = ava_dir / 'Inputs' / 'DEM.asc'
    if dem_file.exists():
        log_lines.append("-" * 80)
        log_lines.append("DEM INFORMATION")
        log_lines.append("-" * 80)
        with open(dem_file) as f:
            for i, line in enumerate(f):
                if i < 6:  # Header lines
                    log_lines.append(line.strip())
                else:
                    break
        log_lines.append("")
    
    # Read release thickness information
    rel_file = ava_dir / 'Inputs' / 'REL' / 'relTh.asc'
    if rel_file.exists():
        log_lines.append("-" * 80)
        log_lines.append("RELEASE THICKNESS INFORMATION")
        log_lines.append("-" * 80)
        with open(rel_file) as f:
            for i, line in enumerate(f):
                if i < 6:  # Header lines
                    log_lines.append(line.strip())
                else:
                    break
        log_lines.append("")
    
    # Read local rock avalanche config
    rock_config_file = ava_dir / 'local_com6RockAvalancheCfg.ini'
    if rock_config_file.exists():
        log_lines.append("-" * 80)
        log_lines.append("ROCK AVALANCHE CONFIGURATION (local_com6RockAvalancheCfg.ini)")
        log_lines.append("-" * 80)
        config = configparser.ConfigParser()
        config.read(rock_config_file)
        
        for section in config.sections():
            log_lines.append(f"\n[{section}]")
            for key, value in config.items(section):
                log_lines.append(f"  {key} = {value}")
        log_lines.append("")
    
    # Read AVAFRAME source configuration (actual parameters used)
    source_config = list((ava_dir / 'Outputs' / 'com1DFA' / 'configurationFiles').glob('sourceConfiguration*.ini'))
    if source_config:
        log_lines.append("-" * 80)
        log_lines.append("AVAFRAME ACTUAL CONFIGURATION USED")
        log_lines.append("-" * 80)
        log_lines.append(f"File: {source_config[0].name}")
        log_lines.append("")
        
        config = configparser.ConfigParser()
        config.read(source_config[0])
        
        # Key parameters to highlight
        if 'GENERAL' in config:
            log_lines.append("KEY PARAMETERS:")
            log_lines.append("-" * 40)
            important_params = [
                'rho', 'meshCellSize', 'massPerPart', 'deltaTh', 
                'frictModel', 'muvoellmy', 'xsivoellmy', 'sphOption',
                'splitOption', 'tEnd', 'dt', 'stopCrit'
            ]
            for param in important_params:
                if param in config['GENERAL']:
                    log_lines.append(f"  {param:20s} = {config['GENERAL'][param]}")
            log_lines.append("")
    
    # Read simulation results
    results_csv = list((ava_dir / 'Outputs' / 'com1DFA').glob('resultsDF*.csv'))
    if results_csv:
        log_lines.append("-" * 80)
        log_lines.append("SIMULATION RESULTS SUMMARY")
        log_lines.append("-" * 80)
        try:
            import pandas as pd
            df = pd.read_csv(results_csv[0])
            if not df.empty:
                result = df.iloc[0]
                log_lines.append(f"  Simulation Name    : {result.get('simName', 'N/A')}")
                log_lines.append(f"  Release Scenario   : {result.get('releaseScenario', 'N/A')}")
                log_lines.append(f"  Max Velocity       : {result.get('ppr_max', 'N/A')} m/s")
                log_lines.append(f"  Max Thickness      : {result.get('pft_max', 'N/A')} m")
                log_lines.append(f"  Max Pressure       : {result.get('pfv_max', 'N/A')} kPa")
                log_lines.append(f"  Runtime            : {result.get('runtime', 'N/A')} s")
        except Exception as e:
            log_lines.append(f"  Could not read results: {e}")
        log_lines.append("")
    
    # Statistics JSON
    stats_file = ava_dir / 'simulation_stats.json'
    if stats_file.exists():
        log_lines.append("-" * 80)
        log_lines.append("CALCULATED STATISTICS")
        log_lines.append("-" * 80)
        try:
            import json
            with open(stats_file) as f:
                stats = json.load(f)
            for key, value in stats.items():
                log_lines.append(f"  {key:20s} : {value}")
        except Exception as e:
            log_lines.append(f"  Could not read stats: {e}")
        log_lines.append("")
    
    log_lines.append("=" * 80)
    log_lines.append("END OF SIMULATION LOG")
    log_lines.append("=" * 80)
    
    return "\n".join(log_lines)

    
    zip_buffer.seek(0)
    return zip_buffer

# Create transformer for coordinate conversion
transformer_to_latlon = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

def raster_to_image_overlay(raster_path, layer_type='velocity'):
    """
    Convert raster to base64 PNG image with bounds for map overlay.
    
    Args:
        raster_path: Path to raster file
        layer_type: Type of data ('velocity', 'thickness', 'pressure', 'time')
    
    Returns:
        Dict with image data, bounds, extent, and legend info
    """
    try:
        with rasterio.open(raster_path) as src:
            # Read first band
            data = src.read(1)
            
            # Get bounds in EPSG:25833
            bounds = src.bounds
            
            # Replace nodata values with NaN
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            # Filter to valid data only (> 0)
            data = np.where(data > 0, data, np.nan)
            
            # Get data range
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            
            if np.isnan(data_min) or np.isnan(data_max) or data_min == data_max:
                normalized = np.zeros_like(data, dtype=np.uint8)
            else:
                # Normalize and handle NaN values
                normalized = np.where(
                    np.isnan(data),
                    0,
                    ((data - data_min) / (data_max - data_min) * 255)
                ).astype(np.uint8)
            
            # Create RGBA image
            rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
            
            # Create mask for valid data (not NaN)
            valid_mask = ~np.isnan(data)
            
            # Apply colormap based on layer type
            if layer_type == 'velocity':
                # Blue -> Yellow -> Red (speed ramp)
                rgba[:, :, 0] = normalized  # Red increases
                rgba[:, :, 1] = np.where(normalized < 128, normalized * 2, 255 - (normalized - 128) * 2)  # Yellow mid
                rgba[:, :, 2] = 255 - normalized  # Blue decreases
                unit = "m/s"
                label = "Velocity"
            elif layer_type == 'thickness':
                # White -> Blue -> Dark blue (depth)
                rgba[:, :, 0] = 255 - normalized  # Red decreases
                rgba[:, :, 1] = 255 - normalized  # Green decreases
                rgba[:, :, 2] = 255  # Blue constant
                unit = "m"
                label = "Thickness"
            elif layer_type == 'pressure':
                # Green -> Yellow -> Red (pressure/danger)
                rgba[:, :, 0] = normalized  # Red increases
                rgba[:, :, 1] = 255 - normalized // 2  # Green decreases slowly
                rgba[:, :, 2] = 0  # No blue
                unit = "Pa"
                label = "Pressure"
            else:  # time
                # Purple gradient
                rgba[:, :, 0] = normalized  # Red
                rgba[:, :, 1] = normalized // 2  # Less green
                rgba[:, :, 2] = 255 - normalized  # Blue decreases
                unit = "s"
                label = "Time"
            
            rgba[:, :, 3] = np.where(valid_mask, 200, 0)  # Alpha (transparent where NaN)
            
            # Convert to PIL Image
            img = Image.fromarray(rgba, 'RGBA')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                'image': f'data:image/png;base64,{img_base64}',
                'bounds': [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)],
                'extent': [float(data_min), float(data_max)],
                'unit': unit,
                'label': label
            }
    except Exception as e:
        st.info(f"Error converting raster: {str(e)}")
        return None


def _write_aaigrid(filepath, data, ncols, nrows, xllcenter, yllcenter, 
                   cellsize, nodata_value=-9999, is_thickness=False):
    """
    Write data in AAIGrid format matching AVAFRAME expectations.
    
    KRITISK: Sjekker HVER verdi for å garantere at ingen NaN/Inf skrives til fil.
    
    Args:
        filepath: Output path
        data: numpy array with data
        ncols, nrows: Grid dimensions
        xllcenter, yllcenter: Grid origin (center of lower-left cell)
        cellsize: Cell size in meters
        nodata_value: Value to use for nodata in header
        is_thickness: If True, replace invalid values with 0 instead of nodata
    """
    with open(filepath, 'w') as f:
        # Write header
        f.write(f"ncols        {ncols}\n")
        f.write(f"nrows        {nrows}\n")
        f.write(f"xllcenter    {xllcenter:.2f}\n")
        f.write(f"yllcenter    {yllcenter:.2f}\n")
        f.write(f"cellsize     {cellsize:.15f}\n")
        f.write(f"nodata_value {nodata_value:.2f}\n")
        
        # Write data - check EACH value individually!
        nan_replaced = 0
        for row in data:
            values = []
            for val in row:
                # KRITISK: Sjekk hver verdi
                # np.isfinite() returnerer False for NaN, Inf, -Inf
                if not np.isfinite(val):
                    # For thickness: use 0 (not nodata, because AVAFRAME treats non-zero as release)
                    # For DEM: use nodata_value
                    if is_thickness:
                        values.append("0.000000")
                    else:
                        values.append(f"{nodata_value:.6f}")
                    nan_replaced += 1
                else:
                    # Format the value
                    formatted = f"{float(val):.6f}"
                    # DOBBELTSJEKK: Aldri skriv "nan" eller "inf" som streng
                    if 'nan' in formatted.lower() or 'inf' in formatted.lower():
                        if is_thickness:
                            values.append("0.000000")
                        else:
                            values.append(f"{nodata_value:.6f}")
                        nan_replaced += 1
                    else:
                        values.append(formatted)
            
            f.write(' '.join(values) + '\n')
    
    # Log if NaN was found and replaced
    if nan_replaced > 0:
        st.warning(f"⚠️ Replaced {nan_replaced} NaN/Inf values when writing {filepath.name}")
    
    return nan_replaced


def _verify_asc_file(filepath):
    """
    Verify that an ASC file contains no NaN or invalid values.
    
    Returns: (is_valid, error_message)
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for "nan" string (case insensitive)
        if 'nan' in content.lower():
            # Find where
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'nan' in line.lower():
                    return False, f"Found 'nan' on line {i+1}"
        
        # Check for "inf" string
        if 'inf' in content.lower():
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'inf' in line.lower():
                    return False, f"Found 'inf' on line {i+1}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_rock_avalanche_simulation(release_gdf, thickness_path, dem_path):
    """
    Run AVAFRAME rock avalanche simulation.
    
    KRITISK: com1DFA krever at tykkelsesrasteret har 
    NØYAKTIG samme dimensjoner som DEM-filen!
    """
    def log(msg, level='info'):
        """Helper for logging that works with or without Streamlit context."""
        try:
            if level == 'info': st.info(msg)
            elif level == 'success': st.success(msg)
            elif level == 'warning': st.warning(msg)
            elif level == 'error': st.error(msg)
        except:
            print(f"[{level.upper()}] {msg}")
    
    try:
        import subprocess
        import time
        
        # Create simulation directory
        sim_base = Path("/mnt/data/simulations")
        sim_base.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ava_dir = sim_base / f"rock_avalanche_{timestamp}"
        
        # Check if folder exists and create unique name
        counter = 1
        original_ava_dir = ava_dir
        while ava_dir.exists():
            ava_dir = Path(f"{original_ava_dir}_{counter}")
            counter += 1
        
        ava_dir.mkdir(parents=True, exist_ok=True)
        log(f"Directory: {ava_dir}")
        
        # Create folder structure manually (avoid importing avaframe here)
        (ava_dir / 'Inputs' / 'REL').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Inputs' / 'RES').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Inputs' / 'ENT').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Inputs' / 'POINTS').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Inputs' / 'LINES').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Inputs' / 'POLYGONS').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Outputs').mkdir(parents=True, exist_ok=True)
        (ava_dir / 'Work').mkdir(parents=True, exist_ok=True)
        
        # Get bounding box from release area with buffer
        bounds = release_gdf.total_bounds  # [minx, miny, maxx, maxy]
        buffer = 4000  # meters - large buffer to capture full runout zone
        bbox = [bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer]
        
        # ============================================================
        # STEP 1: Crop DEM to simulation area
        # ============================================================
        log("Cropping DEM...")
        
        with rasterio.open(dem_path) as src_dem:
            # Calculate window for cropping
            window = rasterio.windows.from_bounds(*bbox, src_dem.transform)
            
            # Read cropped DEM
            dem_data = src_dem.read(1, window=window)
            dem_transform = src_dem.window_transform(window)
            dem_crs = src_dem.crs
            
            nrows, ncols = dem_data.shape
            cellsize = abs(dem_transform[0])
            
            log(f"DEM: {ncols}x{nrows} @ {cellsize:.1f}m")
            
            # Handle nodata in DEM - use np.nan_to_num for robustness
            dem_data = np.nan_to_num(dem_data, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
            if src_dem.nodata is not None:
                dem_data = np.where(dem_data == src_dem.nodata, -9999.0, dem_data)
            
            nodata_count = np.sum(dem_data == -9999)
            if nodata_count > 0:
                log(f"DEM nodata cells: {nodata_count}")
            
            # Calculate grid parameters (center coordinates for AAIGRID)
            xllcenter = dem_transform[2] + cellsize / 2.0
            yllcenter = dem_transform[5] - (nrows * cellsize) + cellsize / 2.0
            
            # Write DEM in AAIGRID format
            dem_dest = ava_dir / 'Inputs' / 'DEM.asc'
            _write_aaigrid(dem_dest, dem_data, ncols, nrows, 
                          xllcenter, yllcenter, cellsize, -9999, is_thickness=False)
            
            # Write .prj file
            with open(dem_dest.with_suffix('.prj'), 'w') as f:
                f.write(dem_crs.to_wkt())
        
        # Verify DEM file
        is_valid, error_msg = _verify_asc_file(dem_dest)
        if not is_valid:
            log(f"DEM verification failed: {error_msg}")
            return None, None, None
        
        log(f"DEM saved: {dem_dest.name}")
        
        # ============================================================
        # STEP 2: Resample thickness raster to EXACT DEM grid
        # ============================================================
        log("Resampling thickness...")
        
        # Create empty array matching DEM dimensions - use float64 for precision
        thickness_resampled = np.zeros((nrows, ncols), dtype=np.float64)
        
        with rasterio.open(thickness_path) as src_thick:
            # Reproject/resample thickness to DEM grid
            reproject(
                source=rasterio.band(src_thick, 1),
                destination=thickness_resampled,
                src_transform=src_thick.transform,
                src_crs=src_thick.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan  # Explicitly set areas outside source to NaN
            )
            
            # Get original thickness stats for logging
            orig_data = src_thick.read(1)
            valid_orig = orig_data[(orig_data > 0) & np.isfinite(orig_data)]
            if len(valid_orig) > 0:
                log(f"Source: {valid_orig.min():.1f}-{valid_orig.max():.1f}m (mean {valid_orig.mean():.1f}m)")
        
        # ============================================================
        # STEP 3: AGGRESSIVE NaN cleanup
        # ============================================================
        # Count NaN before cleanup
        nan_before = np.sum(~np.isfinite(thickness_resampled))
        if nan_before > 0:
            log(f"Cleaning {nan_before} non-finite values...")
        
        # Replace ALL non-finite values with 0
        thickness_resampled = np.nan_to_num(thickness_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Also replace any negative values
        thickness_resampled = np.where(thickness_resampled < 0, 0.0, thickness_resampled)
        
        # ============================================================
        # STEP 4: Mask to release area polygon
        # ============================================================
        from rasterio import features as rio_features
        
        # Create release area mask on DEM grid
        release_mask = np.zeros((nrows, ncols), dtype=np.uint8)
        shapes = [(geom, 1) for geom in release_gdf.geometry]
        rio_features.rasterize(
            shapes=shapes,
            out=release_mask,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8
        )
        
        # Apply mask: keep thickness only inside release area
        thickness_resampled = np.where(release_mask == 1, thickness_resampled, 0.0)
        
        # FINAL verification - no NaN should remain
        nan_after = np.sum(~np.isfinite(thickness_resampled))
        if nan_after > 0:
            log(f"Warning: {nan_after} non-finite values remain")
            # Force replace any remaining
            thickness_resampled = np.where(np.isfinite(thickness_resampled), thickness_resampled, 0.0)
        
        # Statistics
        valid_cells = np.sum(thickness_resampled > 0)
        if valid_cells > 0:
            valid_thickness = thickness_resampled[thickness_resampled > 0]
            log(f"Resampled: {valid_cells} cells, {valid_thickness.min():.1f}-{valid_thickness.max():.1f}m (mean {valid_thickness.mean():.1f}m)")
        else:
            log("No valid thickness cells after resampling")
            return None, None, None
        
        # ============================================================
        # STEP 5: Write thickness raster
        # ============================================================
        thickness_dest = ava_dir / 'Inputs' / 'REL' / 'relTh.asc'
        _write_aaigrid(thickness_dest, thickness_resampled, ncols, nrows,
                      xllcenter, yllcenter, cellsize, nodata_value=-9999, is_thickness=True)
        
        # Write .prj file
        with open(thickness_dest.with_suffix('.prj'), 'w') as f:
            f.write(dem_crs.to_wkt())
        
        # VERIFY thickness file has no NaN
        is_valid, error_msg = _verify_asc_file(thickness_dest)
        if not is_valid:
            log(f"Thickness verification failed: {error_msg}")
            return None, None, None
        
        log(f"Thickness saved: {thickness_dest.name}")
        log(f"Grid verified: {ncols}x{nrows}")
        
        # ============================================================
        # STEP 6: Create local configuration files
        # ============================================================
        log(f"Creating rock avalanche config (meshCellSize={cellsize}m)...")
        
        # Clean up any previous remeshed rasters to avoid conflicts
        remeshed_dir = ava_dir / 'Inputs' / 'remeshedRasters'
        if remeshed_dir.exists():
            import shutil
            shutil.rmtree(remeshed_dir)
            log("Cleaned remeshedRasters directory")
        
        # Create local_com6RockAvalancheCfg.ini with ALL settings including mesh
        # NOTE: com6RockAvalanche calls com1DFA internally, so all com1DFA settings
        # must be in the [com1DFA_com1DFA_override] section
        rock_config = ava_dir / 'local_com6RockAvalancheCfg.ini'
        with open(rock_config, 'w') as f:
            f.write(f"""### Local Config File - Rock Avalanche settings
## Based on AvaFrame/avaframe/com6RockAvalanche/com6RockAvalancheCfg.ini

[GENERAL]

[com1DFA_com1DFA_override]

# use default com1DFA config as base configuration
defaultConfig = True

#+++++++++++++ Mesh settings (CRITICAL - prevents remeshing errors)
# Match meshCellSize to DEM resolution
meshCellSize = {cellsize}
# Large threshold to effectively disable automatic remeshing
meshCellSizeThreshold = {cellsize * 10.0}

#+++++++++++++ Output++++++++++++
# desired result Parameters
resType = pft|pfv|ppr|FT

#+++++++++SNOW properties (actually rock properties)
# density of rock [kg/m³]
rho = 2500

#+++++++++++++SPH parameters
# SPH gradient option (3 = local coord system with reprojection)
sphOption = 3

#++++++++++++++++ Particles
# mass per particle (if MPPDIR is used) [kg]
massPerPart = 280000.
# release thickness per particle (if MPPDH is used) [m]
deltaTh = 2.0
# splitting option (1 = split/merge to keep constant particles per kernel)
splitOption = 1

#++++++++++++Friction model
# Voellmy friction model for rock avalanches
frictModel = Voellmy
# friction coefficient
muvoellmy = 0.035
# turbulence coefficient [m/s²]
xsivoellmy = 700.
""")
        
        log(f"Config saved: {rock_config.name}")
        
        # ============================================================
        # STEP 7: Run AVAFRAME in subprocess
        # ============================================================
        log("Running AVAFRAME com6RockAvalanche...")
        
        # Create runner script
        runner_script = ava_dir / 'run_simulation.py'
        _create_runner_script(runner_script, ava_dir)
        
        # Clean Work directory if it exists (from previous failed runs)
        work_dir = ava_dir / 'Work' / 'com1DFA'
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)
            log("Cleaned Work directory")
        
        # Run in subprocess with 15-minute timeout
        result = subprocess.run(
            [sys.executable, str(runner_script)],  # Use same Python as Streamlit
            capture_output=True,
            text=True,
            timeout=1200  # 20 minute timeout
        )
        
        if result.returncode != 0:
            log(f"❌ Simulation failed!")
            st.code(result.stderr[-2000:] if result.stderr else "No error output")
            
            # Check for partial results anyway
            output_dir = ava_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
            if not output_dir.exists() or not any(output_dir.glob('*.asc')):
                return None, None, None
        else:
            log("Simulation completed")
            if result.stdout:
                st.text(result.stdout[-1000:])  # Last 1000 chars
        
        # ============================================================
        # STEP 7: Collect results
        # ============================================================
        output_dir = ava_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
        
        result_files = {}
        if output_dir.exists():
            for file in output_dir.glob('*.asc'):
                if 'ppr' in file.name:
                    result_files['pressure'] = str(file)
                elif 'pft' in file.name:
                    result_files['thickness'] = str(file)
                elif 'pfv' in file.name:
                    result_files['velocity'] = str(file)
        
        # Read stats from output JSON if available
        stats_file = ava_dir / 'simulation_stats.json'
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
        else:
            # Default stats
            stats = {
                'max_velocity': 0,
                'runout': 0,
                'volume': float(valid_cells * cellsize * cellsize * valid_thickness.mean()),
                'max_depth': float(valid_thickness.max())
            }
        
        log(f"Results saved: {ava_dir}")
        
        return result_files, stats, str(ava_dir)
        
    except subprocess.TimeoutExpired:
        log("Simulation timed out after 20 minutes")
        return None, None, None
    except Exception as e:
        log(f"Simulation error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


def _create_runner_script(script_path, ava_dir):
    """Create a standalone Python script to run AVAFRAME com6RockAvalanche."""
    # Get absolute path to AvaFrame directory
    avaframe_abs_path = str(Path(__file__).parent.parent / 'AvaFrame')
    
    script_content = f'''#!/usr/bin/env python3
"""Standalone AVAFRAME com6RockAvalanche runner - uses local configs."""
import sys
import os
from pathlib import Path

# Add AvaFrame to path (absolute path)
sys.path.insert(0, "{avaframe_abs_path}")

def main():
    from avaframe.in3Utils import cfgUtils
    from avaframe.com6RockAvalanche import com6RockAvalanche
    
    ava_dir = Path("{ava_dir}")
    
    # Configure
    cfgMain = cfgUtils.getGeneralConfig()
    cfgMain['MAIN']['avalancheDir'] = str(ava_dir)
    
    # Get com6RockAvalanche config with fileOverride pointing to our local config
    local_rock_config = ava_dir / 'local_com6RockAvalancheCfg.ini'
    
    if local_rock_config.exists():
        print(f"Using local rock avalanche config: {{local_rock_config}}")
        cfg = cfgUtils.getModuleConfig(com6RockAvalanche, fileOverride=str(local_rock_config))
    else:
        print("WARNING: local_com6RockAvalancheCfg.ini not found, using defaults")
        cfg = cfgUtils.getModuleConfig(com6RockAvalanche)
    
    print(f"Running com6RockAvalanche for: {{ava_dir}}")
    print(f"Density (rho): {{cfg['com1DFA_com1DFA_override'].get('rho', 'not set')}} kg/m³")
    print(f"Mass per particle: {{cfg['com1DFA_com1DFA_override'].get('massPerPart', 'not set')}} kg")
    print(f"Friction model: {{cfg['com1DFA_com1DFA_override'].get('frictModel', 'not set')}}")
    
    # Run simulation (note: parameter name is rockAvalancheCfg, not cfgInfo)
    dem, plotDict, reportDictList, simDF = com6RockAvalanche.com6RockAvalancheMain(cfgMain, rockAvalancheCfg=cfg)
    
    print(f"Simulation completed! Generated {{len(simDF)}} simulation(s)")
    print(f"Output directory: {{ava_dir / 'Outputs' / 'com1DFA'}}")
    
    # Read results CSV and generate stats
    import pandas as pd
    import json
    
    results_csv = list((ava_dir / 'Outputs' / 'com1DFA').glob('resultsDF*.csv'))
    if results_csv:
        df = pd.read_csv(results_csv[0])
        
        # Calculate affected area from peak files
        import rasterio
        import numpy as np
        peak_files = list((ava_dir / 'Outputs' / 'com1DFA' / 'peakFiles').glob('*_pfv.asc'))
        affected_area_ha = 0
        if peak_files:
            with rasterio.open(peak_files[0]) as src:
                data = src.read(1)
                cellsize = src.transform[0]  # Cell width in meters
                valid_cells = np.sum(np.isfinite(data) & (data > 0))
                affected_area_ha = valid_cells * cellsize * cellsize / 10000  # Convert m² to hectares
        
        stats = {{
            'max_velocity': float(df['maxpfv'].max()) if 'maxpfv' in df.columns else 0,
            'max_pressure': float(df['maxppr'].max()) if 'maxppr' in df.columns else 0,
            'max_thickness': float(df['maxpft'].max()) if 'maxpft' in df.columns else 0,
            'affected_area': float(affected_area_ha),
            'runout': 0,  # Would need to calculate from results
            'volume': 0  # Would need to calculate from results
        }}
        
        # Save stats
        with open(ava_dir / 'simulation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Stats: max_velocity={{stats['max_velocity']:.1f}} m/s, max_pressure={{stats['max_pressure']:.0f}} Pa, area={{stats['affected_area']:.1f}} ha")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


# Helper functions for CRS handling
def load_vector_file(uploaded_file, target_crs=TARGET_CRS):
    """Load vector file and reproject to target CRS if needed."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        if uploaded_file.name.endswith('.zip'):
            # Handle shapefile zip
            zip_path = os.path.join(temp_dir, uploaded_file.name)
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find .shp file
            shp_file = None
            for file in os.listdir(temp_dir):
                if file.endswith('.shp'):
                    shp_file = os.path.join(temp_dir, file)
                    break
            
            if not shp_file:
                return None, "No .shp file found in zip"
            
            gdf = gpd.read_file(shp_file)
        
        elif uploaded_file.name.endswith(('.geojson', '.gpkg')):
            # Handle GeoJSON or GeoPackage
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            gdf = gpd.read_file(file_path)
        
        else:
            return None, "Unsupported file format"
        
        # Get original CRS
        original_crs = gdf.crs
        
        # Reproject to target CRS if needed
        if gdf.crs is None:
            return None, "File has no CRS defined"
        
        if gdf.crs != target_crs:
            gdf_reprojected = gdf.to_crs(target_crs)
            crs_info = f"Converted from {original_crs} to {target_crs}"
        else:
            gdf_reprojected = gdf
            crs_info = f"Already in {target_crs}"
        
        return gdf_reprojected, crs_info
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def load_raster_file(uploaded_file, target_crs=TARGET_CRS):
    """Load raster file and reproject to target CRS if needed."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        with rasterio.open(file_path) as src:
            original_crs = src.crs
            
            if src.crs is None:
                return None, None, "Raster has no CRS defined"
            
            if src.crs.to_string() != target_crs:
                # Need to reproject
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Create reprojected raster
                output_path = os.path.join(temp_dir, f"reprojected_{uploaded_file.name}")
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear
                        )
                
                crs_info = f"Converted from {original_crs} to {target_crs}"
                return output_path, kwargs, crs_info
            else:
                crs_info = f"Already in {target_crs}"
                return file_path, src.meta, crs_info
    
    except Exception as e:
        return None, None, f"Error loading raster: {str(e)}"


# Page config
st.set_page_config(
    page_title="Avalanche Modeling Tool",
    page_icon="🏔️",
    layout="wide"
)

# ============================================================
# AUTHENTICATION
# ============================================================
# Load configuration
config_file = Path(__file__).parent.parent / '.streamlit' / 'config.yaml'
with open(config_file) as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login widget
try:
    authenticator.login()
except Exception as e:
    st.error(e)

# Handle authentication
if st.session_state.get('authentication_status') == False:
    st.error('Username/password is incorrect')
    st.stop()
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')
    st.stop()

# User is authenticated - show logout button in sidebar
with st.sidebar:
    st.write(f'Welcome *{st.session_state.get("name")}*')
    authenticator.logout(location='sidebar')
    st.markdown("---")

# ============================================================
# MAIN APPLICATION (only shown if authenticated)
# ============================================================

# Import sweep manager
from sweep_manager import SweepConfig, run_parameter_sweep

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'selected_result_layer' not in st.session_state:
    st.session_state.selected_result_layer = 'velocity'
if 'map_center' not in st.session_state:
    st.session_state.map_center = [61.5, 8.5]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 7
if 'zoom_to_bounds' not in st.session_state:
    st.session_state.zoom_to_bounds = None
if 'zoom_counter' not in st.session_state:
    st.session_state.zoom_counter = 0
if 'dem_path' not in st.session_state:
    st.session_state.dem_path = "/mnt/data/dem/dtm2020_final_COG.tif"
if 'sweep_running' not in st.session_state:
    st.session_state.sweep_running = False
if 'sweep_progress' not in st.session_state:
    st.session_state.sweep_progress = None
if 'sweep_results' not in st.session_state:
    st.session_state.sweep_results = None

# Sidebar
with st.sidebar:
    st.title("Avalanche Modeling")
    
    st.markdown("**📤 Data Upload**")
    
    # Release area upload
    release_file = st.file_uploader(
        "Release Area",
        type=['shp', 'zip', 'geojson', 'gpkg'],
        help="Upload shapefile (zip), GeoJSON, or GeoPackage",
        key="release_upload"
    )
    
    if release_file:
        with st.spinner("Loading..."):
            gdf, crs_info = load_vector_file(release_file)
            
            if gdf is not None:
                st.success(f"✓ {release_file.name}")
                st.caption(f"CRS: {crs_info}")
                st.caption(f"Features: {len(gdf)}")
                st.session_state.uploaded_files['release'] = release_file
                st.session_state.uploaded_files['release_gdf'] = gdf
            else:
                st.error(f"{crs_info}")
    
    # Thickness raster upload
    thickness_file = st.file_uploader(
        "Thickness Raster",
        type=['tif', 'tiff', 'asc'],
        help="Upload thickness distribution raster",
        key="thickness_upload"
    )
    
    if thickness_file:
        with st.spinner("Loading..."):
            raster_path, raster_meta, crs_info = load_raster_file(thickness_file)
            
            if raster_path is not None:
                st.success(f"✓ {thickness_file.name}")
                st.caption(f"CRS: {crs_info}")
                st.caption(f"Size: {raster_meta['width']}x{raster_meta['height']}")
                st.session_state.uploaded_files['thickness'] = thickness_file
                st.session_state.uploaded_files['thickness_path'] = raster_path
                st.session_state.uploaded_files['thickness_meta'] = raster_meta
            else:
                st.error(f"{crs_info}")
    
    # Show file controls if loaded
    if 'release_gdf' in st.session_state.uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Zoom to Area", width='stretch', key="zoom_release", disabled=st.session_state.simulation_running):
                gdf = st.session_state.uploaded_files['release_gdf']
                bounds = gdf.total_bounds
                st.session_state.zoom_to_bounds = {
                    'bounds': [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
                    'counter': st.session_state.zoom_counter
                }
                st.session_state.zoom_counter += 1
                st.rerun()
        with col2:
            if st.button("Remove", width='stretch', key="remove_release", disabled=st.session_state.simulation_running):
                if 'release' in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files['release']
                if 'release_gdf' in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files['release_gdf']
                st.rerun()

    if 'thickness_path' in st.session_state.uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Zoom to Raster", width='stretch', key="zoom_thickness", disabled=st.session_state.simulation_running):
                raster_path = st.session_state.uploaded_files['thickness_path']
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    st.session_state.zoom_to_bounds = {
                        'bounds': [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)],
                        'counter': st.session_state.zoom_counter
                    }
                    st.session_state.zoom_counter += 1
                    st.rerun()
        with col2:
            if st.button("Remove", width='stretch', key="remove_thickness", disabled=st.session_state.simulation_running):
                if 'thickness' in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files['thickness']
                if 'thickness_path' in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files['thickness_path']
                if 'thickness_meta' in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files['thickness_meta']
                st.rerun()

    # Run button
    run_simulation = st.button(
        "Run Simulation",
        type="primary",
        width='stretch',
        disabled=st.session_state.simulation_running
    )
    
    st.markdown("---")
    
    # Parameter Sweep Section
    with st.expander("🔬 Parameter Sweep", expanded=False):
        st.caption("Run multiple simulations to explore parameter space")
        
        # Check if we have all required files - be defensive about session state
        has_dem = bool(st.session_state.get('dem_path') and os.path.exists(st.session_state.dem_path))
        has_thickness = 'thickness_path' in st.session_state.get('uploaded_files', {})
        has_release_area = 'release_gdf' in st.session_state.get('uploaded_files', {})
        
        if not (has_dem and has_thickness and has_release_area):
            missing = []
            if not has_dem:
                missing.append("DEM")
            if not has_release_area:
                missing.append("Release Area (shapefile)")
            if not has_thickness:
                missing.append("Thickness Raster")
            
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
            st.info("📁 Upload all required files in the main form below, then reopen this section to enable parameter sweep")
        else:
            # Sweep configuration
            st.markdown("**Parameter Ranges**")
            
            col1, col2 = st.columns(2)
            with col1:
                mu_min = st.number_input("μ min", value=0.025, min_value=0.025, max_value=0.4, step=0.025, format="%.3f")
                mu_max = st.number_input("μ max", value=0.400, min_value=0.025, max_value=0.4, step=0.025, format="%.3f")
                mu_steps = st.number_input("μ steps", value=16, min_value=2, max_value=16, step=1)
            
            with col2:
                xi_min = st.number_input("ξ min", value=250.0, min_value=250.0, max_value=2000.0, step=250.0, format="%.0f")
                xi_max = st.number_input("ξ max", value=2000.0, min_value=250.0, max_value=2000.0, step=250.0, format="%.0f")
                xi_steps = st.number_input("ξ steps", value=8, min_value=2, max_value=8, step=1)
            
            total_sims = mu_steps * xi_steps
            st.info(f"📊 Total simulations: **{total_sims}**")
            
            # Parallel processing settings
            max_cores = mp.cpu_count()
            num_workers = st.slider(
                "Parallel workers",
                min_value=1,
                max_value=max_cores,
                value=min(4, max_cores),
                help=f"Number of simultaneous simulations (max {max_cores} cores available)"
            )
            
            est_time_per_sim = 25  # minutes
            est_total_minutes = (total_sims / num_workers) * est_time_per_sim
            st.caption(f"⏱️ Estimated time: ~{est_total_minutes:.0f} min with {num_workers} workers")
            
            # Run button
            if st.session_state.sweep_running:
                st.warning("**Sweep in progress...**")
                if st.button("❌ Cancel Sweep", width='stretch'):
                    st.session_state.sweep_running = False
                    st.rerun()
            else:
                if st.button("▶️ Run Parameter Sweep", type="primary", width='stretch'):
                    # Create sweep config
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(f"/mnt/data/simulations/sweep_{timestamp}")
                    
                    # Get release area geodataframe
                    release_gdf = st.session_state.uploaded_files.get('release_gdf')
                    if not release_gdf is not None:
                        st.error("Release area not found - please upload again")
                        st.stop()
                    
                    sweep_config = SweepConfig(
                        mu_min=mu_min,
                        mu_max=mu_max,
                        mu_steps=mu_steps,
                        xi_min=xi_min,
                        xi_max=xi_max,
                        xi_steps=xi_steps,
                        dem_path=Path(st.session_state.dem_path),
                        release_path=Path(st.session_state.uploaded_files['thickness_path']),
                        release_gdf=release_gdf,  # Add release geodataframe
                        output_dir=output_dir,
                        max_workers=num_workers
                    )
                    
                    st.session_state.sweep_config = sweep_config
                    st.session_state.sweep_running = True
                    st.rerun()
    
    st.markdown("---")
    
    # Show simulation status if running
    if st.session_state.simulation_running:
        st.markdown("---")
        st.warning("**Simulation Running**")
        st.caption("Please wait... This may take up to 20 minutes.")
        st.markdown("---")
    
    # Display results export if available
    if st.session_state.simulation_results:
        st.markdown("---")
        st.markdown("**Latest Results**")
        
        result_data = st.session_state.simulation_results
        ava_dir = result_data.get('ava_dir')
        stats = result_data.get('stats', {})
        result_files = result_data.get('files', {})
        
        # Layer selector for visualization
        available_layers = []
        layer_map = {}
        if 'velocity' in result_files:
            available_layers.append("Velocity")
            layer_map["Velocity"] = ('velocity', result_files['velocity'])
        if 'thickness' in result_files:
            available_layers.append("Thickness")
            layer_map["Thickness"] = ('thickness', result_files['thickness'])
        if 'pressure' in result_files:
            available_layers.append("Pressure")
            layer_map["Pressure"] = ('pressure', result_files['pressure'])
        
        # Find time/timeInfo file
        if ava_dir:
            peak_dir = Path(ava_dir) / 'Outputs' / 'com1DFA' / 'peakFiles'
            if peak_dir.exists():
                time_files = list(peak_dir.glob('*timeInfo.asc'))
                if time_files:
                    available_layers.append("Time")
                    layer_map["Time"] = ('time', str(time_files[0]))
        
        if available_layers:
            selected_layer_name = st.selectbox(
                "Display Layer",
                available_layers,
                index=0,
                help="Select which result to display on map"
            )
            
            # Update session state with selected layer
            layer_type, layer_path = layer_map[selected_layer_name]
            st.session_state.selected_result_layer = layer_type
            st.session_state.uploaded_files['result_display'] = layer_path
        
        # Show key statistics - compact version
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Velocity", f"{stats.get('max_velocity', 0):.1f} m/s", label_visibility="visible")
            st.metric("Thickness", f"{stats.get('max_thickness', 0):.1f} m", label_visibility="visible")
        with col2:
            st.metric("Pressure", f"{stats.get('max_pressure', 0)/1000:.0f} kPa", label_visibility="visible")
            st.metric("Area", f"{stats.get('affected_area', 0):.1f} ha", label_visibility="visible")
        
        # Download buttons
        if ava_dir and Path(ava_dir).exists():
            # ZIP download
            try:
                zip_buffer = create_results_zip(ava_dir)
                sim_name = Path(ava_dir).name
                st.download_button(
                    label="Download Results (ZIP)",
                    data=zip_buffer,
                    file_name=f"{sim_name}_results.zip",
                    mime="application/zip",
                    width='stretch'
                )
            except Exception as e:
                st.error(f"Error creating ZIP: {str(e)}")
            
            # JSON statistics download
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                label="Download Statistics (JSON)",
                data=stats_json,
                file_name=f"{sim_name}_stats.json",
                mime="application/json",
                width='stretch'
            )
        
        st.markdown("---")
    
    # Previous simulations browser
    with st.expander("Previous Simulations", expanded=False):
        sim_base_dir = Path("/mnt/data/simulations")
        if sim_base_dir.exists():
            # Get all simulation directories sorted by modification time
            sim_dirs = sorted(
                [d for d in sim_base_dir.glob("rock_avalanche_*") if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]  # Last 10 simulations
            
            if sim_dirs:
                for sim_dir in sim_dirs:
                    sim_name = sim_dir.name
                    # Try to read stats
                    stats_file = sim_dir / 'simulation_stats.json'
                    if stats_file.exists():
                        with open(stats_file) as f:
                            sim_stats = json.load(f)
                        velocity = sim_stats.get('max_velocity', 0)
                        area = sim_stats.get('affected_area', 0)
                        st.caption(f"**{sim_name}**")
                        st.caption(f"Velocity: {velocity:.1f} m/s | Area: {area:.1f} ha")
                    else:
                        st.caption(f"**{sim_name}**")
                        st.caption("Stats not available")
                    
                    # Download button for this simulation
                    try:
                        zip_buffer = create_results_zip(sim_dir)
                        st.download_button(
                            label=f"Download",
                            data=zip_buffer,
                            file_name=f"{sim_name}_results.zip",
                            mime="application/zip",
                            width='stretch',
                            key=f"download_{sim_name}"
                        )
                    except Exception as e:
                        st.caption(f"Error: {str(e)}")
                    
                    st.markdown("---")
            else:
                st.caption("No previous simulations found")
        else:
            st.caption("Simulation directory not found")

# Parameter Sweep Progress Monitor
if st.session_state.sweep_running and 'sweep_config' in st.session_state:
    st.title("🔬 Parameter Sweep in Progress")
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        metric_completed = col1.empty()
        metric_workers = col2.empty()
        metric_cpu = col3.empty()
        metric_memory = col4.empty()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        # Details expander
        details_expander = st.expander("📊 Detailed Progress", expanded=True)
        
        def update_progress(progress_data):
            """Callback to update progress display."""
            completed = progress_data['completed']
            total = progress_data['total']
            cpu = progress_data['cpu_percent']
            mem = progress_data['memory']
            elapsed = progress_data['elapsed']
            est_remaining = progress_data['estimated_remaining']
            active = progress_data['active_workers']
            
            # Update metrics
            metric_completed.metric("Progress", f"{completed}/{total}")
            metric_workers.metric("Active", f"{active} workers")
            metric_cpu.metric("CPU", f"{cpu:.1f}%")
            metric_memory.metric("Memory", f"{mem['percent']:.1f}%")
            
            # Update progress bar
            progress = completed / total if total > 0 else 0
            progress_bar.progress(progress)
            
            # Status text
            status_text.info(f"Completed: {completed}/{total} simulations")
            
            # Time estimate
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            if est_remaining:
                remaining_str = str(timedelta(seconds=int(est_remaining)))
                time_text.caption(f"⏱️ Elapsed: {elapsed_str} | Remaining: ~{remaining_str}")
            else:
                time_text.caption(f"⏱️ Elapsed: {elapsed_str}")
            
            # Detailed status table
            with details_expander:
                statuses = progress_data['statuses']
                status_df = []
                for (mu, xi), status in statuses.items():
                    status_emoji = {
                        'pending': '⏳',
                        'running': '▶️',
                        'completed': '✅',
                        'failed': '❌'
                    }.get(status.status, '❓')
                    
                    status_df.append({
                        'Status': f"{status_emoji} {status.status}",
                        'μ': f"{mu:.3f}",
                        'ξ': f"{xi:.0f}",
                        'Duration': f"{status.duration():.0f}s" if status.duration() > 0 else "-"
                    })
                
                st.dataframe(status_df, width='stretch', height=300)
        
        # Run sweep
        try:
            import multiprocessing as mp
            avaframe_path = Path(avaframe_path)
            
            results = run_parameter_sweep(
                st.session_state.sweep_config,
                avaframe_path,
                progress_callback=update_progress
            )
            
            # Sweep complete
            st.session_state.sweep_running = False
            st.session_state.sweep_results = results
            
            st.success(f"✅ Parameter sweep complete!")
            st.info(f"✓ Successful: {results['successful']}/{results['total']}")
            if results['failed'] > 0:
                st.warning(f"⚠️ Failed: {results['failed']}/{results['total']}")
            
            st.markdown(f"**Results saved to:** `{results['output_dir']}`")
            
            # Link to explorer
            st.markdown("---")
            st.markdown("### 📊 Analyze Results")
            st.info("Open the Runout Explorer to visualize and analyze your parameter sweep results.")
            st.code(f"streamlit run analysis/runout_explorer.py", language="bash")
            st.caption(f"Point it to: `{results['output_dir']}`")
            
            if st.button("🔄 Run Another Sweep"):
                st.session_state.sweep_results = None
                st.rerun()
            
        except Exception as e:
            st.error(f"Error during sweep: {str(e)}")
            st.session_state.sweep_running = False
            if st.button("Try Again"):
                st.rerun()
    
    st.stop()  # Don't show main form while sweep is running

# End of sweep progress monitor

# Rock Avalanche Interface (snow avalanche tool will be separate)
st.markdown("**Rock Avalanche Simulation**")

if run_simulation:
    if not st.session_state.dem_path or not os.path.exists(st.session_state.dem_path):
        st.error("DEM file not found")
    elif 'release_gdf' not in st.session_state.uploaded_files:
        st.error("Upload release area first")
    elif 'thickness_path' not in st.session_state.uploaded_files:
        st.error("Upload thickness raster first")
    else:
        # Set simulation running flag
        st.session_state.simulation_running = True
        
        with st.spinner("Running simulation... (Up to 20 min, please be patient)"):
            release_gdf = st.session_state.uploaded_files['release_gdf']
            thickness_path = st.session_state.uploaded_files['thickness_path']
            dem_path = st.session_state.dem_path
            
            result_files, stats, ava_dir = run_rock_avalanche_simulation(
                release_gdf,
                thickness_path,
                dem_path
            )
            
            # Clear simulation running flag
            st.session_state.simulation_running = False
            
            if result_files and stats:
                st.success("Simulation complete")
                st.session_state.simulation_results = {
                    'type': 'rock',
                    'files': result_files,
                    'stats': stats,
                    'ava_dir': ava_dir
                }
                
                # Store velocity as default display layer
                if 'velocity' in result_files:
                    st.session_state.selected_result_layer = 'velocity'
                    st.session_state.uploaded_files['result_display'] = result_files['velocity']
                st.rerun()
            else:
                st.error("Simulation failed")

st.markdown("---")

# Custom CSS to optimize layout
st.markdown("""
    <style>
    /* Adjust padding to avoid overlap with Streamlit header */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Remove controls bar styling */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Make iframe container fill remaining height */
    .stComponentContainer iframe {
        min-height: calc(100vh - 220px) !important;
        height: calc(100vh - 220px) !important;
        border: none;
    }
    
    /* Force light background for map container in dark mode */
    .stComponentContainer {
        background-color: white !important;
        color-scheme: light !important;
    }
    
    /* Ensure dropdowns appear above everything */
    .stSelectbox {
        z-index: 999;
    }
    
    /* Hide Streamlit branding */
    footer {
        display: none;
    }
    
    /* Reduce header height */
    header[data-testid="stHeader"] {
        background-color: transparent;
        height: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main content area - simple kartlag selector
basemap_choice = st.selectbox(
    "Kartlag",
    ["Grunnkart", "DTM 10m", "Helning", "Geologi"],
    index=0,
    help="Velg bakgrunnskart"
)

# Prepare uploaded features for display
uploaded_features = None
if 'release_gdf' in st.session_state.uploaded_files:
    gdf = st.session_state.uploaded_files['release_gdf']
    # Convert to GeoJSON in EPSG:25833
    uploaded_features = json.loads(gdf.to_json())

# Prepare raster overlay - prioritize simulation results
raster_overlay = None
legend_info = None

if 'result_display' in st.session_state.uploaded_files:
    # Display selected simulation result layer
    raster_path = st.session_state.uploaded_files['result_display']
    layer_type = st.session_state.selected_result_layer
    raster_overlay = raster_to_image_overlay(raster_path, layer_type)
    
    # Create legend if overlay exists
    if raster_overlay:
        legend_info = {
            'label': raster_overlay.get('label', 'Value'),
            'min': raster_overlay['extent'][0],
            'max': raster_overlay['extent'][1],
            'unit': raster_overlay.get('unit', ''),
            'colormap': layer_type
        }
elif 'thickness_path' in st.session_state.uploaded_files:
    # Display thickness raster (input data)
    raster_path = st.session_state.uploaded_files['thickness_path']
    raster_overlay = raster_to_image_overlay(raster_path, 'thickness')

# Display legend if available
if legend_info:
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white; padding: 10px; 
                border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 1000;">
        <div style="font-weight: bold; margin-bottom: 5px;">{legend_info['label']}</div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 12px;">{legend_info['min']:.1f} {legend_info['unit']}</div>
            <div style="width: 200px; height: 20px; background: linear-gradient(to right, 
                {'blue, yellow, red' if legend_info['colormap'] == 'velocity' else
                 'white, blue' if legend_info['colormap'] == 'thickness' else
                 'green, yellow, red' if legend_info['colormap'] == 'pressure' else
                 'purple, red'});
                border: 1px solid #ccc; border-radius: 3px;">
            </div>
            <div style="font-size: 12px;">{legend_info['max']:.1f} {legend_info['unit']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Use a large fixed height that will be overridden by CSS
map_height = 900

# Render OpenLayers map - fills remaining viewport via CSS
map_data = ol_map(
    center=st.session_state.map_center,
    zoom=st.session_state.map_zoom,
    basemap=basemap_choice,
    show_hillshade=False,
    enable_drawing=False,  # Drawing disabled for rock avalanche (use file upload)
    uploaded_features=uploaded_features,
    raster_overlay=raster_overlay,
    zoom_to_bounds=st.session_state.zoom_to_bounds,
    height=map_height,
    key="main_map"
)

# Clear zoom_to_bounds after it's been sent to component
if st.session_state.zoom_to_bounds is not None:
    st.session_state.zoom_to_bounds = None

# Handle drawn features only
if map_data and map_data.get('type') == 'feature_drawn':
    st.success("✅ Release area drawn!")
    st.session_state.uploaded_files['drawn_feature'] = map_data['feature']

