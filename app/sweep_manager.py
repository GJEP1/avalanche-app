"""
Parameter Sweep Manager
=======================
Manages parallel execution of AVAFRAME parameter sweeps with real-time monitoring.

Author: Integration for avalanche-app
"""

import streamlit as st
import numpy as np
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import time
import psutil
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import subprocess
import shutil


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    mu_min: float
    mu_max: float
    mu_steps: int
    xi_min: float
    xi_max: float
    xi_steps: int
    dem_path: Path
    release_path: Path  # Path to thickness raster
    release_gdf: object  # GeoDataFrame with release area polygon
    output_dir: Path
    max_workers: int = 4
    rho: float = 2500.0
    cellsize: float = 10.0
    
    def get_parameter_grid(self):
        """Generate parameter grid."""
        mu_values = np.linspace(self.mu_min, self.mu_max, self.mu_steps)
        xi_values = np.linspace(self.xi_min, self.xi_max, self.xi_steps)
        return [(mu, xi) for mu in mu_values for xi in xi_values]
    
    def total_simulations(self):
        """Total number of simulations."""
        return self.mu_steps * self.xi_steps


@dataclass
class SimulationStatus:
    """Status of a single simulation."""
    mu: float
    xi: float
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_msg: Optional[str] = None
    
    def duration(self):
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0


class SweepMonitor:
    """Monitor system resources and sweep progress."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def get_cpu_percent(self):
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self):
        """Get memory usage in MB."""
        mem = psutil.virtual_memory()
        return {
            'used_mb': mem.used / 1024 / 1024,
            'available_mb': mem.available / 1024 / 1024,
            'percent': mem.percent
        }
    
    def get_elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def estimate_remaining_time(self, completed: int, total: int):
        """Estimate remaining time based on progress."""
        if completed == 0:
            return None
        elapsed = self.get_elapsed_time()
        avg_time_per_sim = elapsed / completed
        remaining_sims = total - completed
        return avg_time_per_sim * remaining_sims


def create_simulation_config(mu: float, xi: float, cellsize: float, rho: float) -> str:
    """Generate AVAFRAME configuration for rock avalanche."""
    return f"""# AVAFRAME Rock Avalanche Configuration
# Generated: {datetime.now().isoformat()}
# Parameters: mu={mu:.4f}, xi={xi:.1f}

[com1DFA_com1DFA_override]
meshCellSize = {cellsize}
rho = {rho}
frictModel = Voellmy
muvoellmy = {mu}
xsivoellmy = {xi}
sphOption = 3
massPerPart = 280000
deltaTh = 2.0
splitOption = 1
resType = ppr|pft|pfv|FT
tSteps = 0:1

[com1DFA_com1DFA_defaultConfig]
"""


def setup_simulation_folder(config: SweepConfig, mu: float, xi: float) -> tuple[Path, bool]:
    """
    Create simulation folder structure and prepare input files.
    Returns (sim_folder, success).
    
    This function replicates the DEM/thickness processing from the main app
    to ensure proper AVAFRAME compatibility.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    import geopandas as gpd
    
    folder_name = f"mu_{mu:.3f}_turb_{xi:.0f}"
    sim_folder = config.output_dir / folder_name
    
    try:
        # Create structure
        inputs_folder = sim_folder / "Inputs"
        rel_folder = inputs_folder / "REL"
        
        for folder in [sim_folder, inputs_folder, rel_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Create other required folders
        for subfolder in ["RES", "ENT", "LINES", "POINTS", "POLYGONS"]:
            (inputs_folder / subfolder).mkdir(exist_ok=True)
        
        # Get bounding box from release area with buffer
        bounds = config.release_gdf.total_bounds  # [minx, miny, maxx, maxy]
        buffer = 4000  # meters - large buffer to capture full runout zone
        bbox = [bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer]
        
        # STEP 1: Crop DEM to simulation area
        with rasterio.open(config.dem_path) as src_dem:
            # Calculate window for cropping
            window = rasterio.windows.from_bounds(*bbox, src_dem.transform)
            
            # Read cropped DEM
            dem_data = src_dem.read(1, window=window)
            dem_transform = src_dem.window_transform(window)
            dem_crs = src_dem.crs
            
            nrows, ncols = dem_data.shape
            cellsize = abs(dem_transform[0])
            
            # Handle nodata in DEM
            dem_data = np.nan_to_num(dem_data, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
            if src_dem.nodata is not None:
                dem_data = np.where(dem_data == src_dem.nodata, -9999.0, dem_data)
            
            # Calculate grid parameters (center coordinates for AAIGRID)
            xllcenter = dem_transform[2] + cellsize / 2.0
            yllcenter = dem_transform[5] - (nrows * cellsize) + cellsize / 2.0
            
            # Write DEM in AAIGRID format
            dem_dest = inputs_folder / 'DEM.asc'
            _write_aaigrid(dem_dest, dem_data, ncols, nrows, 
                          xllcenter, yllcenter, cellsize, -9999, is_thickness=False)
            
            # Write .prj file
            with open(dem_dest.with_suffix('.prj'), 'w') as f:
                f.write(dem_crs.to_wkt())
        
        # STEP 2: Resample thickness raster to EXACT DEM grid
        thickness_resampled = np.zeros((nrows, ncols), dtype=np.float64)
        
        with rasterio.open(config.release_path) as src_thick:
            # Reproject/resample thickness to DEM grid
            reproject(
                source=rasterio.band(src_thick, 1),
                destination=thickness_resampled,
                src_transform=src_thick.transform,
                src_crs=src_thick.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )
        
        # Handle nodata and ensure non-negative
        thickness_resampled = np.nan_to_num(thickness_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        thickness_resampled = np.maximum(thickness_resampled, 0.0)
        
        # Write thickness in AAIGRID format
        thickness_dest = rel_folder / 'relTh.asc'
        _write_aaigrid(thickness_dest, thickness_resampled, ncols, nrows,
                      xllcenter, yllcenter, cellsize, -9999, is_thickness=True)
        
        # Write .prj file for thickness
        with open(thickness_dest.with_suffix('.prj'), 'w') as f:
            f.write(dem_crs.to_wkt())
        
        # STEP 3: Create configuration file
        cfg_content = create_simulation_config(mu, xi, config.cellsize, config.rho)
        (sim_folder / "local_com6RockAvalancheCfg.ini").write_text(cfg_content)
        
        return sim_folder, True
        
    except Exception as e:
        # Log error but don't crash the whole sweep
        print(f"Error setting up {folder_name}: {e}")
        return sim_folder, False


def _write_aaigrid(filepath: Path, data: np.ndarray, ncols: int, nrows: int,
                   xllcenter: float, yllcenter: float, cellsize: float, 
                   nodata: float, is_thickness: bool = False):
    """
    Write data to AAIGRID ASCII format.
    
    CRITICAL: Check EACH value to ensure no NaN/Inf is written to file.
    For thickness: NaN/Inf → 0.0 (not nodata, as AVAFRAME treats non-zero as release)
    For DEM: NaN/Inf → nodata value
    """
    with open(filepath, 'w') as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcenter     {xllcenter:.6f}\n")
        f.write(f"yllcenter     {yllcenter:.6f}\n")
        f.write(f"cellsize      {cellsize:.6f}\n")
        f.write(f"NODATA_value  {nodata:.0f}\n")
        
        # Write data - check EACH value individually
        for row in data:
            values = []
            for val in row:
                # Check if value is finite (not NaN, not Inf)
                if not np.isfinite(val):
                    if is_thickness:
                        values.append("0.000000")
                    else:
                        values.append(f"{nodata:.6f}")
                else:
                    # Format the value
                    formatted = f"{float(val):.6f}"
                    # Double-check: never write "nan" or "inf" as string
                    if 'nan' in formatted.lower() or 'inf' in formatted.lower():
                        if is_thickness:
                            values.append("0.000000")
                        else:
                            values.append(f"{nodata:.6f}")
                    else:
                        values.append(formatted)
            
            f.write(' '.join(values) + '\n')


def run_single_simulation(sim_folder: Path, avaframe_path: Path) -> bool:
    """Run a single AVAFRAME simulation."""
    runner_script = f'''
import sys
sys.path.insert(0, r"{avaframe_path}")

from avaframe.com6RockAvalanche import com6RockAvalanche
from avaframe.in3Utils import cfgUtils, logUtils

avalancheDir = r"{sim_folder}"
logUtils.initiateLogger(avalancheDir)

# Configure main config with correct avalanche directory
cfgMain = cfgUtils.getGeneralConfig()
cfgMain['MAIN']['avalancheDir'] = str(avalancheDir)

# Get com6RockAvalanche config with local override
cfg = cfgUtils.getModuleConfig(
    com6RockAvalanche, 
    fileOverride=r"{sim_folder / 'local_com6RockAvalancheCfg.ini'}"
)

# Run simulation
com6RockAvalanche.com6RockAvalancheMain(cfgMain, rockAvalancheCfg=cfg)
'''
    
    runner_path = sim_folder / "run_simulation.py"
    runner_path.write_text(runner_script)
    
    try:
        result = subprocess.run(
            ["python", str(runner_path)],
            capture_output=True,
            text=True,
            cwd=str(sim_folder),
            timeout=3600
        )
        return result.returncode == 0
    except Exception:
        return False


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, 
                   config: SweepConfig, avaframe_path: Path):
    """Worker process for running simulations."""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
        
        mu, xi = task
        status = SimulationStatus(mu=mu, xi=xi, status='running', start_time=time.time())
        result_queue.put(('update', status))
        
        try:
            sim_folder, setup_success = setup_simulation_folder(config, mu, xi)
            
            if not setup_success:
                status.status = 'failed'
                status.end_time = time.time()
                status.error_msg = "Failed to setup simulation folder"
            else:
                success = run_single_simulation(sim_folder, avaframe_path)
                status.status = 'completed' if success else 'failed'
                status.end_time = time.time()
                if not success:
                    status.error_msg = "Simulation failed - check logs"
        except Exception as e:
            status.status = 'failed'
            status.end_time = time.time()
            status.error_msg = str(e)
        
        result_queue.put(('complete', status))


def run_parameter_sweep(config: SweepConfig, avaframe_path: Path,
                       progress_callback=None) -> Dict:
    """
    Run parameter sweep with parallel processing.
    
    Args:
        config: Sweep configuration
        avaframe_path: Path to AVAFRAME installation
        progress_callback: Optional callback(status_dict) for progress updates
    
    Returns:
        Dictionary with sweep results
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sweep config
    sweep_metadata = {
        'created': datetime.now().isoformat(),
        'mu_range': [config.mu_min, config.mu_max, config.mu_steps],
        'xi_range': [config.xi_min, config.xi_max, config.xi_steps],
        'total_simulations': config.total_simulations(),
        'max_workers': config.max_workers
    }
    (config.output_dir / "sweep_config.json").write_text(json.dumps(sweep_metadata, indent=2))
    
    # Get parameter grid
    param_grid = config.get_parameter_grid()
    total = len(param_grid)
    
    # Initialize tracking
    statuses = {(mu, xi): SimulationStatus(mu, xi, 'pending') for mu, xi in param_grid}
    monitor = SweepMonitor()
    
    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add all tasks
    for params in param_grid:
        task_queue.put(params)
    
    # Add poison pills
    for _ in range(config.max_workers):
        task_queue.put(None)
    
    # Start workers
    workers = []
    for _ in range(config.max_workers):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue, config, avaframe_path))
        p.start()
        workers.append(p)
    
    # Monitor progress
    completed = 0
    while completed < total:
        try:
            msg_type, status = result_queue.get(timeout=1)
            
            if msg_type == 'update':
                statuses[(status.mu, status.xi)] = status
            elif msg_type == 'complete':
                statuses[(status.mu, status.xi)] = status
                completed += 1
            
            # Call progress callback with current state
            if progress_callback:
                progress_data = {
                    'completed': completed,
                    'total': total,
                    'statuses': statuses,
                    'cpu_percent': monitor.get_cpu_percent(),
                    'memory': monitor.get_memory_usage(),
                    'elapsed': monitor.get_elapsed_time(),
                    'estimated_remaining': monitor.estimate_remaining_time(completed, total),
                    'active_workers': sum(1 for s in statuses.values() if s.status == 'running')
                }
                progress_callback(progress_data)
        
        except:
            # Timeout - update monitoring data anyway
            if progress_callback:
                progress_data = {
                    'completed': completed,
                    'total': total,
                    'statuses': statuses,
                    'cpu_percent': monitor.get_cpu_percent(),
                    'memory': monitor.get_memory_usage(),
                    'elapsed': monitor.get_elapsed_time(),
                    'estimated_remaining': monitor.estimate_remaining_time(completed, total),
                    'active_workers': sum(1 for s in statuses.values() if s.status == 'running')
                }
                progress_callback(progress_data)
    
    # Wait for workers to finish
    for p in workers:
        p.join()
    
    # Final results
    successful = sum(1 for s in statuses.values() if s.status == 'completed')
    failed = sum(1 for s in statuses.values() if s.status == 'failed')
    
    return {
        'total': total,
        'successful': successful,
        'failed': failed,
        'statuses': statuses,
        'duration': monitor.get_elapsed_time(),
        'output_dir': config.output_dir
    }
