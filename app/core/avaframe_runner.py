"""
AvaFrame Simulation Runner Module.

Provides functions for running single simulations and parameter sweeps
using AVAFRAME com6RockAvalanche module.
"""

import os
import sys
import json
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import features as rio_features
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List, Tuple, Dict, Any
from datetime import datetime
import traceback

# Get AvaFrame path
AVAFRAME_PATH = Path(__file__).parent.parent.parent / 'AvaFrame'


def get_avaframe_version() -> str:
    """Get AvaFrame version from git or package metadata."""
    try:
        # Try git describe first
        result = subprocess.run(
            ['git', 'describe', '--tags', '--always'],
            cwd=AVAFRAME_PATH,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback to package version
    try:
        import importlib.metadata
        return importlib.metadata.version('avaframe')
    except Exception:
        return "unknown"


@dataclass
class SimulationConfig:
    """Configuration for a single AVAFRAME simulation."""
    
    # Voellmy friction parameters
    mu: float = 0.035           # Friction coefficient
    xi: float = 700.0           # Turbulence coefficient [m/s²]
    
    # Material properties
    rho: float = 2500.0         # Density [kg/m³]
    
    # Mesh settings
    mesh_cell_size: Optional[float] = None  # None = use DEM resolution
    
    # SPH settings
    sph_option: int = 3
    mass_per_part: float = 500000.0  # Fast default (fewer particles)
    delta_th: float = 4.0            # Fast default (fewer particles)
    split_option: int = 1
    
    # Output settings
    result_types: str = "pft|pfv|ppr|FT"
    
    def to_ini_content(self, mesh_size: float) -> str:
        """Generate INI file content for AVAFRAME."""
        return f"""### Local Config File - Rock Avalanche settings

[GENERAL]

[com1DFA_com1DFA_override]

defaultConfig = True

# Mesh settings
meshCellSize = {mesh_size}
meshCellSizeThreshold = {mesh_size * 10.0}

# Output
resType = {self.result_types}

# Material properties
rho = {self.rho}

# SPH parameters
sphOption = {self.sph_option}
massPerPart = {self.mass_per_part}
deltaTh = {self.delta_th}
splitOption = {self.split_option}

# Friction model
frictModel = Voellmy
muvoellmy = {self.mu}
xsivoellmy = {self.xi}
"""


@dataclass
class SimulationResult:
    """Result from a single simulation."""
    sim_id: str
    mu: float
    xi: float
    
    # Output paths
    sim_dir: str = ""
    thickness_path: str = ""
    velocity_path: str = ""
    pressure_path: str = ""
    
    # Computed metrics
    max_velocity: float = 0.0
    max_thickness: float = 0.0
    max_pressure: float = 0.0
    affected_area_ha: float = 0.0
    runout_m: float = 0.0
    
    # Status
    success: bool = False
    error: Optional[str] = None
    runtime_s: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    
    # Parameter ranges
    mu_min: float = 0.025
    mu_max: float = 0.1
    mu_steps: int = 4
    
    xi_min: float = 500.0
    xi_max: float = 1500.0
    xi_steps: int = 4
    
    # Input data
    dem_path: Path = None
    thickness_path: Path = None  # SLBL output or manual upload
    release_gdf: gpd.GeoDataFrame = None  # Release polygon
    
    # Output
    output_dir: Path = None
    
    # Processing
    max_workers: int = 16  # Parallel workers for sweep
    timeout_per_sim: int = 7200  # 2 hours - large volumes may need extended runtime
    
    @property
    def mu_values(self) -> List[float]:
        return list(np.linspace(self.mu_min, self.mu_max, self.mu_steps))
    
    @property
    def xi_values(self) -> List[float]:
        return list(np.linspace(self.xi_min, self.xi_max, self.xi_steps))
    
    @property
    def total_simulations(self) -> int:
        return self.mu_steps * self.xi_steps


def prepare_simulation_directory(
    output_dir: Path,
    dem_path: Path,
    thickness_path: Path,
    release_gdf: gpd.GeoDataFrame,
    config: SimulationConfig,
    sim_name: str
) -> Tuple[Path, float]:
    """
    Prepare simulation directory with all required inputs.

    Returns:
        Tuple of (simulation directory path, mesh cell size)
    """
    sim_dir = output_dir / sim_name

    # Clean up existing Work and Outputs directories (AvaFrame requires clean dirs)
    for cleanup_dir in ['Work', 'Outputs']:
        cleanup_path = sim_dir / cleanup_dir
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path)

    # Create directory structure
    inputs_dir = sim_dir / 'Inputs'
    rel_dir = inputs_dir / 'REL'

    for d in [sim_dir, inputs_dir, rel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create empty subdirectories AVAFRAME expects
    for subdir in ['ENT', 'RES', 'LINES', 'POINTS', 'POLYGONS']:
        (inputs_dir / subdir).mkdir(exist_ok=True)
    
    # Get bounds from release area with buffer
    bounds = release_gdf.total_bounds
    buffer = 4000  # meters
    bbox = [
        bounds[0] - buffer,
        bounds[1] - buffer,
        bounds[2] + buffer,
        bounds[3] + buffer
    ]
    
    # Read and crop DEM
    with rasterio.open(dem_path) as src_dem:
        window = rasterio.windows.from_bounds(*bbox, src_dem.transform)
        dem_data = src_dem.read(1, window=window)
        dem_transform = src_dem.window_transform(window)
        dem_crs = src_dem.crs
        
        nrows, ncols = dem_data.shape
        cellsize = abs(dem_transform[0])
        
        # Handle nodata
        dem_data = np.nan_to_num(dem_data, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
        if src_dem.nodata is not None:
            dem_data = np.where(dem_data == src_dem.nodata, -9999.0, dem_data)
        
        # Calculate grid parameters
        xllcenter = dem_transform[2] + cellsize / 2.0
        yllcenter = dem_transform[5] - (nrows * cellsize) + cellsize / 2.0
        
        # Write DEM
        dem_dest = inputs_dir / 'DEM.asc'
        _write_aaigrid(dem_dest, dem_data, ncols, nrows, xllcenter, yllcenter, cellsize, -9999)
        
        # Write projection file
        with open(dem_dest.with_suffix('.prj'), 'w') as f:
            f.write(dem_crs.to_wkt())
    
    # Resample thickness to DEM grid
    thickness_resampled = np.zeros((nrows, ncols), dtype=np.float64)
    
    with rasterio.open(thickness_path) as src_thick:
        reproject(
            source=rasterio.band(src_thick, 1),
            destination=thickness_resampled,
            src_transform=src_thick.transform,
            src_crs=src_thick.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )
    
    # Clean up NaN values
    thickness_resampled = np.nan_to_num(thickness_resampled, nan=0.0, posinf=0.0, neginf=0.0)
    thickness_resampled = np.where(thickness_resampled < 0, 0.0, thickness_resampled)
    
    # Mask to release area
    release_mask = np.zeros((nrows, ncols), dtype=np.uint8)
    shapes = [(geom, 1) for geom in release_gdf.geometry]
    rio_features.rasterize(
        shapes=shapes,
        out=release_mask,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8
    )
    thickness_resampled = np.where(release_mask == 1, thickness_resampled, 0.0)
    
    # Write thickness
    thickness_dest = rel_dir / 'relTh.asc'
    _write_aaigrid(thickness_dest, thickness_resampled, ncols, nrows, 
                   xllcenter, yllcenter, cellsize, -9999, is_thickness=True)
    
    with open(thickness_dest.with_suffix('.prj'), 'w') as f:
        f.write(dem_crs.to_wkt())
    
    # Write configuration file
    mesh_size = config.mesh_cell_size or cellsize
    config_content = config.to_ini_content(mesh_size)
    config_path = sim_dir / 'local_com6RockAvalancheCfg.ini'
    config_path.write_text(config_content)
    
    return sim_dir, mesh_size


def _write_aaigrid(filepath: Path, data: np.ndarray, ncols: int, nrows: int,
                   xllcenter: float, yllcenter: float, cellsize: float,
                   nodata_value: float = -9999, is_thickness: bool = False):
    """Write ESRI ASCII Grid file."""
    with open(filepath, 'w') as f:
        f.write(f"ncols        {ncols}\n")
        f.write(f"nrows        {nrows}\n")
        f.write(f"xllcenter    {xllcenter:.2f}\n")
        f.write(f"yllcenter    {yllcenter:.2f}\n")
        f.write(f"cellsize     {cellsize:.15f}\n")
        f.write(f"nodata_value {nodata_value:.2f}\n")
        
        for row in data:
            values = []
            for val in row:
                if not np.isfinite(val):
                    values.append("0.000000" if is_thickness else f"{nodata_value:.6f}")
                else:
                    values.append(f"{float(val):.6f}")
            f.write(' '.join(values) + '\n')


def run_single_simulation(
    sim_dir: Path,
    timeout: int = 1800,
    created_by: str = None,
    input_files: Dict[str, str] = None,
    job_id: str = None
) -> SimulationResult:
    """
    Run a single AVAFRAME simulation.

    Args:
        sim_dir: Prepared simulation directory
        timeout: Timeout in seconds
        created_by: Username who initiated the simulation
        input_files: Dict of input file paths for metadata tracking
        job_id: Optional job ID for process tracking (enables cancellation)

    Returns:
        SimulationResult with metrics and paths
    """
    # Import process tracker if job_id provided
    process_tracker = None
    if job_id:
        try:
            from core.job_queue import get_process_tracker
            process_tracker = get_process_tracker()
        except ImportError:
            pass

    # Parse config for all parameters
    config_path = sim_dir / 'local_com6RockAvalancheCfg.ini'
    sim_config = _parse_full_config(config_path)
    mu = sim_config.get('mu', 0.035)
    xi = sim_config.get('xi', 700.0)

    result = SimulationResult(
        sim_id=sim_dir.name,
        mu=mu,
        xi=xi,
        sim_dir=str(sim_dir)
    )

    start_time = time.time()
    start_datetime = datetime.now()
    proc_stdout = ""
    proc_stderr = ""
    proc = None

    try:
        # Create runner script
        runner_script = _create_runner_script(sim_dir)

        # Run simulation using Popen for process tracking
        proc = subprocess.Popen(
            [sys.executable, str(runner_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True  # Create new process group for clean termination
        )

        # Register process for cancellation support
        if process_tracker and job_id:
            process_tracker.register(job_id, proc.pid)

        try:
            proc_stdout, proc_stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the process group on timeout
            import os
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except:
                proc.kill()
            proc.wait()
            raise

        result.runtime_s = time.time() - start_time

        if proc.returncode != 0:
            result.error = proc_stderr[-2000:] if proc_stderr else "Unknown error"
        else:
            # Collect results
            output_dir = sim_dir / 'Outputs' / 'com1DFA' / 'peakFiles'

            if output_dir.exists():
                # Find result files
                for f in output_dir.glob('*.asc'):
                    if '_pft' in f.name:
                        result.thickness_path = str(f)
                    elif '_pfv' in f.name:
                        result.velocity_path = str(f)
                    elif '_ppr' in f.name:
                        result.pressure_path = str(f)

                # Compute statistics
                result = _compute_simulation_stats(result)
                result.success = True
            else:
                result.error = "No output files generated"

    except subprocess.TimeoutExpired:
        result.error = f"Simulation timed out after {timeout}s"
        result.runtime_s = timeout
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        result.runtime_s = time.time() - start_time
    finally:
        # Unregister process
        if process_tracker and job_id and proc:
            process_tracker.unregister(job_id, proc.pid)

    # Save reproducibility files
    _save_reproducibility_files(
        sim_dir=sim_dir,
        sim_config=sim_config,
        result=result,
        start_datetime=start_datetime,
        created_by=created_by,
        input_files=input_files,
        stdout=proc_stdout,
        stderr=proc_stderr
    )

    return result


def _parse_full_config(config_path: Path) -> Dict[str, Any]:
    """Parse all simulation parameters from config file."""
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)

    params = {
        'mu': 0.035,
        'xi': 700.0,
        'rho': 2500.0,
        'sph_option': 3,
        'mass_per_part': 280000.0,
        'delta_th': 1.0,
        'split_option': 1,
        'mesh_cell_size': None,
        'result_types': 'pft|pfv|ppr|FT'
    }

    for section in config.sections():
        if config.has_option(section, 'muvoellmy'):
            params['mu'] = config.getfloat(section, 'muvoellmy')
        if config.has_option(section, 'xsivoellmy'):
            params['xi'] = config.getfloat(section, 'xsivoellmy')
        if config.has_option(section, 'rho'):
            params['rho'] = config.getfloat(section, 'rho')
        if config.has_option(section, 'sphOption'):
            params['sph_option'] = config.getint(section, 'sphOption')
        if config.has_option(section, 'massPerPart'):
            params['mass_per_part'] = config.getfloat(section, 'massPerPart')
        if config.has_option(section, 'deltaTh'):
            params['delta_th'] = config.getfloat(section, 'deltaTh')
        if config.has_option(section, 'splitOption'):
            params['split_option'] = config.getint(section, 'splitOption')
        if config.has_option(section, 'meshCellSize'):
            params['mesh_cell_size'] = config.getfloat(section, 'meshCellSize')
        if config.has_option(section, 'resType'):
            params['result_types'] = config.get(section, 'resType')

    return params


def _save_reproducibility_files(
    sim_dir: Path,
    sim_config: Dict[str, Any],
    result: SimulationResult,
    start_datetime: datetime,
    created_by: str = None,
    input_files: Dict[str, str] = None,
    stdout: str = "",
    stderr: str = ""
):
    """Save files needed to reproduce the simulation."""

    # 1. Save simulation_config.json - all parameters in readable format
    config_json = {
        'mu': sim_config.get('mu'),
        'xi': sim_config.get('xi'),
        'rho': sim_config.get('rho'),
        'sph_option': sim_config.get('sph_option'),
        'mass_per_part': sim_config.get('mass_per_part'),
        'delta_th': sim_config.get('delta_th'),
        'split_option': sim_config.get('split_option'),
        'mesh_cell_size': sim_config.get('mesh_cell_size'),
        'result_types': sim_config.get('result_types')
    }
    config_path = sim_dir / 'simulation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)

    # 2. Save avaframe_log.txt - full simulation output
    log_path = sim_dir / 'avaframe_log.txt'
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AVAFRAME SIMULATION LOG\n")
        f.write(f"Simulation: {result.sim_id}\n")
        f.write(f"Started: {start_datetime.isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- STDOUT ---\n")
        f.write(stdout if stdout else "(no output)\n")
        f.write("\n")

        if stderr:
            f.write("--- STDERR ---\n")
            f.write(stderr)
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Completed: {datetime.now().isoformat()}\n")
        f.write(f"Runtime: {result.runtime_s:.2f} seconds\n")
        f.write(f"Success: {result.success}\n")
        if result.error:
            f.write(f"Error: {result.error}\n")

    # 3. Save metadata.json - provenance info
    metadata = {
        'created_at': start_datetime.isoformat(),
        'completed_at': datetime.now().isoformat(),
        'created_by': created_by or 'unknown',
        'avaframe_version': get_avaframe_version(),
        'runtime_seconds': result.runtime_s,
        'success': result.success,
        'error': result.error,
        'input_files': input_files or {},
        'output_files': {
            'thickness': result.thickness_path,
            'velocity': result.velocity_path,
            'pressure': result.pressure_path
        },
        'metrics': {
            'max_velocity': result.max_velocity,
            'max_thickness': result.max_thickness,
            'max_pressure': result.max_pressure,
            'affected_area_ha': result.affected_area_ha
        }
    }
    metadata_path = sim_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def _parse_config_params(config_path: Path) -> Tuple[float, float]:
    """Parse mu and xi from config file."""
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    
    mu = 0.035
    xi = 700.0
    
    for section in config.sections():
        if config.has_option(section, 'muvoellmy'):
            mu = config.getfloat(section, 'muvoellmy')
        if config.has_option(section, 'xsivoellmy'):
            xi = config.getfloat(section, 'xsivoellmy')
    
    return mu, xi


def _create_runner_script(sim_dir: Path) -> Path:
    """Create standalone runner script."""
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.insert(0, "{AVAFRAME_PATH}")

from avaframe.in3Utils import cfgUtils
from avaframe.com6RockAvalanche import com6RockAvalanche

ava_dir = "{sim_dir}"
local_config = "{sim_dir / 'local_com6RockAvalancheCfg.ini'}"

cfgMain = cfgUtils.getGeneralConfig()
cfgMain['MAIN']['avalancheDir'] = ava_dir

cfg = cfgUtils.getModuleConfig(com6RockAvalanche, fileOverride=local_config)

dem, plotDict, reportDictList, simDF = com6RockAvalanche.com6RockAvalancheMain(
    cfgMain, rockAvalancheCfg=cfg
)

print(f"Completed {{len(simDF)}} simulation(s)")
'''
    
    runner_path = sim_dir / 'run_simulation.py'
    runner_path.write_text(script_content)
    return runner_path


def _compute_simulation_stats(result: SimulationResult) -> SimulationResult:
    """Compute statistics from simulation output files."""
    try:
        # Read velocity for max velocity and affected area
        if result.velocity_path and Path(result.velocity_path).exists():
            with rasterio.open(result.velocity_path) as src:
                data = src.read(1)
                cellsize = src.transform[0]
                
                valid = np.isfinite(data) & (data > 0)
                if np.any(valid):
                    result.max_velocity = float(np.nanmax(data))
                    result.affected_area_ha = float(np.sum(valid) * cellsize * cellsize / 10000)
        
        # Read thickness for max thickness
        if result.thickness_path and Path(result.thickness_path).exists():
            with rasterio.open(result.thickness_path) as src:
                data = src.read(1)
                valid = np.isfinite(data) & (data > 0)
                if np.any(valid):
                    result.max_thickness = float(np.nanmax(data))
        
        # Read pressure for max pressure
        if result.pressure_path and Path(result.pressure_path).exists():
            with rasterio.open(result.pressure_path) as src:
                data = src.read(1)
                valid = np.isfinite(data) & (data > 0)
                if np.any(valid):
                    result.max_pressure = float(np.nanmax(data))
    
    except Exception as e:
        print(f"Warning: Error computing stats: {e}")
    
    return result


def run_parameter_sweep(
    sweep_config: SweepConfig,
    base_sim_config: SimulationConfig,
    progress_callback: Optional[Callable] = None,
    created_by: str = None
) -> List[SimulationResult]:
    """
    Run a parameter sweep across mu-xi combinations in parallel.

    Args:
        sweep_config: Sweep configuration with parameter ranges
        base_sim_config: Base simulation configuration
        progress_callback: Callback(current, total, message, sub_progress)
        created_by: Username who initiated the sweep

    Returns:
        List of SimulationResult objects
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    mu_values = sweep_config.mu_values
    xi_values = sweep_config.xi_values
    total = sweep_config.total_simulations

    # Create sweep output directory
    sweep_config.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of parallel workers
    max_workers = min(sweep_config.max_workers or multiprocessing.cpu_count(), total)

    # Build list of simulation tasks
    tasks = []
    for mu in mu_values:
        for xi in xi_values:
            sim_name = f"mu_{mu:.3f}_xi_{xi:.0f}"
            sim_config = SimulationConfig(
                mu=mu,
                xi=xi,
                rho=base_sim_config.rho,
                mesh_cell_size=base_sim_config.mesh_cell_size,
                sph_option=base_sim_config.sph_option,
                mass_per_part=base_sim_config.mass_per_part,
                delta_th=base_sim_config.delta_th,
                split_option=base_sim_config.split_option,
            )
            tasks.append((sim_name, sim_config, mu, xi))

    results = []
    completed_count = 0

    # Track input files for reproducibility
    input_files = {
        'dem': str(sweep_config.dem_path) if sweep_config.dem_path else None,
        'thickness': str(sweep_config.thickness_path) if sweep_config.thickness_path else None,
    }

    def run_one_simulation(task):
        """Run a single simulation task."""
        sim_name, sim_config, mu, xi = task
        try:
            # Prepare directory
            sim_dir, _ = prepare_simulation_directory(
                output_dir=sweep_config.output_dir,
                dem_path=sweep_config.dem_path,
                thickness_path=sweep_config.thickness_path,
                release_gdf=sweep_config.release_gdf,
                config=sim_config,
                sim_name=sim_name
            )

            # Run simulation with reproducibility tracking
            result = run_single_simulation(
                sim_dir=sim_dir,
                timeout=sweep_config.timeout_per_sim,
                created_by=created_by,
                input_files=input_files
            )
            return result

        except Exception as e:
            return SimulationResult(
                sim_id=sim_name,
                mu=mu,
                xi=xi,
                error=f"{type(e).__name__}: {str(e)}"
            )

    if progress_callback:
        progress_callback(0, total, f"Starting {total} simulations with {max_workers} workers", 0.0)

    # Run simulations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_one_simulation, task): task for task in tasks}

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            sim_name = task[0]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(SimulationResult(
                    sim_id=sim_name,
                    mu=task[2],
                    xi=task[3],
                    error=f"{type(e).__name__}: {str(e)}"
                ))

            completed_count += 1
            if progress_callback:
                progress_callback(
                    completed_count, total,
                    f"Completed {completed_count}/{total} ({sim_name})",
                    0.0
                )
    
    if progress_callback:
        progress_callback(total, total, "Sweep complete", 1.0)
    
    return results


def save_sweep_summary(results: List[SimulationResult], output_path: Path):
    """Save sweep results to CSV."""
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
