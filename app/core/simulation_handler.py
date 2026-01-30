"""
Simulation job handlers for the job queue system.
"""

from pathlib import Path
from typing import Callable
import json
import geopandas as gpd

from .job_queue import Job
from .avaframe_runner import (
    SweepConfig, SimulationConfig, 
    run_parameter_sweep, save_sweep_summary,
    prepare_simulation_directory, run_single_simulation
)
from .project_manager import Project


def handle_simulation_sweep_job(job: Job, progress_callback: Callable) -> dict:
    """
    Handler for simulation parameter sweep jobs.
    
    Expected job.params:
        - thickness_path: Path to SLBL thickness raster
        - release_polygon_path: Path to release polygon shapefile
        - mu_min, mu_max, mu_steps: Friction parameter range
        - xi_min, xi_max, xi_steps: Turbulence parameter range
        - rho: Material density
        - sweep_name: Optional name for this sweep
    """
    project = Project.load(job.project_name)
    params = job.params
    
    # Load release polygon
    release_path = params.get('release_polygon_path')
    if release_path:
        release_gdf = gpd.read_file(release_path)
        if release_gdf.crs.to_epsg() != project.config.target_epsg:
            release_gdf = release_gdf.to_crs(epsg=project.config.target_epsg)
    else:
        raise ValueError("Release polygon path required")
    
    # Create sweep name
    sweep_name = params.get('sweep_name') or f"sweep_{job.id}"
    output_dir = project.simulations_dir / sweep_name
    
    # Determine number of workers for parallel sweep (use all available cores)
    import multiprocessing
    max_workers = params.get('max_workers', multiprocessing.cpu_count())

    # Build configs
    sweep_config = SweepConfig(
        mu_min=params.get('mu_min', 0.025),
        mu_max=params.get('mu_max', 0.1),
        mu_steps=params.get('mu_steps', 4),
        xi_min=params.get('xi_min', 500),
        xi_max=params.get('xi_max', 1500),
        xi_steps=params.get('xi_steps', 4),
        dem_path=Path(params.get('dem_path', project.config.dem_path)),
        thickness_path=Path(params['thickness_path']),
        release_gdf=release_gdf,
        output_dir=output_dir,
        max_workers=max_workers,
        timeout_per_sim=params.get('timeout', 7200)
    )
    
    base_config = SimulationConfig(
        rho=params.get('rho', 2500.0),
        sph_option=params.get('sph_option', 3),
        mass_per_part=params.get('mass_per_part', 500000.0),
        delta_th=params.get('delta_th', 4.0),
    )
    
    # Run sweep with reproducibility tracking
    results = run_parameter_sweep(
        sweep_config=sweep_config,
        base_sim_config=base_config,
        progress_callback=progress_callback,
        created_by=job.created_by
    )
    
    # Save summary
    summary_path = output_dir / "sweep_summary.csv"
    save_sweep_summary(results, summary_path)
    
    # Also save as JSON
    results_json = output_dir / "sweep_results.json"
    with open(results_json, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    
    # Compute summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    return {
        'sweep_name': sweep_name,
        'output_dir': str(output_dir),
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'summary_csv': str(summary_path),
        'max_velocities': [r.max_velocity for r in successful],
        'affected_areas': [r.affected_area_ha for r in successful],
    }


def handle_single_simulation_job(job: Job, progress_callback: Callable) -> dict:
    """
    Handler for single simulation jobs.
    
    Expected job.params:
        - thickness_path: Path to thickness raster
        - release_polygon_path: Path to release polygon
        - mu: Friction coefficient
        - xi: Turbulence coefficient
        - sim_name: Optional simulation name
    """
    project = Project.load(job.project_name)
    params = job.params
    
    # Load release polygon
    release_gdf = gpd.read_file(params['release_polygon_path'])
    if release_gdf.crs.to_epsg() != project.config.target_epsg:
        release_gdf = release_gdf.to_crs(epsg=project.config.target_epsg)
    
    # Create simulation config
    sim_config = SimulationConfig(
        mu=params.get('mu', 0.035),
        xi=params.get('xi', 700.0),
        rho=params.get('rho', 2500.0),
        mass_per_part=params.get('mass_per_part', 500000.0),
        delta_th=params.get('delta_th', 4.0),
    )
    
    sim_name = params.get('sim_name') or f"sim_{job.id}"
    
    progress_callback(0, 3, "Preparing simulation directory", 0)
    
    # Prepare directory
    sim_dir, _ = prepare_simulation_directory(
        output_dir=project.simulations_dir,
        dem_path=Path(params.get('dem_path', project.config.dem_path)),
        thickness_path=Path(params['thickness_path']),
        release_gdf=release_gdf,
        config=sim_config,
        sim_name=sim_name
    )
    
    progress_callback(1, 3, "Running simulation", 0.3)

    # Track input files for reproducibility
    input_files = {
        'dem': str(params.get('dem_path', project.config.dem_path)),
        'thickness': params['thickness_path'],
        'release_polygon': params['release_polygon_path']
    }

    # Run simulation with reproducibility tracking
    result = run_single_simulation(
        sim_dir=sim_dir,
        timeout=params.get('timeout', 1800),
        created_by=job.created_by,
        input_files=input_files
    )
    
    progress_callback(2, 3, "Processing results", 0.9)
    
    # Save result info
    result_json = sim_dir / "simulation_result.json"
    with open(result_json, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    progress_callback(3, 3, "Complete", 1.0)
    
    return result.to_dict()


def register_simulation_handlers(job_queue):
    """Register simulation handlers with the job queue."""
    job_queue.register_handler('simulation_sweep', handle_simulation_sweep_job)
    job_queue.register_handler('single_simulation', handle_single_simulation_job)
