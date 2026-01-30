"""
Probability Handler for Job Queue Integration

This module provides the main entry point for executing probability ensemble
jobs through the existing job queue system. It integrates with the existing
avaframe_runner module for simulation execution.

Integration Points:
------------------
- job_queue.py: Submits/receives jobs of type probability_ensemble
- avaframe_runner.py: Executes individual simulations
- project structure: Follows existing directory conventions
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime
import geopandas as gpd

from .ensemble_manager import (
    EnsembleManager,
    EnsembleConfig,
    EnsembleStatus,
    SLBLSurface
)
from .weighting_schemes import SimulationMetrics, WeightingMethod
from .probability_aggregator import ProbabilityAggregator, ProbabilityConfig


def run_probability_ensemble(
    project,
    ensemble_config: EnsembleConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    job_id: str = None
) -> Dict[str, Any]:
    """
    Execute complete probability ensemble workflow.

    This is the main entry point called by the job queue worker.
    Orchestrates the entire process from parameter sampling through
    probability map generation.

    Workflow:
    ---------
    1. Initialize ensemble manager and create output directories
    2. Generate parameter samples (mu, xi combinations)
    3. Run AvaFrame simulations for each sample
    4. Extract metrics from each simulation
    5. Compute weights using empirical relationships
    6. Aggregate results into probability maps
    7. Generate summary report

    Args:
        project: Project instance with directory structure
        ensemble_config: Configuration defining the ensemble
        progress_callback: Optional function(message, progress_fraction)
                          for reporting progress to UI

    Returns:
        Dictionary containing:
        - ensemble_id: Unique identifier
        - output_dir: Path to output directory
        - probability_maps: Dict of map name to file path
        - summary: Summary statistics and metadata

    Raises:
        RuntimeError: If all simulations fail
        ValueError: If configuration is invalid
    """
    def report_progress(msg: str, pct: float):
        """Report progress if callback provided."""
        if progress_callback:
            progress_callback(msg, pct)
        print(f"[{pct*100:.1f}%] {msg}")

    # Initialize
    report_progress("Initializing probability ensemble...", 0.01)
    manager = EnsembleManager(project, ensemble_config)
    manager.setup_directories()
    manager.save_config()

    # Phase 1: Generate parameter samples
    report_progress("Generating parameter samples...", 0.05)
    samples = manager.prepare_ensemble()
    manager.save_samples()

    total_sims = len(samples)
    report_progress(f"Generated {total_sims} parameter combinations", 0.08)

    # Phase 2: Run simulations in parallel
    manager.status = EnsembleStatus.RUNNING
    report_progress("Starting simulations...", 0.10)

    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Determine number of workers
    max_workers = min(multiprocessing.cpu_count(), total_sims)
    report_progress(f"Running {total_sims} simulations with {max_workers} workers...", 0.10)

    # Get simulation performance settings from config
    mass_per_part = ensemble_config.mass_per_part
    delta_th = ensemble_config.delta_th

    # Build simulation tasks
    def run_single_sim(sample):
        """Run a single simulation task."""
        # Check for cancellation before starting
        if job_id:
            from core.job_queue import get_job_queue, JobStatus
            current_job = get_job_queue().get_job(job_id)
            if current_job and current_job.status == JobStatus.CANCELLED:
                return {"sample_id": sample.sample_id, "success": False, "error": "Cancelled"}

        slbl = manager.get_slbl_for_sample(sample)
        sim_output_dir = manager.ensemble_dir / f"sim_{sample.sample_id:04d}"
        sim_output_dir.mkdir(exist_ok=True)

        try:
            result = run_avaframe_simulation(
                project=project,
                thickness_raster_path=slbl.thickness_raster_path,
                mu=sample.mu,
                xi=sample.xi,
                output_dir=sim_output_dir,
                job_id=job_id,
                mass_per_part=mass_per_part,
                delta_th=delta_th
            )

            metrics = extract_metrics_from_result(
                result=result,
                sample=sample,
                slbl=slbl
            )

            return {
                "sample_id": sample.sample_id,
                "success": True,
                "output_dir": sim_output_dir,
                "metrics": metrics,
                "raster_paths": {
                    "depth": result.get("depth_raster_path"),
                    "velocity": result.get("velocity_raster_path"),
                    "pressure": result.get("pressure_raster_path"),
                }
            }
        except Exception as e:
            return {
                "sample_id": sample.sample_id,
                "success": False,
                "error": str(e)
            }

    # Run in parallel with cancellation support
    completed_count = 0
    cancelled = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(run_single_sim, s): s for s in samples}

        for future in as_completed(future_to_sample):
            # Check for cancellation before processing each result
            if job_id:
                from core.job_queue import get_job_queue, JobStatus
                current_job = get_job_queue().get_job(job_id)
                if current_job and current_job.status == JobStatus.CANCELLED:
                    # Cancel all pending futures and shutdown
                    cancelled = True
                    for f in future_to_sample:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise InterruptedError("Job was cancelled")

            sample = future_to_sample[future]
            completed_count += 1

            sim_progress = 0.10 + 0.70 * (completed_count / total_sims)
            report_progress(
                f"Completed {completed_count}/{total_sims}: mu={sample.mu:.3f}, xi={sample.xi:.0f}",
                sim_progress
            )

            try:
                result = future.result()
                if result["success"]:
                    manager.add_simulation_result(
                        sample_id=result["sample_id"],
                        success=True,
                        output_dir=result["output_dir"],
                        metrics=result["metrics"],
                        raster_paths=result["raster_paths"]
                    )
                else:
                    manager.add_simulation_result(
                        sample_id=result["sample_id"],
                        success=False,
                        error=result.get("error", "Unknown error")
                    )
                    print(f"  Warning: Simulation {result['sample_id']} failed: {result.get('error')}")
            except Exception as e:
                manager.add_simulation_result(
                    sample_id=sample.sample_id,
                    success=False,
                    error=str(e)
                )
                print(f"  Warning: Simulation {sample.sample_id} failed: {e}")

    # Check we have enough successful simulations
    successful = [r for r in manager.simulation_results if r["success"]]
    if len(successful) < 5:
        raise RuntimeError(
            f"Only {len(successful)} simulations succeeded. "
            f"Need at least 5 for meaningful probability maps."
        )

    report_progress(
        f"Completed {len(successful)}/{total_sims} simulations successfully",
        0.80
    )

    # Phase 3: Compute weights
    report_progress("Computing simulation weights...", 0.82)
    manager.status = EnsembleStatus.WEIGHTING
    manager.compute_weights()

    # Phase 4: Aggregate probability maps
    report_progress("Aggregating probability maps...", 0.85)
    manager.status = EnsembleStatus.AGGREGATING

    probability_maps = aggregate_ensemble_results(
        manager=manager,
        successful_results=successful,
        output_dir=manager.results_dir,
        progress_callback=lambda msg, pct: report_progress(msg, 0.85 + 0.10 * pct)
    )

    manager.probability_maps = probability_maps

    # Phase 5: Generate report
    report_progress("Generating summary report...", 0.97)
    manager.status = EnsembleStatus.COMPLETED
    report_path = manager.save_report()

    report_progress("Probability ensemble complete!", 1.0)

    return {
        "ensemble_id": manager.ensemble_id,
        "output_dir": str(manager.output_dir),
        "probability_maps": {k: str(v) for k, v in probability_maps.items()},
        "summary_report": str(report_path),
        "statistics": {
            "total_simulations": total_sims,
            "successful": len(successful),
            "failed": total_sims - len(successful)
        }
    }


def run_avaframe_simulation(
    project,
    thickness_raster_path: Path,
    mu: float,
    xi: float,
    output_dir: Path,
    job_id: str = None,
    mass_per_part: float = 500000.0,
    delta_th: float = 4.0
) -> Dict[str, Any]:
    """
    Run single AvaFrame simulation with specified parameters.

    This function interfaces with the existing avaframe_runner module
    to execute a com1DFA simulation.

    Args:
        project: Project instance
        thickness_raster_path: Path to SLBL thickness raster (release area)
        mu: Friction coefficient for Voellmy model
        xi: Turbulence coefficient for Voellmy model (m/s^2)
        output_dir: Directory for simulation outputs
        job_id: Optional job ID for process tracking
        mass_per_part: Mass per particle [kg] (default: 500000 for fast mode)
        delta_th: Release thickness per particle [m] (default: 4.0 for fast mode)

    Returns:
        Dictionary with simulation results including:
        - depth_raster_path: Path to max flow depth raster
        - velocity_raster_path: Path to max velocity raster
        - runout_distance: Maximum runout in meters
        - affected_area: Total affected area in m^2
        - max_velocity: Maximum velocity achieved
        - max_depth: Maximum flow depth
        - success: Boolean indicating completion
    """
    from core.avaframe_runner import (
        SimulationConfig,
        prepare_simulation_directory,
        run_single_simulation
    )

    # Create simulation configuration
    sim_config = SimulationConfig(
        mu=mu,
        xi=xi,
        rho=2500.0,
        sph_option=3,
        mass_per_part=mass_per_part,
        delta_th=delta_th,
        split_option=1,
    )

    # Get release polygon from thickness raster extent
    with rasterio.open(thickness_raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    # Create release polygon from raster bounds
    from shapely.geometry import box
    release_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    release_gdf = gpd.GeoDataFrame(
        {'geometry': [release_geom]},
        crs=crs
    )

    # Prepare simulation directory
    sim_name = f"mu_{mu:.3f}_xi_{xi:.0f}"
    sim_dir, _ = prepare_simulation_directory(
        output_dir=output_dir,
        dem_path=Path(project.config.dem_path),
        thickness_path=thickness_raster_path,
        release_gdf=release_gdf,
        config=sim_config,
        sim_name=sim_name
    )

    # Run simulation
    result = run_single_simulation(
        sim_dir=sim_dir,
        timeout=7200,  # 2 hours - large volumes may need extended runtime
        created_by="probability_ensemble",
        job_id=job_id
    )

    if not result.success:
        raise RuntimeError(f"AvaFrame simulation failed: {result.error}")

    # Calculate runout distance from velocity raster
    runout_distance = 0.0
    affected_area = 0.0

    if result.velocity_path and Path(result.velocity_path).exists():
        with rasterio.open(result.velocity_path) as src:
            data = src.read(1)
            transform = src.transform
            cellsize = abs(transform[0])

            # Affected area
            impact_mask = (data > 0.1) & np.isfinite(data)
            affected_area = float(np.sum(impact_mask) * cellsize * cellsize)

            # Runout distance (furthest point from release area)
            if np.any(impact_mask):
                rows, cols = np.where(impact_mask)
                if len(rows) > 0:
                    # Get geographic coordinates
                    xs = transform[2] + cols * cellsize
                    ys = transform[5] - rows * cellsize

                    # Simple runout as distance from centroid to furthest point
                    cx, cy = np.mean(xs), np.mean(ys)
                    distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)
                    runout_distance = float(np.max(distances))

    return {
        "success": True,
        "depth_raster_path": Path(result.thickness_path) if result.thickness_path else None,
        "velocity_raster_path": Path(result.velocity_path) if result.velocity_path else None,
        "pressure_raster_path": Path(result.pressure_path) if result.pressure_path else None,
        "runout_distance": runout_distance,
        "affected_area": affected_area,
        "max_velocity": result.max_velocity,
        "max_depth": result.max_thickness
    }


def extract_metrics_from_result(
    result: Dict[str, Any],
    sample,  # ParameterSample
    slbl     # SLBLSurface
) -> SimulationMetrics:
    """
    Extract metrics from simulation result for weighting.

    Converts raw simulation outputs into SimulationMetrics object
    that can be used for empirical weight calculation.

    Args:
        result: Dictionary from run_avaframe_simulation
        sample: ParameterSample with mu, xi, volume info
        slbl: SLBLSurface with height drop info

    Returns:
        SimulationMetrics suitable for weighting calculation
    """
    return SimulationMetrics(
        sample_id=sample.sample_id,
        slbl_id=sample.slbl_id,
        mu=sample.mu,
        xi=sample.xi,
        volume_m3=slbl.volume_m3,
        height_drop_m=slbl.height_drop_m,
        runout_m=result.get("runout_distance", 0),
        affected_area_m2=result.get("affected_area", 0),
        max_velocity_ms=result.get("max_velocity", 0),
        max_depth_m=result.get("max_depth", 0)
    )


def aggregate_ensemble_results(
    manager: EnsembleManager,
    successful_results: List[Dict],
    output_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Path]:
    """
    Aggregate simulation results into probability maps.

    Loads rasters from all successful simulations, stacks them,
    and computes probability/percentile maps using configured weights.

    Args:
        manager: EnsembleManager with weights and config
        successful_results: List of successful simulation result dicts
        output_dir: Directory for output probability maps
        progress_callback: Optional progress reporter

    Returns:
        Dictionary mapping output name to file path
    """
    def report(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)

    report("Loading simulation rasters...", 0.0)

    # Get reference raster for grid dimensions
    first_result = successful_results[0]
    depth_path = first_result["raster_paths"].get("depth")

    if not depth_path or not Path(depth_path).exists():
        # Try velocity instead
        depth_path = first_result["raster_paths"].get("velocity")

    if not depth_path or not Path(depth_path).exists():
        raise RuntimeError("No valid output rasters found")

    with rasterio.open(depth_path) as src:
        profile = src.profile.copy()
        shape = src.shape
        transform = src.transform
        crs = src.crs

    n_sims = len(successful_results)

    # Allocate stacks
    depth_stack = np.zeros((n_sims, shape[0], shape[1]), dtype=np.float32)
    velocity_stack = np.zeros_like(depth_stack)

    # Load all rasters
    for i, result in enumerate(successful_results):
        report(f"Loading raster {i+1}/{n_sims}", 0.1 + 0.3 * (i / n_sims))

        # Depth
        depth_path = result["raster_paths"].get("depth")
        if depth_path and Path(depth_path).exists():
            with rasterio.open(depth_path) as src:
                data = src.read(1)
                # Handle shape mismatch by resampling if needed
                if data.shape == shape:
                    depth_stack[i] = data
                else:
                    # Basic resize - in production use proper resampling
                    depth_stack[i] = np.zeros(shape, dtype=np.float32)

        # Velocity
        velocity_path = result["raster_paths"].get("velocity")
        if velocity_path and Path(velocity_path).exists():
            with rasterio.open(velocity_path) as src:
                data = src.read(1)
                if data.shape == shape:
                    velocity_stack[i] = data

    # Create aggregator
    aggregator = ProbabilityAggregator(ProbabilityConfig(
        percentiles=manager.config.percentiles
    ))

    report("Computing probability maps...", 0.5)

    # Get weights array
    weights = manager.get_weights_array()

    # Compute all outputs
    outputs = aggregator.aggregate_all_outputs(
        depth_stack=depth_stack,
        velocity_stack=velocity_stack,
        weights=weights
    )

    # Save outputs as GeoTIFFs
    report("Saving probability maps...", 0.8)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    for name, data in outputs.items():
        output_path = output_dir / f"{name}.tif"

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(np.float32), 1)

            # Add metadata
            dst.update_tags(
                ensemble_id=manager.ensemble_id,
                weighting_method=manager.config.weighting_method.value,
                n_simulations=str(n_sims),
                created=datetime.now().isoformat()
            )

        output_paths[name] = output_path

    report("Probability maps complete", 1.0)

    return output_paths


def load_ensemble_results(ensemble_dir: Path) -> Dict[str, Any]:
    """
    Load results from a completed ensemble.

    Useful for viewing/analyzing results after the fact.

    Args:
        ensemble_dir: Path to ensemble output directory

    Returns:
        Dictionary with configuration, weights, and output paths
    """
    ensemble_dir = Path(ensemble_dir)

    # Load config
    config_files = list((ensemble_dir / "config").glob("config_*.json"))
    if config_files:
        with open(config_files[0]) as f:
            config = json.load(f)
    else:
        config = {}

    # Load weights
    weight_files = list((ensemble_dir / "weights").glob("weights_*.json"))
    if weight_files:
        with open(weight_files[0]) as f:
            weights = json.load(f)
    else:
        weights = {}

    # Load summary report
    report_path = ensemble_dir / "results" / "summary_report.json"
    if report_path.exists():
        with open(report_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Find output maps
    output_maps = {}
    results_dir = ensemble_dir / "results"
    if results_dir.exists():
        for tif_path in results_dir.glob("*.tif"):
            output_maps[tif_path.stem] = tif_path

    return {
        "ensemble_dir": str(ensemble_dir),
        "config": config,
        "weights": weights,
        "summary": summary,
        "output_maps": output_maps
    }


def compare_weighting_methods(
    project,
    ensemble_config: EnsembleConfig,
    methods: List[WeightingMethod] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Run the same ensemble with different weighting methods for comparison.

    This allows direct comparison of how weighting affects probability
    maps, which can be useful for sensitivity analysis.

    Args:
        project: Project instance
        ensemble_config: Base configuration
        methods: List of weighting methods to compare
                 (default: all available methods)

    Returns:
        Dictionary mapping method name to output paths
    """
    if methods is None:
        methods = list(WeightingMethod)

    results = {}

    for method in methods:
        # Create modified config
        config = EnsembleConfig(
            slbl_surfaces=ensemble_config.slbl_surfaces,
            sims_per_slbl=ensemble_config.sims_per_slbl,
            sampling_method=ensemble_config.sampling_method,
            mu_min=ensemble_config.mu_min,
            mu_max=ensemble_config.mu_max,
            xi_min=ensemble_config.xi_min,
            xi_max=ensemble_config.xi_max,
            weighting_method=method,
            confinement_type=ensemble_config.confinement_type,
            percentiles=ensemble_config.percentiles,
            notes=f"Weighting comparison: {method.value}"
        )

        # Run ensemble
        result = run_probability_ensemble(project, config)
        results[method.value] = result["probability_maps"]

    return results


def register_probability_handlers(job_queue):
    """
    Register probability ensemble handlers with the job queue.

    This function is called by job_queue.py to register the
    probability_ensemble job type handler.

    Args:
        job_queue: JobQueue instance to register handlers with
    """
    from core.project_manager import Project

    def probability_ensemble_handler(job, progress_callback):
        """Handler for probability_ensemble jobs."""
        # Load project
        project = Project.load(job.project_name)

        # Parse config from job params
        config_dict = job.params.get("config", {})
        config = EnsembleConfig.from_dict(config_dict)

        # Wrap progress callback
        def wrapped_progress(msg: str, pct: float):
            progress_callback(
                int(pct * 100),
                100,
                msg,
                pct % 1
            )

        # Run ensemble
        result = run_probability_ensemble(
            project=project,
            ensemble_config=config,
            progress_callback=wrapped_progress,
            job_id=job.id
        )

        return result

    job_queue.register_handler("probability_ensemble", probability_ensemble_handler)
