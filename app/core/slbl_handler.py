"""
SLBL job handler for the job queue system.
"""

from pathlib import Path
from typing import Callable
import json

from .job_queue import Job
from .slbl_engine import SLBLConfig, run_slbl_batch, save_batch_summary
from .project_manager import Project


def handle_slbl_batch_job(job: Job, progress_callback: Callable) -> dict:
    """
    Handler for SLBL batch processing jobs.
    
    Expected job.params:
        - scenario_paths: List of shapefile paths
        - e_ratios: List of e-ratio values
        - max_depths: List of max depth values (None for no limit)
        - mode: "failure" or "inverse"
        - neighbours: 4 or 8
        - write_ascii: bool
    
    Returns:
        Dictionary with results summary
    """
    # Load project
    project = Project.load(job.project_name)
    
    # Build config
    params = job.params
    config = SLBLConfig(
        mode=params.get('mode', 'failure'),
        e_ratios=params.get('e_ratios', [0.05, 0.10, 0.15, 0.20, 0.25]),
        max_depths=params.get('max_depths', [None]),
        neighbours=params.get('neighbours', 4),
        use_z_floor=params.get('use_z_floor', True),
        buffer_pixels=params.get('buffer_pixels', 4),
        write_geotiff=True,
        write_ascii=params.get('write_ascii', False),
        stop_eps=params.get('stop_eps', 1e-7),
        max_iters=params.get('max_iters', 3000),
    )
    
    # Get paths
    scenario_paths = params.get('scenario_paths', [])
    dem_path = params.get('dem_path', project.config.dem_path)
    output_dir = str(project.slbl_dir)
    
    # Cross-section parameters - add to config
    config.write_xsections = params.get('write_xsections', False)
    config.xsect_lines_source = params.get('xsect_lines_source', [])
    config.xsect_step_m = params.get('xsect_step_m', 1.0)
    config.xsect_clip_to_poly = params.get('xsect_clip_to_poly', False)

    # Initialize results list
    results = []
    successful = []
    failed = []

    # Run batch processing
    try:
        results = run_slbl_batch(
            scenario_paths=scenario_paths,
            dem_path=dem_path,
            output_dir=output_dir,
            config=config,
            target_epsg=project.config.target_epsg,
            progress_callback=progress_callback,
            job_id=job.id,
            job_created_by=job.created_by,
            job_created=str(job.created_at) if job.created_at else None,
        )
        
        # Ensure results is a list
        if results is None:
            results = []
        
        # Compute summary stats
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
    except Exception as e:
        # Log the error but don't crash - return what we have
        import traceback
        print(f"[SLBL Handler] Error during batch processing: {e}")
        traceback.print_exc()
        # Re-raise so the job queue marks it as failed with the real error
        raise
    
    # Save summary (only if we have results)
    summary_path = project.slbl_dir / "slbl_summary.csv"
    results_json = project.slbl_dir / "slbl_results.json"
    
    if results:
        save_batch_summary(results, str(summary_path))
        
        with open(results_json, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    
    return {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'summary_csv': str(summary_path),
        'results_json': str(results_json),
        'output_dir': output_dir,
        'volumes_m3': [r.volume_m3 for r in successful],
        'errors': [{'scenario': r.scenario_name, 'error': r.error} for r in failed]
    }


def register_slbl_handlers(job_queue):
    """Register SLBL handlers with the job queue."""
    job_queue.register_handler('slbl_batch', handle_slbl_batch_job)