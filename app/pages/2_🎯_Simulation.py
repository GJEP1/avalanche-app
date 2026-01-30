"""
Simulation Page
===============
Run AVAFRAME rock avalanche simulations using SLBL outputs or manual uploads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import sys
import json
import tempfile
import zipfile
import shutil
import rasterio

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import project_selector_sidebar, get_current_project
from core.job_queue import get_job_queue, JobStatus
from core.simulation_handler import register_simulation_handlers

# Page config
st.set_page_config(
    page_title="Rock Avalanche Simulation",
    page_icon="🎯",
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

# Initialize job queue
job_queue = get_job_queue()
register_simulation_handlers(job_queue)

# Page title
st.title("🎯 Rock Avalanche Simulation")
st.markdown(f"**Project:** {project.name}")

# Tabs
tab_single, tab_sweep, tab_results, tab_jobs = st.tabs([
    "🎯 Single Simulation", "🔬 Parameter Sweep", "📊 Results", "⚙️ Jobs"
])

# =============================================================================
# Helper functions
# =============================================================================

def get_available_thickness_rasters():
    """Get list of available thickness rasters (SLBL outputs + uploads), sorted alphabetically."""
    rasters = []

    # SLBL outputs
    if project.slbl_dir.exists():
        for f in project.slbl_dir.glob("*_thickness.tif"):
            rasters.append({
                'name': f.stem,
                'path': str(f),
                'source': 'SLBL',
                'label': f.stem.replace('_thickness', '')
            })

    # Manual uploads
    uploads_dir = project.inputs_dir / 'thickness_rasters'
    if uploads_dir.exists():
        for f in uploads_dir.glob("*.tif"):
            rasters.append({
                'name': f.stem,
                'path': str(f),
                'source': 'Upload',
                'label': f.stem
            })

    # Sort alphabetically by name
    return sorted(rasters, key=lambda x: x['name'].lower())


def get_available_release_polygons():
    """Get list of available release polygons, sorted alphabetically."""
    polygons = []

    # Scenarios directory
    if project.scenarios_dir.exists():
        for f in project.scenarios_dir.glob("*.shp"):
            polygons.append({
                'name': f.stem,
                'path': str(f),
                'source': 'Scenario'
            })

    # Uploads
    uploads_dir = project.inputs_dir / 'release_polygons'
    if uploads_dir.exists():
        for f in uploads_dir.glob("*.shp"):
            polygons.append({
                'name': f.stem,
                'path': str(f),
                'source': 'Upload'
            })

    # Sort alphabetically by name
    return sorted(polygons, key=lambda x: x['name'].lower())


# =============================================================================
# TAB 1: Single Simulation
# =============================================================================
with tab_single:
    st.header("Single Simulation")
    st.markdown("Run a single AVAFRAME simulation with specific parameters.")
    
    col_inputs, col_params = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("Input Data")
        
        # Thickness raster selection
        st.markdown("**Thickness Raster**")
        thickness_rasters = get_available_thickness_rasters()
        
        if thickness_rasters:
            thickness_options = {f"{r['label']} ({r['source']})": r['path'] 
                               for r in thickness_rasters}
            selected_thickness_label = st.selectbox(
                "Select thickness raster:",
                options=list(thickness_options.keys()),
                key="single_thickness"
            )
            selected_thickness_path = thickness_options[selected_thickness_label]
            
            # Show raster info
            try:
                with rasterio.open(selected_thickness_path) as src:
                    st.caption(f"Size: {src.width}×{src.height}, CRS: {src.crs}")
            except:
                pass
        else:
            st.warning("No thickness rasters available. Generate SLBL outputs or upload manually.")
            selected_thickness_path = None
        
        # Upload option
        with st.expander("📤 Upload thickness raster"):
            uploaded_thickness = st.file_uploader(
                "Upload GeoTIFF",
                type=['tif', 'tiff'],
                key="upload_thickness_single"
            )
            if uploaded_thickness:
                uploads_dir = project.inputs_dir / 'thickness_rasters'
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = uploads_dir / uploaded_thickness.name
                with open(dest_path, 'wb') as f:
                    f.write(uploaded_thickness.getbuffer())
                
                st.success(f"Uploaded: {uploaded_thickness.name}")
                st.rerun()
        
        st.divider()
        
        # Release polygon selection
        st.markdown("**Release Polygon**")
        release_polygons = get_available_release_polygons()
        
        if release_polygons:
            polygon_options = {f"{p['name']} ({p['source']})": p['path']
                            for p in release_polygons}
            selected_polygon_label = st.selectbox(
                "Select release polygon:",
                options=list(polygon_options.keys()),
                key="single_polygon"
            )
            selected_polygon_path = polygon_options[selected_polygon_label]
        else:
            st.warning("No release polygons available. Upload a shapefile.")
            selected_polygon_path = None
        
        # Upload option
        with st.expander("📤 Upload release polygon"):
            uploaded_polygon = st.file_uploader(
                "Upload shapefile (ZIP)",
                type=['zip'],
                key="upload_polygon_single"
            )
            if uploaded_polygon:
                uploads_dir = project.inputs_dir / 'release_polygons'
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / uploaded_polygon.name
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_polygon.getbuffer())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(tmpdir)
                    
                    for shp in Path(tmpdir).glob("**/*.shp"):
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            src = shp.with_suffix(ext)
                            if src.exists():
                                shutil.copy(src, uploads_dir / src.name)
                
                st.success("Uploaded shapefile")
                st.rerun()
    
    with col_params:
        st.subheader("Simulation Parameters")
        
        # Voellmy parameters
        st.markdown("**Voellmy Friction Model**")
        
        col_mu, col_xi = st.columns(2)
        with col_mu:
            mu = st.number_input(
                "μ (friction coefficient)",
                min_value=0.01,
                max_value=0.5,
                value=0.035,
                step=0.005,
                format="%.3f",
                help="Default: 0.035. Typical range: 0.025 - 0.15"
            )

        with col_xi:
            xi = st.number_input(
                "ξ (turbulence) [m/s²]",
                min_value=100.0,
                max_value=3000.0,
                value=700.0,
                step=100.0,
                help="Default: 700. Typical range: 250 - 2000"
            )

        st.divider()

        # Material properties
        st.markdown("**Material Properties**")
        rho = st.number_input(
            "Density [kg/m³]",
            min_value=1000.0,
            max_value=4000.0,
            value=2500.0,
            step=100.0,
            help="Default: 2500 (rock)"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            # Performance preset
            preset = st.radio(
                "Performance preset",
                ["Standard (accurate)", "Fast (coarser)"],
                horizontal=True,
                help="Standard: more particles, slower but more accurate. Fast: fewer particles, quicker runs."
            )

            if preset == "Standard (accurate)":
                default_mass = 280000.0
                default_delta = 2.0
            else:
                default_mass = 500000.0
                default_delta = 4.0

            sph_option = st.selectbox("SPH Option", [1, 2, 3], index=2,
                                       help="Default: 3 (local coord system with surface reprojection)")
            mass_per_part = st.number_input("Mass per particle [kg]",
                                           value=default_mass, step=10000.0,
                                           help="Default: 280,000 (Standard) / 500,000 (Fast)")
            delta_th = st.number_input("Release thickness per particle [m]",
                                      value=default_delta, step=0.5,
                                      help="Default: 2.0 (Standard) / 4.0 (Fast)")
            timeout = st.number_input("Timeout [seconds]",
                                     value=3600, min_value=300, max_value=14400,
                                     help="Default: 3600 (1 hour)")
    
    st.divider()
    
    # Submit button
    col_submit, col_status = st.columns([1, 2])
    
    with col_submit:
        can_submit = selected_thickness_path and selected_polygon_path
        
        if st.button("🚀 Run Simulation", type="primary", 
                    disabled=not can_submit, width="stretch"):
            # Generate simulation name
            thickness_name = Path(selected_thickness_path).stem.replace('_thickness', '')
            sim_name = f"{thickness_name}_mu{mu:.3f}_xi{xi:.0f}"
            
            job = job_queue.submit(
                job_type='single_simulation',
                project_name=project.name,
                created_by=username,
                params={
                    'thickness_path': selected_thickness_path,
                    'release_polygon_path': selected_polygon_path,
                    'mu': mu,
                    'xi': xi,
                    'rho': rho,
                    'sph_option': sph_option,
                    'mass_per_part': mass_per_part,
                    'delta_th': delta_th,
                    'timeout': timeout,
                    'sim_name': sim_name,
                    'dem_path': project.config.dem_path,
                }
            )
            
            st.success(f"Simulation submitted! Job ID: {job.id}")
    
    with col_status:
        if not can_submit:
            st.warning("Select both a thickness raster and release polygon to run simulation.")


# =============================================================================
# TAB 2: Parameter Sweep
# =============================================================================
with tab_sweep:
    st.header("Parameter Sweep")
    st.markdown("Run multiple simulations across a range of μ-ξ parameters.")
    
    col_inputs_sweep, col_params_sweep = st.columns([1, 1])
    
    with col_inputs_sweep:
        st.subheader("Input Data")
        
        # Thickness selection (can select multiple for batch)
        st.markdown("**Thickness Raster(s)**")
        thickness_rasters = get_available_thickness_rasters()
        
        if thickness_rasters:
            # For sweep, we'll use one thickness at a time (can extend later)
            thickness_options = {f"{r['label']} ({r['source']})": r['path'] 
                               for r in thickness_rasters}
            
            sweep_thickness_label = st.selectbox(
                "Select thickness raster:",
                options=list(thickness_options.keys()),
                key="sweep_thickness"
            )
            sweep_thickness_path = thickness_options[sweep_thickness_label]
        else:
            st.warning("No thickness rasters available.")
            sweep_thickness_path = None
        
        st.divider()
        
        # Release polygon
        st.markdown("**Release Polygon**")
        if release_polygons:
            sweep_polygon_label = st.selectbox(
                "Select release polygon:",
                options=list(polygon_options.keys()),
                key="sweep_polygon"
            )
            sweep_polygon_path = polygon_options[sweep_polygon_label]
        else:
            st.warning("No release polygons available.")
            sweep_polygon_path = None
    
    with col_params_sweep:
        st.subheader("Parameter Ranges")
        
        # Mu range
        st.markdown("**Friction coefficient (μ)** — single value default: 0.035")
        col_mu1, col_mu2, col_mu3 = st.columns(3)
        with col_mu1:
            mu_min = st.number_input("Min μ", value=0.025, step=0.005, format="%.3f")
        with col_mu2:
            mu_max = st.number_input("Max μ", value=0.400, step=0.005, format="%.3f")
        with col_mu3:
            mu_steps = st.number_input("Steps", value=16, min_value=2, max_value=32)

        mu_values = np.linspace(mu_min, mu_max, mu_steps)
        st.caption(f"Values: {', '.join([f'{v:.3f}' for v in mu_values])}")

        st.divider()

        # Xi range
        st.markdown("**Turbulence coefficient (ξ)** — single value default: 700")
        col_xi1, col_xi2, col_xi3 = st.columns(3)
        with col_xi1:
            xi_min = st.number_input("Min ξ", value=250.0, step=50.0)
        with col_xi2:
            xi_max = st.number_input("Max ξ", value=2000.0, step=100.0)
        with col_xi3:
            xi_steps = st.number_input("Steps", value=8, min_value=2, max_value=32, key="xi_steps")
        
        xi_values = np.linspace(xi_min, xi_max, xi_steps)
        st.caption(f"Values: {', '.join([f'{v:.0f}' for v in xi_values])}")
        
        st.divider()
        
        # Summary
        total_sims = mu_steps * xi_steps
        st.info(f"**Total simulations: {total_sims}**")
        
        est_time_min = total_sims * 5  # Rough estimate: 5 min per sim
        st.caption(f"Estimated time: ~{est_time_min} minutes")
        
        # Material properties
        with st.expander("Material & Advanced Settings"):
            # Performance preset
            sweep_preset = st.radio(
                "Performance preset",
                ["Standard (accurate)", "Fast (coarser)"],
                horizontal=True,
                key="sweep_preset",
                help="Standard: more particles, slower but more accurate. Fast: fewer particles, quicker runs."
            )

            # Determine default values based on preset
            if sweep_preset == "Standard (accurate)":
                sweep_default_mass = 280000.0
                sweep_default_delta = 2.0
            else:
                sweep_default_mass = 500000.0
                sweep_default_delta = 4.0

            # Track preset changes and reset values when preset changes
            if 'sweep_preset_prev' not in st.session_state:
                st.session_state.sweep_preset_prev = sweep_preset

            if st.session_state.sweep_preset_prev != sweep_preset:
                # Preset changed - update the values
                st.session_state.sweep_mass = sweep_default_mass
                st.session_state.sweep_delta = sweep_default_delta
                st.session_state.sweep_preset_prev = sweep_preset
                st.rerun()

            sweep_rho = st.number_input("Density [kg/m³]", value=2500.0, key="sweep_rho",
                                        help="Default: 2500 (rock)")
            sweep_mass_per_part = st.number_input("Mass per particle [kg]",
                                                  value=sweep_default_mass, step=10000.0, key="sweep_mass",
                                                  help="Default: 280,000 (Standard) / 500,000 (Fast)")
            sweep_delta_th = st.number_input("Release thickness per particle [m]",
                                            value=sweep_default_delta, step=0.5, key="sweep_delta",
                                            help="Default: 2.0 (Standard) / 4.0 (Fast)")
            sweep_timeout = st.number_input("Timeout per sim [s]", value=3600, key="sweep_timeout",
                                           help="Default: 3600 (1 hour)")
    
    st.divider()
    
    # Submit sweep
    col_sweep_submit, col_sweep_status = st.columns([1, 2])
    
    with col_sweep_submit:
        can_sweep = sweep_thickness_path and sweep_polygon_path
        
        if st.button("🚀 Start Parameter Sweep", type="primary",
                    disabled=not can_sweep, width="stretch"):
            
            thickness_name = Path(sweep_thickness_path).stem.replace('_thickness', '')
            sweep_name = f"sweep_{thickness_name}_{mu_steps}x{xi_steps}"
            
            job = job_queue.submit(
                job_type='simulation_sweep',
                project_name=project.name,
                created_by=username,
                params={
                    'thickness_path': sweep_thickness_path,
                    'release_polygon_path': sweep_polygon_path,
                    'mu_min': mu_min,
                    'mu_max': mu_max,
                    'mu_steps': mu_steps,
                    'xi_min': xi_min,
                    'xi_max': xi_max,
                    'xi_steps': xi_steps,
                    'rho': sweep_rho,
                    'mass_per_part': sweep_mass_per_part,
                    'delta_th': sweep_delta_th,
                    'timeout': sweep_timeout,
                    'sweep_name': sweep_name,
                    'dem_path': project.config.dem_path,
                }
            )
            
            st.success(f"Sweep submitted! Job ID: {job.id}")
    
    with col_sweep_status:
        if not can_sweep:
            st.warning("Select thickness raster and release polygon.")


# =============================================================================
# TAB 3: Results
# =============================================================================
with tab_results:
    st.header("Simulation Results")
    
    # Find completed simulations
    sim_dirs = []
    if project.simulations_dir.exists():
        # Single simulations
        for d in project.simulations_dir.iterdir():
            if d.is_dir() and not d.name.startswith('sweep_'):
                result_file = d / 'simulation_result.json'
                if result_file.exists():
                    sim_dirs.append(('single', d))
        
        # Sweeps
        for d in project.simulations_dir.iterdir():
            if d.is_dir() and d.name.startswith('sweep_'):
                summary_file = d / 'sweep_summary.csv'
                if summary_file.exists():
                    sim_dirs.append(('sweep', d))
    
    if not sim_dirs:
        st.info("No completed simulations yet.")
    else:
        # Organize by type
        singles = [d for t, d in sim_dirs if t == 'single']
        sweeps = [d for t, d in sim_dirs if t == 'sweep']
        
        # Sweeps section
        if sweeps:
            st.subheader(f"Parameter Sweeps ({len(sweeps)})")
            
            for sweep_dir in sweeps:
                with st.expander(f"📊 {sweep_dir.name}", expanded=False):
                    summary_path = sweep_dir / 'sweep_summary.csv'
                    df = pd.read_csv(summary_path)
                    
                    # Summary metrics
                    successful = df[df['success'] == True] if 'success' in df.columns else df
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(df))
                    with col2:
                        st.metric("Successful", len(successful))
                    with col3:
                        if 'max_velocity' in successful.columns:
                            st.metric("Max Velocity", f"{successful['max_velocity'].max():.1f} m/s")
                    with col4:
                        if 'affected_area_ha' in successful.columns:
                            st.metric("Max Area", f"{successful['affected_area_ha'].max():.1f} ha")
                    
                    # Show table
                    display_cols = ['sim_id', 'mu', 'xi', 'max_velocity', 
                                   'max_thickness', 'affected_area_ha', 'success']
                    display_cols = [c for c in display_cols if c in df.columns]
                    st.dataframe(df[display_cols], width="stretch")
                    
                    # Actions
                    col_action1, col_action2 = st.columns(2)
                    with col_action1:
                        if st.button("📊 Open in Explorer", key=f"explore_{sweep_dir.name}"):
                            st.session_state['explorer_sweep_path'] = str(sweep_dir)
                            st.switch_page("pages/3_📊_Explorer.py")
                    with col_action2:
                        if st.button("📤 Export", key=f"export_{sweep_dir.name}"):
                            st.info("Export coming soon...")
        
        # Singles section
        if singles:
            st.subheader(f"Single Simulations ({len(singles)})")
            
            for sim_dir in singles[:10]:  # Show last 10
                result_file = sim_dir / 'simulation_result.json'
                with open(result_file) as f:
                    result = json.load(f)
                
                success_icon = "✅" if result.get('success') else "❌"
                
                with st.expander(f"{success_icon} {sim_dir.name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("μ", f"{result.get('mu', 0):.3f}")
                    with col2:
                        st.metric("ξ", f"{result.get('xi', 0):.0f}")
                    with col3:
                        st.metric("Max Velocity", f"{result.get('max_velocity', 0):.1f} m/s")
                    
                    if result.get('error'):
                        st.error(result['error'][:500])


# =============================================================================
# TAB 4: Jobs
# =============================================================================
with tab_jobs:
    st.header("Simulation Jobs")

    # System resource monitoring
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        has_psutil = True
    except ImportError:
        has_psutil = False
        cpu_percent = 0
        mem_percent = 0

    # Queue status
    queue_status = job_queue.get_queue_status()
    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    with col_w1:
        st.metric("Running", queue_status['running_jobs'], help="Number of jobs currently running")
    with col_w2:
        st.metric("Pending", queue_status['pending_jobs'], help="Jobs waiting to start")
    with col_w3:
        st.metric("Job Slots", f"{queue_status['max_workers']}", help="Max concurrent jobs")
    with col_w4:
        if has_psutil:
            st.metric("CPU", f"{cpu_percent:.0f}%")
        else:
            st.metric("CPU", "N/A")
    with col_w5:
        if has_psutil:
            st.metric("Memory", f"{mem_percent:.0f}%", help=f"{mem_used_gb:.1f} / {mem_total_gb:.1f} GB")
        else:
            st.metric("Memory", "N/A")
    st.divider()

    # Get jobs (include probability ensembles)
    sim_jobs = [j for j in job_queue.list_jobs(project_name=project.name, limit=30)
                if j.job_type in ['single_simulation', 'simulation_sweep', 'probability_ensemble']]

    if not sim_jobs:
        st.info("No simulation jobs yet.")
    else:
        for job in sim_jobs:
            status_emoji = {
                JobStatus.PENDING: "⏳",
                JobStatus.RUNNING: "▶️",
                JobStatus.COMPLETED: "✅",
                JobStatus.FAILED: "❌",
                JobStatus.CANCELLED: "🚫"
            }.get(job.status, "❓")

            job_type_labels = {
                'simulation_sweep': "Sweep",
                'single_simulation': "Single",
                'probability_ensemble': "Prob. Ensemble"
            }
            job_type_label = job_type_labels.get(job.job_type, job.job_type)
            
            with st.expander(
                f"{status_emoji} {job_type_label} — {job.id} — {job.status.value}",
                expanded=(job.status == JobStatus.RUNNING)
            ):
                col_info, col_progress = st.columns([1, 2])
                
                with col_info:
                    st.markdown(f"**Type:** {job.job_type}")
                    st.markdown(f"**Created:** {job.created_at}")
                    st.markdown(f"**By:** {job.created_by}")
                
                with col_progress:
                    if job.status == JobStatus.RUNNING:
                        progress = job.progress
                        worker_info = f" (Worker #{progress.worker_id})" if progress.worker_id else ""
                        st.progress(progress.percent / 100)
                        st.caption(f"{progress.current}/{progress.total} — {progress.message}{worker_info}")

                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("🔄 Refresh", key=f"refresh_sim_{job.id}"):
                                st.rerun()
                        with col_btn2:
                            if st.button("❌ Cancel", key=f"cancel_running_{job.id}"):
                                job_queue.cancel_job(job.id)
                                st.rerun()

                    elif job.status == JobStatus.COMPLETED and job.result:
                        # Handle different job result formats
                        if job.job_type == 'probability_ensemble':
                            stats = job.result.get('statistics', {})
                            st.success(f"✓ {stats.get('successful', 0)}/{stats.get('total_simulations', 1)} simulations")
                            st.caption(f"Output: {job.result.get('output_dir', 'N/A')}")
                        elif 'successful' in job.result:
                            st.success(f"✓ {job.result.get('successful', 0)}/{job.result.get('total', 1)} successful")
                        elif job.result.get('success'):
                            st.success(f"✓ Completed — Max velocity: {job.result.get('max_velocity', 0):.1f} m/s, Area: {job.result.get('affected_area_ha', 0):.1f} ha")
                        else:
                            st.error(f"Failed: {job.result.get('error', 'Unknown error')[:200]}")
                        # Get output path for display
                        output_path = None
                        if job.result:
                            output_path = job.result.get('output_dir') or job.result.get('sim_dir')
                        if st.button("🗑️ Delete Job & Files", key=f"delete_completed_{job.id}",
                                    help=f"Delete job and output files{': ' + output_path if output_path else ''}"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.FAILED:
                        st.error("Failed")
                        if job.error:
                            st.code(job.error[:500])
                        if st.button("🗑️ Delete Job & Files", key=f"delete_failed_{job.id}",
                                    help="Delete job record and any partial output files"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.CANCELLED:
                        st.warning("Cancelled")
                        # Get output path for display
                        output_path = None
                        if job.result:
                            output_path = job.result.get('output_dir') or job.result.get('sim_dir')
                        if st.button("🗑️ Delete Job & Files", key=f"delete_cancelled_{job.id}",
                                    help=f"Delete job and output files{': ' + output_path if output_path else ''}"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.PENDING:
                        if st.button("❌ Cancel", key=f"cancel_sim_{job.id}"):
                            job_queue.cancel_job(job.id)
                            st.rerun()
    
    # Auto-refresh
    st.divider()
    if st.checkbox("Auto-refresh (5s)", key="auto_refresh_sim"):
        import time
        time.sleep(5)
        st.rerun()
