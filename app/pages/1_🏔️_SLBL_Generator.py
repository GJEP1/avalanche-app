"""
SLBL Generator Page
===================
Generate potential failure surfaces using the SLBL method.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import tempfile
import zipfile
import shutil

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import project_selector_sidebar, get_current_project
from core.job_queue import get_job_queue, JobStatus
from core.slbl_handler import register_slbl_handlers

# Page config
st.set_page_config(
    page_title="SLBL Generator",
    page_icon="🏔️",
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
    st.info("Go to Home to create a new project.")
    st.stop()

# Initialize job queue and register handlers
job_queue = get_job_queue()
register_slbl_handlers(job_queue)

# Page title
st.title("🏔️ SLBL Failure Surface Generator")
st.caption(f"Project: {project.name}")

# Tabs for different sections
tab_generate, tab_results, tab_jobs = st.tabs([
    "📤 Generate", "📊 Results", "⚙️ Jobs"
])

# =============================================================================
# TAB 1: Generate
# =============================================================================
with tab_generate:
    st.header("Generate SLBL Thickness Rasters")
    
    col_upload, col_params = st.columns([1, 1])
    
    with col_upload:
        st.subheader("1. Scenario Polygons")
        
        # Show existing scenarios
        existing_scenarios = list(project.scenarios_dir.glob("*.shp"))
        scenario_names = [s.stem for s in existing_scenarios]
        
        if existing_scenarios:
            st.markdown("**Available scenarios:**")
            for s in existing_scenarios:
                col_name, col_del = st.columns([4,1])
                with col_name:
                    st.text(s.stem)
                with col_del:
                    if st.button("Delete", key=f"del_scenario_{s.stem}"):
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            f = s.with_suffix(ext)
                            if f.exists():
                                f.unlink()
                        st.rerun()
            # Multi-select for scenarios to process
            selected_scenarios = st.multiselect(
                "Select scenarios to process:",
                options=scenario_names,
                default=None,
                key="selected_scenarios"
            )
            if selected_scenarios:
                st.caption(f"✓ {len(selected_scenarios)} scenario(s) selected")
            else:
                st.caption("⚠️ No scenarios selected")
        else:
            st.info("No scenarios uploaded yet.")
            selected_scenarios = []
        
        st.divider()

        # Upload new scenarios
        st.markdown("**Upload new scenario:**")

        # Initialize session state for tracking processed uploads
        if 'scenario_upload_processed' not in st.session_state:
            st.session_state.scenario_upload_processed = None

        uploaded_file = st.file_uploader(
            "Upload shapefile (ZIP containing .shp, .shx, .dbf, .prj)",
            type=['zip'],
            key="scenario_upload"
        )

        if uploaded_file is not None:
            # Check if we've already processed this exact file
            upload_id = f"{uploaded_file.name}_{uploaded_file.size}"

            if st.session_state.scenario_upload_processed != upload_id:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Extract zip
                    zip_path = Path(tmpdir) / uploaded_file.name
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(tmpdir)

                    # Find and copy shapefiles
                    shp_files = list(Path(tmpdir).glob("**/*.shp"))

                    if shp_files:
                        for shp in shp_files:
                            # Copy all related files
                            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                src = shp.with_suffix(ext)
                                if src.exists():
                                    dst = project.scenarios_dir / src.name
                                    shutil.copy(src, dst)

                        st.session_state.scenario_upload_processed = upload_id
                        st.success(f"Uploaded {len(shp_files)} scenario(s)")
                        st.rerun()
                    else:
                        st.error("No .shp files found in the uploaded ZIP")

        # Cross-section lines upload section
        st.divider()
        st.subheader("Cross-Section Lines (Optional)")

        # Show existing cross-section lines
        xsect_lines_dir = project.inputs_dir / "xsection_lines"
        xsect_lines_dir.mkdir(exist_ok=True)
        existing_xsect_files = list(xsect_lines_dir.glob("*.shp"))
        xsect_names = [f.stem for f in existing_xsect_files]

        if existing_xsect_files:
            st.markdown("**Available cross-section lines:**")
            for f in existing_xsect_files:
                col_name, col_del = st.columns([4, 1])
                with col_name:
                    st.text(f.stem)
                with col_del:
                    if st.button("Delete", key=f"del_xsect_{f.stem}"):
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            comp = f.with_suffix(ext)
                            if comp.exists():
                                comp.unlink()
                        st.rerun()
        else:
            st.info("No cross-section lines uploaded yet.")

        st.divider()

        # Upload new cross-section lines
        st.markdown("**Upload cross-section lines:**")

        # Initialize session state for tracking processed uploads
        if 'xsect_upload_processed' not in st.session_state:
            st.session_state.xsect_upload_processed = None

        xsect_upload = st.file_uploader(
            "Upload shapefile (ZIP containing .shp, .shx, .dbf, .prj)",
            type=['zip'],
            key="xsect_upload"
        )

        if xsect_upload is not None:
            # Check if we've already processed this exact file
            upload_id = f"{xsect_upload.name}_{xsect_upload.size}"

            if st.session_state.xsect_upload_processed != upload_id:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / xsect_upload.name
                    with open(zip_path, 'wb') as f:
                        f.write(xsect_upload.getbuffer())

                    # Extract ZIP
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)

                    # Find and copy shapefiles
                    shp_files = list(Path(tmpdir).rglob("*.shp"))

                    if shp_files:
                        for shp in shp_files:
                            # Copy all shapefile components
                            stem = shp.stem
                            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                src = shp.parent / f"{stem}{ext}"
                                if src.exists():
                                    dst = xsect_lines_dir / f"{stem}{ext}"
                                    shutil.copy2(src, dst)

                        st.session_state.xsect_upload_processed = upload_id
                        st.success(f"Uploaded {len(shp_files)} cross-section line file(s)")
                        st.rerun()
                    else:
                        st.error("No .shp files found in the uploaded ZIP")
    
    with col_params:
        st.subheader("2. SLBL Parameters")

        # E-ratio configuration
        st.markdown("**E-ratio values** (z_max / L_rh)")
        st.caption("Typical range: 0.05 - 0.25")
        
        e_ratio_input = st.text_input(
            "E-ratios (comma-separated):",
            value="0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25",
            key="e_ratios_input"
        )
        
        try:
            e_ratios = [float(x.strip()) for x in e_ratio_input.split(",")]
            st.caption(f"✓ {len(e_ratios)} e-ratio values")
        except ValueError:
            st.error("Invalid e-ratio format")
            e_ratios = []
        
        # Max depth configuration
        st.markdown("**Maximum depth constraints** (optional)")
        use_max_depth = st.checkbox("Apply maximum depth limits", value=False)
        
        if use_max_depth:
            max_depth_input = st.text_input(
                "Max depths in meters (comma-separated):",
                value="50, 100, 150",
                key="max_depths_input"
            )
            try:
                max_depths = [float(x.strip()) for x in max_depth_input.split(",")]
            except ValueError:
                st.error("Invalid max depth format")
                max_depths = [None]
        else:
            max_depths = [None]
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            mode = st.selectbox(
                "Mode:",
                options=["failure", "inverse"],
                index=0,
                help="'failure' for excavation (landslide), 'inverse' for filling"
            )

            neighbours = st.selectbox(
                "Neighbor rule:",
                options=[4, 8],
                index=0,
                help="4 = von Neumann, 8 = Moore neighborhood"
            )
            
            use_z_floor = st.checkbox(
                "Apply z-floor constraint",
                value=True,
                help="Prevents SLBL from going below surrounding terrain"
            )
            
            buffer_pixels = st.number_input(
                "Buffer pixels:",
                min_value=0,
                max_value=20,
                value=4,
                help="Buffer around polygon in pixels"
            )
            
            write_ascii = st.checkbox(
                "Also write ASCII files (for standalone AvaFrame)",
                value=False,
                help="Creates .asc files in addition to GeoTIFF"
            )
            
            # Cross-section generation
            xsect_lines_dir = project.inputs_dir / "xsection_lines"
            has_xsect_lines = len(list(xsect_lines_dir.glob("*.shp"))) > 0 if xsect_lines_dir.exists() else False
            
            write_xsections = st.checkbox(
                "Generate cross-sections",
                value=has_xsect_lines,
                disabled=not has_xsect_lines,
                help="Generate cross-section profiles along uploaded lines" if has_xsect_lines 
                     else "Upload cross-section lines first"
            )
            
            if write_xsections:
                xsect_step_m = st.number_input(
                    "Cross-section sampling step (m):",
                    min_value=0.5,
                    max_value=10.0,
                    value=1.0,
                    step=0.5,
                    help="Distance between sample points along cross-section lines"
                )
                
                xsect_clip_to_poly = st.checkbox(
                    "Clip cross-sections to polygon boundary",
                    value=False,
                    help="Only sample points inside the scenario polygon"
                )
            else:
                xsect_step_m = 1.0
                xsect_clip_to_poly = False
            
            max_iters = st.number_input(
                "Max iterations:",
                min_value=100,
                max_value=10000,
                value=3000
            )

        # Citation info
        st.info(
            "**SLBL Method Reference:**\n\n"
            "Jaboyedoff, M., et al. (2019). \"Testing a failure surface prediction and "
            "deposit reconstruction method for a landslide cluster that occurred during "
            "Typhoon Talas (Japan)\". *Earth Surf. Dynam.*, 7, 439–458.\n\n"
            "[https://esurf.copernicus.org/articles/7/439/2019/](https://esurf.copernicus.org/articles/7/439/2019/)"
        )

    st.divider()
    
    # Job summary and submit
    st.subheader("3. Submit Job")
    
    # Check if we have valid inputs
    can_submit = len(selected_scenarios) > 0 and len(e_ratios) > 0
    
    if can_submit:
        n_scenarios = len(selected_scenarios)
        n_e_ratios = len(e_ratios)
        n_max_depths = len(max_depths)
        total_runs = n_scenarios * n_e_ratios * n_max_depths
        
        col_summary, col_submit = st.columns([2, 1])
        
        with col_summary:
            st.info(f"""
            **Job Summary:**
            - Scenarios: {n_scenarios}
            - E-ratios: {n_e_ratios}
            - Max depth variants: {n_max_depths}
            - **Total SLBL runs: {total_runs}**
            """)
        
        with col_submit:
            if st.button("🚀 Submit SLBL Job", type="primary", width="stretch"):
                # Build scenario paths
                scenario_paths = [
                    str(project.scenarios_dir / f"{name}.shp")
                    for name in selected_scenarios
                ]
                
                # Build cross-section lines paths if enabled
                xsect_lines_source = []
                if write_xsections and has_xsect_lines:
                    xsect_lines_source = [str(project.inputs_dir / "xsection_lines")]
                
                # Submit job
                job = job_queue.submit(
                    job_type='slbl_batch',
                    project_name=project.name,
                    created_by=username,
                    params={
                        'scenario_paths': scenario_paths,
                        'dem_path': project.config.dem_path,
                        'e_ratios': e_ratios,
                        'max_depths': max_depths,
                        'mode': mode,
                        'neighbours': neighbours,
                        'use_z_floor': use_z_floor,
                        'buffer_pixels': buffer_pixels,
                        'write_ascii': write_ascii,
                        'write_xsections': write_xsections,
                        'xsect_lines_source': xsect_lines_source,
                        'xsect_step_m': xsect_step_m,
                        'xsect_clip_to_poly': xsect_clip_to_poly,
                        'max_iters': max_iters,
                    }
                )
                
                st.success(f"Job submitted! ID: {job.id}")
                st.info("Go to the 'Jobs' tab to monitor progress.")
    else:
        if len(selected_scenarios) == 0:
            st.warning("⚠️ Please select at least one scenario from the list above.")
        elif len(e_ratios) == 0:
            st.warning("⚠️ Please configure valid e-ratios.")
        else:
            st.warning("Select at least one scenario and configure e-ratios to submit a job.")


# =============================================================================
# TAB 2: Results
# =============================================================================
with tab_results:
    # Check for results
    summary_path = project.slbl_dir / "slbl_summary.csv"

    if not summary_path.exists():
        st.info("No SLBL results yet. Generate some in the 'Generate' tab.")
    else:
        df = pd.read_csv(summary_path)

        # Compact navigation row at top
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([2, 1, 2, 1, 1])

        # Job filter
        if 'job_id' in df.columns and df['job_id'].notna().any():
            jobs = df[['job_id', 'job_created_by', 'job_created']].drop_duplicates().sort_values('job_created', ascending=False)
            job_options = ["All"] + [f"{row['job_id'][:8]}" for _, row in jobs.iterrows()]

            with nav_col1:
                selected_job_display = st.selectbox("Job", options=job_options, key="job_filter", label_visibility="collapsed")

            if selected_job_display != "All":
                full_job_id = jobs[jobs['job_id'].str.startswith(selected_job_display)]['job_id'].iloc[0]
                df = df[df['job_id'] == full_job_id].copy()

        # Scenario selector
        if 'scenario_name' in df.columns:
            scenarios = sorted(df['scenario_name'].unique().tolist())

            with nav_col2:
                selected_scenario = st.selectbox("Scenario", options=scenarios, key="results_scenario_select", label_visibility="collapsed")

            scenario_df = df[df['scenario_name'] == selected_scenario].copy()
            labels = sorted(scenario_df['label'].unique().tolist()) if 'label' in scenario_df.columns else []

            if labels:
                # Initialize/reset label index
                if 'results_label_idx' not in st.session_state:
                    st.session_state.results_label_idx = 0
                if 'results_last_scenario' not in st.session_state or st.session_state.results_last_scenario != selected_scenario:
                    st.session_state.results_label_idx = 0
                    st.session_state.results_last_scenario = selected_scenario
                st.session_state.results_label_idx = max(0, min(st.session_state.results_label_idx, len(labels) - 1))

                # Initialize selectbox state if needed
                selectbox_key = f"slbl_label_{selected_scenario}"
                if selectbox_key not in st.session_state:
                    st.session_state[selectbox_key] = labels[st.session_state.results_label_idx]

                # Prev/Next buttons - check first but render in columns later
                prev_clicked = False
                next_clicked = False

                with nav_col4:
                    prev_clicked = st.button("←", key="prev_label")

                with nav_col5:
                    next_clicked = st.button("→", key="next_label")

                # Handle button clicks by updating the selectbox's session state key
                if prev_clicked:
                    new_idx = (st.session_state.results_label_idx - 1) % len(labels)
                    st.session_state.results_label_idx = new_idx
                    st.session_state[selectbox_key] = labels[new_idx]
                    st.rerun()

                if next_clicked:
                    new_idx = (st.session_state.results_label_idx + 1) % len(labels)
                    st.session_state.results_label_idx = new_idx
                    st.session_state[selectbox_key] = labels[new_idx]
                    st.rerun()

                # Label selector
                with nav_col3:
                    selected_label = st.selectbox(
                        "Label",
                        options=labels,
                        key=selectbox_key,
                        label_visibility="collapsed"
                    )
                    # Sync index from selectbox selection
                    st.session_state.results_label_idx = labels.index(selected_label)

                label_data = scenario_df[scenario_df['label'] == selected_label].iloc[0]

                # Compact stats row
                vol = label_data.get('volume_m3', 0) / 1e6
                max_d = label_data.get('max_depth_m', 0)
                mean_d = label_data.get('mean_depth_m', 0)
                e_rat = label_data.get('e_ratio', 0)

                st.markdown(f"""
                <div style="display: flex; gap: 20px; font-size: 13px; padding: 8px 0; border-bottom: 1px solid #ddd; margin-bottom: 10px;">
                    <span><b>{selected_scenario}</b> — <code>{selected_label}</code> ({st.session_state.results_label_idx + 1}/{len(labels)})</span>
                    <span>Vol: <b>{vol:.3f}</b> Mm³</span>
                    <span>Max: <b>{max_d:.1f}</b>m</span>
                    <span>Mean: <b>{mean_d:.1f}</b>m</span>
                    <span>E: <b>{e_rat:.3f}</b></span>
                    <span style="margin-left: auto;">
                </div>
                """, unsafe_allow_html=True)

                # Show parameters option
                show_params = st.checkbox("Show parameters", key="show_all_params")

                if show_params:
                    with st.expander("All Parameters", expanded=True):
                        param_df = pd.DataFrame([label_data]).T
                        param_df.columns = ['Value']
                        param_df['Value'] = param_df['Value'].astype(str)
                        st.dataframe(param_df, height=200)

                # 2-column grid for map and cross-sections
                if 'thickness_path' in label_data.index:
                    thickness_path = Path(label_data['thickness_path'])

                    # Collect all plots to display in grid
                    plots = []  # List of (title, fig) tuples

                    # Thickness map
                    if thickness_path.exists():
                        try:
                            import rasterio
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')

                            with rasterio.open(thickness_path) as src:
                                thickness = src.read(1, masked=True)
                                valid_data = thickness.compressed()

                                if len(valid_data) > 0:
                                    plt.close('all')
                                    fig, ax = plt.subplots(figsize=(5, 3.5))

                                    vmin = 0
                                    vmax = np.percentile(valid_data, 95)

                                    im = ax.imshow(
                                        thickness, cmap='viridis', vmin=vmin, vmax=vmax,
                                        extent=[src.bounds.left, src.bounds.right,
                                               src.bounds.bottom, src.bounds.top],
                                        origin='upper'
                                    )

                                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                    cbar.set_label('Thickness (m)', fontsize=8)
                                    cbar.ax.tick_params(labelsize=7)

                                    ax.set_title(f"Thickness Map", fontsize=9)
                                    ax.set_xlabel('Easting (m)', fontsize=8)
                                    ax.set_ylabel('Northing (m)', fontsize=8)
                                    ax.tick_params(labelsize=7)

                                    stats_text = f"Max: {valid_data.max():.1f}m\nMean: {valid_data.mean():.1f}m"
                                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                           verticalalignment='top', fontsize=8,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                                    plt.tight_layout()
                                    plots.append(("Thickness", fig))
                        except Exception as e:
                            st.error(f"Error loading thickness map: {e}")

                    # Cross-sections
                    xsect_dir = project.slbl_dir / "xsections" / selected_scenario
                    if xsect_dir.exists():
                        xsect_files = list(xsect_dir.glob(f"{selected_scenario}_{selected_label}_xsect_*.csv"))

                        if xsect_files:
                            # Group by line ID
                            line_map = {}
                            for f in xsect_files:
                                try:
                                    line_id = f.stem.split('_xsect_')[1].split('_p')[0]
                                    line_map.setdefault(line_id, []).append(f)
                                except:
                                    continue

                            try:
                                line_ids = sorted(line_map.keys(), key=lambda x: int(x))
                            except:
                                line_ids = sorted(line_map.keys())

                            # Create cross-section plots
                            for line_id in line_ids:
                                parts = sorted(line_map[line_id])
                                dfs = []
                                for p in parts:
                                    try:
                                        dfs.append(pd.read_csv(p))
                                    except:
                                        continue

                                if dfs:
                                    xs_df = pd.concat(dfs, ignore_index=True)

                                    for col in ['Distance_m', 'DEM_z', 'Base_z']:
                                        if col in xs_df.columns:
                                            xs_df[col] = pd.to_numeric(xs_df[col], errors='coerce')

                                    if 'Distance_m' in xs_df.columns:
                                        xs_df = xs_df.sort_values('Distance_m')

                                    if 'DEM_z' in xs_df.columns and 'Base_z' in xs_df.columns:
                                        try:
                                            plt.close('all')
                                            fig, ax = plt.subplots(figsize=(5, 3.5))

                                            dist = xs_df['Distance_m'].to_numpy()
                                            dem = xs_df['DEM_z'].to_numpy()
                                            base = xs_df['Base_z'].to_numpy()

                                            # Smooth the data to reduce stair-stepping from DEM resolution
                                            mask = np.isfinite(dist) & np.isfinite(dem) & np.isfinite(base)
                                            if np.sum(mask) > 15:
                                                try:
                                                    from scipy.ndimage import uniform_filter1d
                                                    window = min(11, np.sum(mask) // 3)
                                                    if window % 2 == 0:
                                                        window += 1
                                                    dem_smooth = uniform_filter1d(dem[mask].astype(float), size=window, mode='nearest')
                                                    base_smooth = uniform_filter1d(base[mask].astype(float), size=window, mode='nearest')
                                                except ImportError:
                                                    dem_smooth = dem[mask]
                                                    base_smooth = base[mask]
                                            else:
                                                dem_smooth = dem[mask]
                                                base_smooth = base[mask]
                                            dist_plot = dist[mask]

                                            ax.plot(dist_plot, dem_smooth,
                                                   label='DEM', linewidth=1.5, color='#2E4057')
                                            ax.plot(dist_plot, base_smooth,
                                                   label='Base', linewidth=1.5, color='#D2691E')

                                            ax.fill_between(dist_plot, base_smooth, dem_smooth,
                                                           where=(dem_smooth >= base_smooth),
                                                           alpha=0.3, color='#8B4513', label='Thickness')

                                            ax.set_xlabel('Distance (m)', fontsize=8)
                                            ax.set_ylabel('Elevation (m)', fontsize=8)
                                            ax.set_title(f"X-Section {line_id}", fontsize=9)
                                            ax.legend(fontsize=7, loc='best')
                                            ax.grid(True, alpha=0.3, linestyle='--')
                                            ax.tick_params(labelsize=7)

                                            plt.tight_layout()
                                            plots.append((f"XS-{line_id}", fig))
                                        except:
                                            pass

                    # Display plots in 2-column grid
                    if plots:
                        import io
                        n_cols = 2
                        for i in range(0, len(plots), n_cols):
                            cols = st.columns(n_cols)
                            for j in range(n_cols):
                                if i + j < len(plots):
                                    title, fig = plots[i + j]
                                    with cols[j]:
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                        buf.seek(0)
                                        st.image(buf.getvalue(), use_container_width=False, width=500)
                                        plt.close(fig)
                    else:
                        st.info("No plots available")

            else:
                st.info("No results for this scenario")
        else:
            st.warning("No scenario information in results")


# =============================================================================
# TAB 3: Jobs
# =============================================================================
with tab_jobs:
    st.header("Job Queue")

    # Queue status
    queue_status = job_queue.get_queue_status()
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.metric("Active Workers", f"{queue_status['active_workers']}/{queue_status['max_workers']}")
    with col_w2:
        st.metric("Running Jobs", queue_status['running_jobs'])
    with col_w3:
        st.metric("Pending", queue_status['pending_jobs'])
    st.divider()

    # Get jobs for this project
    jobs = job_queue.list_jobs(project_name=project.name, limit=20)

    if not jobs:
        st.info("No jobs submitted yet.")
    else:
        for job in jobs:
            status_emoji = {
                JobStatus.PENDING: "⏳",
                JobStatus.RUNNING: "▶️",
                JobStatus.COMPLETED: "✅",
                JobStatus.FAILED: "❌",
                JobStatus.CANCELLED: "🚫"
            }.get(job.status, "❓")
            
            with st.expander(
                f"{status_emoji} {job.job_type} — {job.id} — {job.status.value}",
                expanded=(job.status == JobStatus.RUNNING)
            ):
                col_info, col_progress = st.columns([1, 2])
                
                with col_info:
                    st.markdown(f"**Created:** {job.created_at}")
                    st.markdown(f"**By:** {job.created_by}")
                    if job.completed_at:
                        st.markdown(f"**Completed:** {job.completed_at}")
                
                with col_progress:
                    if job.status == JobStatus.RUNNING:
                        progress = job.progress
                        worker_info = f" (Worker #{progress.worker_id})" if progress.worker_id else ""
                        st.progress(progress.percent / 100)
                        st.caption(f"{progress.current}/{progress.total} — {progress.message}{worker_info}")

                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("🔄 Refresh", key=f"refresh_{job.id}"):
                                st.rerun()
                        with col_btn2:
                            if st.button("❌ Cancel", key=f"cancel_running_{job.id}"):
                                job_queue.cancel_job(job.id)
                                st.rerun()

                    elif job.status == JobStatus.COMPLETED:
                        if job.result:
                            st.success(f"Completed: {job.result.get('successful', 0)}/{job.result.get('total', 0)} successful")
                            if job.result.get('failed', 0) > 0:
                                st.warning(f"Failed: {job.result.get('failed', 0)}")
                        if st.button("🗑️ Delete", key=f"delete_completed_{job.id}"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.FAILED:
                        st.error("Job failed")
                        if job.error:
                            st.code(job.error[:500])
                        if st.button("🗑️ Delete", key=f"delete_failed_{job.id}"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.CANCELLED:
                        st.warning("Cancelled")
                        if st.button("🗑️ Delete", key=f"delete_cancelled_{job.id}"):
                            job_queue.delete_job(job.id)
                            st.rerun()

                    elif job.status == JobStatus.PENDING:
                        if st.button("❌ Cancel", key=f"cancel_{job.id}"):
                            job_queue.cancel_job(job.id)
                            st.rerun()
                
                # Show parameters
                with st.expander("Parameters"):
                    st.json(job.params)

    # Auto-refresh toggle
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (every 5 seconds)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
