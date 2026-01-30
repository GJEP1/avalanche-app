"""
Avalanche Modeling Application - Home Page
==========================================
Main entry point with authentication and workflow navigation.
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import project_selector_sidebar, get_current_project

# Page configuration
st.set_page_config(
    page_title="Avalanche Modeling",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .workflow-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .workflow-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .step-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Authentication
is_authenticated, username, name = require_authentication()

if not is_authenticated:
    st.stop()

# Show user info in sidebar
show_user_info_sidebar()

# Project selector
project = project_selector_sidebar()

# Main content
st.title("🏔️ Rock Avalanche Modeling")
st.markdown("*Integrated workflow for SLBL failure surface generation and runout simulation*")

if project is None:
    st.warning("Please create or select a project to continue.")
    st.stop()

st.success(f"**Active Project:** {project.name}")
if project.config.description:
    st.caption(project.config.description)

st.divider()

# Workflow overview
st.header("Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">1</div>
        <h3>🗺️ Scenario Editor</h3>
        <p>Draw and manage scenario polygons and cross-section lines
        on an interactive map.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Scenario Editor", key="btn_scenario", width="stretch"):
        st.switch_page("pages/0_🗺️_Scenario_Editor.py")

with col2:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">2</div>
        <h3>🏔️ SLBL Generation</h3>
        <p>Generate potential failure surfaces using the SLBL method
        with various e-ratio parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open SLBL Generator", key="btn_slbl", width="stretch"):
        st.switch_page("pages/1_🏔️_SLBL_Generator.py")

with col3:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">3</div>
        <h3>🎯 Simulation</h3>
        <p>Run AvaFrame rock avalanche simulations using SLBL outputs.
        Single runs or parameter sweeps.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Simulation", key="btn_sim", width="stretch"):
        st.switch_page("pages/2_🎯_Simulation.py")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">4</div>
        <h3>📊 Results Explorer</h3>
        <p>Explore simulation results interactively. Compare against
        empirical relationships (Strom, Brideau).</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Explorer", key="btn_explorer", width="stretch"):
        st.switch_page("pages/3_📊_Results_Explorer.py")

with col5:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">5</div>
        <h3>🎲 Probability Ensemble</h3>
        <p>Generate probability maps combining multiple scenarios
        and parameter sets.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Probability Ensemble", key="btn_prob", width="stretch"):
        st.switch_page("pages/4_🎲_Probability_Ensemble.py")

with col6:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">6</div>
        <h3>📁 File Manager</h3>
        <p>View and manage project files, simulations, SLBL results,
        and monitor disk space usage.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open File Manager", key="btn_files", width="stretch"):
        st.switch_page("pages/5_📁_File_Manager.py")

st.divider()

# Project status summary
st.header("Project Status")

col_a, col_b, col_c = st.columns(3)

# Count SLBL outputs
slbl_count = len(list(project.slbl_dir.glob("*_slbl_thickness.tif"))) if project.slbl_dir.exists() else 0
# Count simulations  
sim_count = len(list(project.simulations_dir.glob("*/Outputs"))) if project.simulations_dir.exists() else 0
# Count scenarios
scenario_count = len(list(project.scenarios_dir.glob("*.shp"))) if project.scenarios_dir.exists() else 0

with col_a:
    st.metric("Scenario Polygons", scenario_count)
    
with col_b:
    st.metric("SLBL Thickness Rasters", slbl_count)

with col_c:
    st.metric("Completed Simulations", sim_count)

# Quick actions
st.header("Quick Actions")

qa_col1, qa_col2, qa_col3 = st.columns(3)

with qa_col1:
    st.markdown("**Upload Scenario Polygon**")
    uploaded_shp = st.file_uploader(
        "Upload shapefile (ZIP)",
        type=['zip'],
        key="quick_upload_shp",
        help="Upload a zipped shapefile containing scenario outline polygons"
    )
    if uploaded_shp:
        # Handle upload - extract to scenarios dir
        import zipfile
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / uploaded_shp.name
            with open(zip_path, 'wb') as f:
                f.write(uploaded_shp.getbuffer())
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(project.scenarios_dir)
        
        st.success(f"Uploaded to {project.scenarios_dir}")
        st.rerun()

with qa_col2:
    st.markdown("**Project Settings**")
    if st.button("Edit Project Settings", width="stretch"):
        st.session_state['show_project_settings'] = True

with qa_col3:
    st.markdown("**Export Project**")
    if st.button("Export All Results", width="stretch"):
        st.info("Export functionality coming soon...")

# Project settings modal
if st.session_state.get('show_project_settings'):
    with st.expander("⚙️ Project Settings", expanded=True):
        st.text_input("DEM Path", value=project.config.dem_path, key="settings_dem")
        st.number_input("Target EPSG", value=project.config.target_epsg, key="settings_epsg")
        
        st.markdown("**SLBL Default Parameters**")
        st.text_input("E-ratios (comma-separated)", 
                     value=",".join(map(str, project.config.slbl_e_ratios)),
                     key="settings_e_ratios")
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("Save Settings", type="primary"):
                # Update config and save
                project.config.dem_path = st.session_state.settings_dem
                project.config.target_epsg = st.session_state.settings_epsg
                project.config.slbl_e_ratios = [float(x.strip()) for x in 
                                                st.session_state.settings_e_ratios.split(",")]
                project.save_config()
                st.session_state['show_project_settings'] = False
                st.success("Settings saved!")
                st.rerun()
        with col_cancel:
            if st.button("Cancel"):
                st.session_state['show_project_settings'] = False
                st.rerun()
