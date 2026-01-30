"""
Scenario Editor Page
====================
Interactive map for creating and managing scenario polygons and cross-section lines.
"""

import streamlit as st
import geopandas as gpd
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import shutil
from datetime import datetime

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import (
    project_selector_sidebar, get_current_project, list_projects, Project,
    copy_scenarios_between_projects, copy_xsections_between_projects,
    get_project_scenarios, get_project_xsections
)
from components.ol_map import ol_map
from components.ol_map.utils import (
    load_project_scenarios,
    load_project_xsections,
    geodataframe_to_geojson,
    save_feature_as_shapefile,
    save_geojson_as_shapefiles,
    delete_shapefile,
    get_project_center,
    get_project_bounds,
    calculate_polygon_area,
    calculate_line_length
)

# Page config
st.set_page_config(
    page_title="Scenario Editor",
    page_icon="🗺️",
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

st.title("🗺️ Scenario Editor")
st.caption(f"Project: {project.name}")

# =============================================================================
# Session State
# =============================================================================

if 'map_event_processed' not in st.session_state:
    st.session_state.map_event_processed = None

if 'selected_feature' not in st.session_state:
    st.session_state.selected_feature = None

if 'selected_feature_layer' not in st.session_state:
    st.session_state.selected_feature_layer = None

# Visibility state for features (keyed by project name to isolate per project)
visibility_key = f"feature_visibility_{project.name}"
if visibility_key not in st.session_state:
    st.session_state[visibility_key] = {'scenarios': {}, 'xsections': {}}

# =============================================================================
# Load Project Data
# =============================================================================

scenarios_gdf = load_project_scenarios(project)
xsections_gdf = load_project_xsections(project)

# Initialize visibility for any new features (default to visible)
for _, row in scenarios_gdf.iterrows():
    name = row.get('name', 'unnamed')
    if name not in st.session_state[visibility_key]['scenarios']:
        st.session_state[visibility_key]['scenarios'][name] = True

for _, row in xsections_gdf.iterrows():
    name = row.get('name', 'unnamed')
    if name not in st.session_state[visibility_key]['xsections']:
        st.session_state[visibility_key]['xsections'][name] = True

# Filter GeoDataFrames based on visibility
visible_scenarios = scenarios_gdf[scenarios_gdf['name'].apply(
    lambda n: st.session_state[visibility_key]['scenarios'].get(n, True)
)]
visible_xsections = xsections_gdf[xsections_gdf['name'].apply(
    lambda n: st.session_state[visibility_key]['xsections'].get(n, True)
)]

scenarios_geojson = geodataframe_to_geojson(visible_scenarios)
xsections_geojson = geodataframe_to_geojson(visible_xsections)

# Get initial view
center = get_project_center(project)
bounds = get_project_bounds(project)

if center is None:
    # Default to central Norway
    center = [62.0, 10.0]
    zoom = 6
else:
    zoom = 12

# =============================================================================
# Layout
# =============================================================================

col_map, col_panel = st.columns([3, 1])

with col_panel:
    st.subheader("Map Settings")

    basemap = st.selectbox(
        "Base map:",
        ["Grunnkart", "Terrain", "Satellite (ESRI)"],
        index=0
    )

    show_hillshade = False  # Disabled - use Terrain basemap instead

    # Legend
    st.subheader("Legend")
    st.markdown("""
    <div style="font-size: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 20px; height: 12px; background: rgba(255,120,0,0.4); border: 2px solid #ff7800; margin-right: 8px;"></div>
            <span>Scenario polygon</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 20px; height: 3px; background: #0066ff; margin-right: 8px;"></div>
            <span>Cross-section line</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 12px; background: rgba(0,200,0,0.4); border: 2px dashed #00c800; margin-right: 8px;"></div>
            <span>Drawing (unsaved)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # === Selected Feature Info ===
    if st.session_state.selected_feature:
        st.subheader("Selected Feature")
        feat = st.session_state.selected_feature
        layer = st.session_state.selected_feature_layer
        props = feat.get('properties', {}) or {}
        geom = feat.get('geometry', {})

        name = props.get('name', 'Unnamed')
        geom_type = geom.get('type', 'Unknown')

        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Type:** {layer.replace('s', '') if layer else geom_type}")

        # Calculate measurements
        if geom_type in ['Polygon', 'MultiPolygon']:
            from components.ol_map.utils import calculate_polygon_area
            area = calculate_polygon_area(feat, source_crs=25833)
            st.markdown(f"**Area:** {area:.3f} km²")
        elif geom_type in ['LineString', 'MultiLineString']:
            from components.ol_map.utils import calculate_line_length
            length = calculate_line_length(feat, source_crs=25833)
            st.markdown(f"**Length:** {length:.3f} km")

        # Show coordinates summary
        if geom_type == 'Polygon' and geom.get('coordinates'):
            coords = geom['coordinates'][0]
            st.caption(f"Vertices: {len(coords)}")
        elif geom_type == 'LineString' and geom.get('coordinates'):
            coords = geom['coordinates']
            st.caption(f"Points: {len(coords)}")

        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_feature = None
            st.session_state.selected_feature_layer = None
            st.rerun()

        st.divider()

    # === Existing Features ===
    st.subheader("Project Features")

    # Scenarios
    visible_scen_count = sum(1 for v in st.session_state[visibility_key]['scenarios'].values() if v)
    with st.expander(f"📍 Scenarios ({visible_scen_count}/{len(scenarios_gdf)} visible)", expanded=True):
        if scenarios_gdf.empty:
            st.caption("No scenarios yet. Draw polygons on the map.")
        else:
            for idx, row in scenarios_gdf.iterrows():
                name = row.get('name', 'unnamed')
                col_vis, col_name, col_del = st.columns([1, 3, 1])

                with col_vis:
                    is_visible = st.checkbox(
                        "👁",
                        value=st.session_state[visibility_key]['scenarios'].get(name, True),
                        key=f"vis_scen_{idx}",
                        label_visibility="collapsed",
                        help="Toggle visibility"
                    )
                    if is_visible != st.session_state[visibility_key]['scenarios'].get(name, True):
                        st.session_state[visibility_key]['scenarios'][name] = is_visible
                        st.rerun()

                with col_name:
                    if row.geometry and row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                        area_km2 = row.geometry.area / 1e6
                        label = f"{name} ({area_km2:.2f} km²)"
                    else:
                        label = name
                    # Gray out if hidden
                    if not is_visible:
                        st.markdown(f"<span style='color: #888;'>{label}</span>", unsafe_allow_html=True)
                    else:
                        st.text(label)

                with col_del:
                    if st.button("×", key=f"del_scen_{idx}", help="Delete scenario"):
                        shp_path = project.scenarios_dir / f"{row['name']}.shp"
                        if delete_shapefile(shp_path):
                            # Remove from visibility state
                            st.session_state[visibility_key]['scenarios'].pop(name, None)
                            st.rerun()

    # Cross-sections
    xsect_dir = project.inputs_dir / "xsection_lines"
    xsect_dir.mkdir(exist_ok=True)

    visible_xsect_count = sum(1 for v in st.session_state[visibility_key]['xsections'].values() if v)
    with st.expander(f"📏 Cross-sections ({visible_xsect_count}/{len(xsections_gdf)} visible)", expanded=True):
        if xsections_gdf.empty:
            st.caption("No cross-sections yet. Draw lines on the map.")
        else:
            for idx, row in xsections_gdf.iterrows():
                name = row.get('name', 'unnamed')
                col_vis, col_name, col_del = st.columns([1, 3, 1])

                with col_vis:
                    is_visible = st.checkbox(
                        "👁",
                        value=st.session_state[visibility_key]['xsections'].get(name, True),
                        key=f"vis_xsect_{idx}",
                        label_visibility="collapsed",
                        help="Toggle visibility"
                    )
                    if is_visible != st.session_state[visibility_key]['xsections'].get(name, True):
                        st.session_state[visibility_key]['xsections'][name] = is_visible
                        st.rerun()

                with col_name:
                    if row.geometry and row.geometry.geom_type in ['LineString', 'MultiLineString']:
                        length_km = row.geometry.length / 1000
                        label = f"{name} ({length_km:.2f} km)"
                    else:
                        label = name
                    # Gray out if hidden
                    if not is_visible:
                        st.markdown(f"<span style='color: #888;'>{label}</span>", unsafe_allow_html=True)
                    else:
                        st.text(label)

                with col_del:
                    if st.button("×", key=f"del_xsect_{idx}", help="Delete cross-section"):
                        shp_path = xsect_dir / f"{row['name']}.shp"
                        if delete_shapefile(shp_path):
                            # Remove from visibility state
                            st.session_state[visibility_key]['xsections'].pop(name, None)
                            st.rerun()

    st.divider()

    # === Save Drawings ===
    st.subheader("Save Drawings")

    # Feature name input
    feature_name = st.text_input(
        "Feature name:",
        value=f"feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        key="save_feature_name"
    )

    save_col1, save_col2 = st.columns(2)

    with save_col1:
        save_as_scenario = st.button(
            "💾 As Scenario",
            use_container_width=True,
            help="Save drawn polygons as scenario"
        )

    with save_col2:
        save_as_xsection = st.button(
            "💾 As X-Section",
            use_container_width=True,
            help="Save drawn lines as cross-section"
        )

    st.divider()

    # === Upload ===
    st.subheader("Upload Shapefile")

    # Initialize session state for tracking processed uploads
    if 'editor_upload_processed' not in st.session_state:
        st.session_state.editor_upload_processed = None

    upload_type = st.radio(
        "Upload as:",
        ["Scenario polygon", "Cross-section line"],
        key="editor_upload_type",
        horizontal=True
    )

    uploaded_file = st.file_uploader(
        "Shapefile (ZIP)",
        type=['zip'],
        key="editor_upload_shapefile"
    )

    if uploaded_file is not None:
        upload_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.editor_upload_processed != upload_id:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / uploaded_file.name
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)

                shp_files = list(Path(tmpdir).rglob("*.shp"))

                if shp_files:
                    if upload_type == "Scenario polygon":
                        output_dir = project.scenarios_dir
                    else:
                        output_dir = xsect_dir

                    output_dir.mkdir(parents=True, exist_ok=True)

                    for shp in shp_files:
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            src = shp.with_suffix(ext)
                            if src.exists():
                                dst = output_dir / src.name
                                shutil.copy(src, dst)

                    st.session_state.editor_upload_processed = upload_id
                    st.success(f"Uploaded {len(shp_files)} file(s)")
                    st.rerun()
                else:
                    st.error("No .shp files found in ZIP")

    st.divider()

    # === Copy from Other Projects ===
    st.subheader("Import from Project")

    other_projects = [p for p in list_projects() if p != project.name]

    if other_projects:
        source_project_name = st.selectbox(
            "Source project:",
            other_projects,
            key="copy_source_project"
        )

        if source_project_name:
            try:
                source_project = Project.load(source_project_name)

                # Get available scenarios and xsections from source
                source_scenarios = get_project_scenarios(source_project)
                source_xsections = get_project_xsections(source_project)

                with st.expander(f"📍 Scenarios ({len(source_scenarios)})", expanded=False):
                    if source_scenarios:
                        selected_scenarios = st.multiselect(
                            "Select scenarios to copy:",
                            source_scenarios,
                            key="copy_scenarios_select"
                        )
                        if st.button("Copy Scenarios", disabled=not selected_scenarios,
                                    key="copy_scenarios_btn"):
                            copied = copy_scenarios_between_projects(
                                source_project, project, selected_scenarios
                            )
                            if copied:
                                st.success(f"Copied {len(copied)} scenario(s)")
                                st.rerun()
                            else:
                                st.error("Failed to copy scenarios")
                    else:
                        st.caption("No scenarios in source project")

                with st.expander(f"📏 Cross-sections ({len(source_xsections)})", expanded=False):
                    if source_xsections:
                        selected_xsections = st.multiselect(
                            "Select cross-sections to copy:",
                            source_xsections,
                            key="copy_xsections_select"
                        )
                        if st.button("Copy Cross-sections", disabled=not selected_xsections,
                                    key="copy_xsections_btn"):
                            copied = copy_xsections_between_projects(
                                source_project, project, selected_xsections
                            )
                            if copied:
                                st.success(f"Copied {len(copied)} cross-section(s)")
                                st.rerun()
                            else:
                                st.error("Failed to copy cross-sections")
                    else:
                        st.caption("No cross-sections in source project")

            except Exception as e:
                st.error(f"Error loading project: {e}")
    else:
        st.caption("No other projects available")

with col_map:
    # Render the map
    map_result = ol_map(
        center=center,
        zoom=zoom,
        basemap=basemap,
        show_hillshade=show_hillshade,
        enable_drawing=True,
        show_toolbar=True,
        scenario_features=scenarios_geojson,
        xsection_features=xsections_geojson,
        zoom_to_bounds=bounds,
        height=650,
        key="scenario_editor_map"
    )

    # Handle map events
    if map_result:
        event_type = map_result.get('type')
        all_drawings = map_result.get('allDrawings')
        selected_layer = map_result.get('selectedFeatureLayer')

        # Store drawings in session state for save buttons
        if all_drawings:
            st.session_state['current_drawings'] = all_drawings

        # Handle feature selection
        if event_type == 'feature_selected':
            st.session_state.selected_feature = map_result.get('feature')
            st.session_state.selected_feature_layer = selected_layer
            st.rerun()
        elif event_type == 'feature_deselected':
            st.session_state.selected_feature = None
            st.session_state.selected_feature_layer = None
            st.rerun()
        elif event_type == 'delete_project_feature_requested':
            # Inform user to use sidebar to delete project features
            st.toast("Use the × button in the sidebar to delete project features")


    # Process save buttons (outside map_result check so they work on button click)
    if save_as_scenario and 'current_drawings' in st.session_state:
        drawings = st.session_state['current_drawings']
        if drawings and drawings.get('features'):
            # Filter to polygons only
            polygons = [f for f in drawings['features']
                       if f['geometry']['type'] in ['Polygon', 'MultiPolygon']]

            if polygons:
                for i, feat in enumerate(polygons):
                    name = feature_name if len(polygons) == 1 else f"{feature_name}_{i+1}"
                    feat['properties'] = feat.get('properties', {}) or {}
                    feat['properties']['name'] = name

                    save_feature_as_shapefile(
                        feat,
                        project.scenarios_dir,
                        name,
                        source_crs=25833,
                        target_crs=project.config.target_epsg
                    )

                st.success(f"Saved {len(polygons)} scenario(s)")
                # Clear drawings from session
                del st.session_state['current_drawings']
                st.rerun()
            else:
                st.warning("No polygons found in drawings. Use the polygon tool.")
        else:
            st.warning("No drawings to save. Draw features on the map first.")

    if save_as_xsection and 'current_drawings' in st.session_state:
        drawings = st.session_state['current_drawings']
        if drawings and drawings.get('features'):
            # Filter to lines only
            lines = [f for f in drawings['features']
                    if f['geometry']['type'] in ['LineString', 'MultiLineString']]

            if lines:
                for i, feat in enumerate(lines):
                    name = feature_name if len(lines) == 1 else f"{feature_name}_{i+1}"
                    feat['properties'] = feat.get('properties', {}) or {}
                    feat['properties']['name'] = name

                    save_feature_as_shapefile(
                        feat,
                        xsect_dir,
                        name,
                        source_crs=25833,
                        target_crs=project.config.target_epsg
                    )

                st.success(f"Saved {len(lines)} cross-section(s)")
                del st.session_state['current_drawings']
                st.rerun()
            else:
                st.warning("No lines found in drawings. Use the line tool.")
        else:
            st.warning("No drawings to save. Draw features on the map first.")

    # Instructions
    st.caption("""
    **Drawing Tools:** Navigate (pan/zoom) | Select | Draw Polygon | Draw Line | Edit | Delete | Clear | Fit

    **Workflow:** Draw features → Enter name → Click "Save As Scenario" or "Save As X-Section"
    """)

# =============================================================================
# Footer Stats
# =============================================================================

st.divider()

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric("Scenarios", len(scenarios_gdf))

with col_stat2:
    st.metric("Cross-sections", len(xsections_gdf))

with col_stat3:
    if not scenarios_gdf.empty:
        total_area = scenarios_gdf.geometry.area.sum() / 1e6
        st.metric("Total Area", f"{total_area:.2f} km²")
    else:
        st.metric("Total Area", "—")

with col_stat4:
    if not xsections_gdf.empty:
        total_length = xsections_gdf.geometry.length.sum() / 1000
        st.metric("Total Length", f"{total_length:.2f} km")
    else:
        st.metric("Total Length", "—")
