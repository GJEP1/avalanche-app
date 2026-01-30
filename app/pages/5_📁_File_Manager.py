"""
File Manager Page
=================
View and manage project files, simulations, SLBL results, and disk space.
"""

import streamlit as st
import shutil
from pathlib import Path
from datetime import datetime
import sys
import json
import zipfile
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auth import require_authentication, show_user_info_sidebar
from core.project_manager import list_projects, Project, project_selector_sidebar

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="File Manager",
    page_icon="📁",
    layout="wide"
)

authenticated, username, name = require_authentication()
if not authenticated:
    st.stop()

show_user_info_sidebar()
project = project_selector_sidebar()
st.title("File Manager")

DATA_ROOT = Path("/mnt/data")
PROJECTS_ROOT = DATA_ROOT / "projects"
JOB_QUEUE_ROOT = DATA_ROOT / "job_queue"

# =============================================================================
# Utility functions
# =============================================================================

def get_disk_usage(path: Path = DATA_ROOT):
    try:
        total, used, free = shutil.disk_usage(path)
        return total, used, free, used / total * 100
    except Exception:
        return 0, 0, 0, 0.0


def get_directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    size = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                size += p.stat().st_size
            except Exception:
                pass
    return size


def get_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def format_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def get_modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def delete_path(path: Path):
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception:
        return False


def list_immediate_subdirs(path: Path):
    if not path.exists():
        return []
    return [
        {"path": p, "name": p.name}
        for p in path.iterdir()
        if p.is_dir()
    ]


def create_zip_from_files(files: list, base_name: str) -> bytes:
    """Create a zip file from a list of file paths."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if isinstance(f, Path) and f.exists():
                zf.write(f, f.name)
    buffer.seek(0)
    return buffer.getvalue()


def create_zip_from_shapefile(shp_path: Path) -> bytes:
    """Create a zip file containing all shapefile components."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            f = shp_path.with_suffix(ext)
            if f.exists():
                zf.write(f, f.name)
    buffer.seek(0)
    return buffer.getvalue()


def create_zip_from_directory(dir_path: Path) -> bytes:
    """Create a zip file from a directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in dir_path.rglob('*'):
            if f.is_file():
                arcname = f.relative_to(dir_path.parent)
                zf.write(f, arcname)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# Checkbox helper (core fix)
# =============================================================================

def checkbox_list(title, items, key_prefix, label_func=lambda x: x.name):
    if not items:
        return []

    state_key = f"{key_prefix}_state"
    # Reset state if number of items changed
    if state_key not in st.session_state or len(st.session_state[state_key]) != len(items):
        st.session_state[state_key] = {i: False for i in range(len(items))}

    st.markdown(f"**{title}**")

    selected = []
    for i, item in enumerate(items):
        checked = st.checkbox(
            label_func(item),
            key=f"{key_prefix}_{i}",
            value=st.session_state[state_key].get(i, False),
        )
        st.session_state[state_key][i] = checked
        if checked:
            selected.append(item)

    st.caption(f"Selected: {len(selected)} of {len(items)}")
    return selected


# =============================================================================
# Storage overview
# =============================================================================

total, used, free, pct = get_disk_usage()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", format_bytes(total))
c2.metric("Used", format_bytes(used))
c3.metric("Available", format_bytes(free))
c4.metric("Usage", f"{pct:.1f}%")
st.progress(min(pct / 100, 1.0))

st.divider()

# =============================================================================
# Tabs
# =============================================================================

tab_projects, tab_jobs = st.tabs(["Projects", "Job Queue"])

# =============================================================================
# PROJECTS
# =============================================================================

with tab_projects:
    if project is None:
        st.info("No project selected. Please select or create a project in the sidebar.")
        st.stop()

    proj = project
    project_size = get_directory_size(proj.root)
    st.caption(f"Project: **{proj.name}** ({format_bytes(project_size)})")

    # -------------------------------------------------------------------------
    # SLBL
    # -------------------------------------------------------------------------
    with st.expander("SLBL Results"):
        if proj.slbl_dir.exists():
            thickness = sorted(proj.slbl_dir.glob("*_thickness.tif"))
            selected = checkbox_list(
                "Thickness rasters",
                thickness,
                "slbl_thick",
                lambda f: f.stem.replace("_thickness", "")
            )
            if selected:
                col_del, col_exp = st.columns(2)
                with col_del:
                    if st.button("Delete selected", key="del_slbl"):
                        for f in selected:
                            delete_path(f)
                            base = f.with_name(f.name.replace("_thickness.tif", "_base.tif"))
                            if base.exists():
                                delete_path(base)
                        st.rerun()
                with col_exp:
                    # Collect thickness and base files
                    export_files = []
                    for f in selected:
                        export_files.append(f)
                        base = f.with_name(f.name.replace("_thickness.tif", "_base.tif"))
                        if base.exists():
                            export_files.append(base)
                    zip_data = create_zip_from_files(export_files, "slbl_export")
                    st.download_button(
                        "Export selected",
                        data=zip_data,
                        file_name="slbl_export.zip",
                        mime="application/zip",
                        key="exp_slbl"
                    )

    # -------------------------------------------------------------------------
    # Simulations
    # -------------------------------------------------------------------------
    with st.expander("Simulations"):
        if proj.simulations_dir.exists():
            dirs = list_immediate_subdirs(proj.simulations_dir)
            sweeps = [d["path"] for d in dirs if d["name"].startswith("sweep_")]
            singles = [d["path"] for d in dirs if not d["name"].startswith("sweep_")]

            selected_sweeps = checkbox_list("Parameter sweeps", sweeps, "sim_sweep")
            if selected_sweeps:
                col_del, col_exp = st.columns(2)
                with col_del:
                    if st.button("Delete selected", key="del_sweeps"):
                        for p in selected_sweeps:
                            delete_path(p)
                        st.rerun()
                with col_exp:
                    if len(selected_sweeps) == 1:
                        zip_data = create_zip_from_directory(selected_sweeps[0])
                        st.download_button(
                            "Export selected",
                            data=zip_data,
                            file_name=f"{selected_sweeps[0].name}.zip",
                            mime="application/zip",
                            key="exp_sweeps"
                        )
                    else:
                        st.caption("Select 1 sweep to export")

            st.divider()

            selected_singles = checkbox_list("Single simulations", singles[:15], "sim_single")
            if selected_singles:
                col_del, col_exp = st.columns(2)
                with col_del:
                    if st.button("Delete selected", key="del_singles"):
                        for p in selected_singles:
                            delete_path(p)
                        st.rerun()
                with col_exp:
                    if len(selected_singles) == 1:
                        zip_data = create_zip_from_directory(selected_singles[0])
                        st.download_button(
                            "Export selected",
                            data=zip_data,
                            file_name=f"{selected_singles[0].name}.zip",
                            mime="application/zip",
                            key="exp_singles"
                        )
                    else:
                        st.caption("Select 1 sim to export")

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    with st.expander("Inputs"):
        # Scenario polygons
        shp_files = sorted(proj.scenarios_dir.glob("*.shp"))
        selected_sc = checkbox_list("Scenario polygons", shp_files, "input_sc", lambda p: p.stem)
        if selected_sc:
            col_del, col_exp = st.columns(2)
            with col_del:
                if st.button("Delete selected", key="del_scenarios"):
                    for shp in selected_sc:
                        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                            f = shp.with_suffix(ext)
                            if f.exists():
                                f.unlink()
                    st.rerun()
            with col_exp:
                # Export all selected shapefiles
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for shp in selected_sc:
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            f = shp.with_suffix(ext)
                            if f.exists():
                                zf.write(f, f.name)
                buffer.seek(0)
                st.download_button(
                    "Export selected",
                    data=buffer.getvalue(),
                    file_name="scenarios_export.zip",
                    mime="application/zip",
                    key="exp_scenarios"
                )

        st.divider()

        # Cross-section lines
        xsect_dir = proj.inputs_dir / "xsection_lines"
        xsect_files = sorted(xsect_dir.glob("*.shp")) if xsect_dir.exists() else []
        selected_xsect = checkbox_list("Cross-section lines", xsect_files, "input_xsect", lambda p: p.stem)
        if selected_xsect:
            col_del, col_exp = st.columns(2)
            with col_del:
                if st.button("Delete selected", key="del_xsections"):
                    for shp in selected_xsect:
                        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                            f = shp.with_suffix(ext)
                            if f.exists():
                                f.unlink()
                    st.rerun()
            with col_exp:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for shp in selected_xsect:
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            f = shp.with_suffix(ext)
                            if f.exists():
                                zf.write(f, f.name)
                buffer.seek(0)
                st.download_button(
                    "Export selected",
                    data=buffer.getvalue(),
                    file_name="xsections_export.zip",
                    mime="application/zip",
                    key="exp_xsections"
                )

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    with st.expander("Analysis"):
        if proj.analysis_dir.exists():
            files = sorted([f for f in proj.analysis_dir.iterdir() if f.is_file()])
            selected = checkbox_list("Analysis files", files, "analysis", lambda f: f.name)
            if selected:
                col_del, col_exp = st.columns(2)
                with col_del:
                    if st.button("Delete selected", key="del_analysis"):
                        for f in selected:
                            delete_path(f)
                        st.rerun()
                with col_exp:
                    zip_data = create_zip_from_files(selected, "analysis_export")
                    st.download_button(
                        "Export selected",
                        data=zip_data,
                        file_name="analysis_export.zip",
                        mime="application/zip",
                        key="exp_analysis"
                    )

    # -------------------------------------------------------------------------
    # Probability Ensembles
    # -------------------------------------------------------------------------
    with st.expander("Probability Ensembles"):
        if proj.probability_dir.exists():
            ensembles = list_immediate_subdirs(proj.probability_dir)
            ensemble_paths = [e["path"] for e in ensembles]

            selected_ensembles = checkbox_list(
                "Ensemble runs",
                ensemble_paths,
                "prob_ensembles"
            )
            if selected_ensembles:
                col_del, col_exp = st.columns(2)
                with col_del:
                    if st.button("Delete selected", key="del_ensembles"):
                        for p in selected_ensembles:
                            delete_path(p)
                        st.rerun()
                with col_exp:
                    if len(selected_ensembles) == 1:
                        zip_data = create_zip_from_directory(selected_ensembles[0])
                        st.download_button(
                            "Export selected",
                            data=zip_data,
                            file_name=f"{selected_ensembles[0].name}.zip",
                            mime="application/zip",
                            key="exp_ensembles"
                        )
                    else:
                        st.caption("Select 1 ensemble to export")
        else:
            st.caption("No probability ensembles yet")

    st.divider()

    if st.button("Delete project", type="primary"):
        delete_path(proj.root)
        st.rerun()

# =============================================================================
# JOB QUEUE
# =============================================================================

with tab_jobs:
    if not JOB_QUEUE_ROOT.exists():
        st.info("Job queue directory does not exist.")
        st.stop()

    jobs = list(JOB_QUEUE_ROOT.glob("*.json"))
    if not jobs:
        st.info("No jobs in queue.")
        st.stop()

    selected_jobs = checkbox_list(
        "Jobs",
        jobs,
        "jobs",
        lambda f: f.name,
    )

    if selected_jobs and st.button("Delete selected jobs"):
        for j in selected_jobs:
            delete_path(j)
        st.rerun()

# =============================================================================
# Footer
# =============================================================================
st.divider()
if st.button("Refresh"):
    st.rerun()
