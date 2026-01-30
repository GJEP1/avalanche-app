"""
Project management for organizing SLBL, simulation, and analysis outputs.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import streamlit as st


PROJECTS_ROOT = Path("/mnt/data/projects")


@dataclass
class ProjectConfig:
    """Configuration for a project."""
    name: str
    created: str
    created_by: str
    description: str = ""
    dem_path: str = "/mnt/data/dem/dtm2020_final_COG.tif"
    target_epsg: int = 25833
    
    # SLBL settings
    slbl_e_ratios: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
    slbl_max_depths: List[Optional[float]] = field(default_factory=lambda: [None])
    slbl_neighbours: int = 4
    
    # Simulation settings
    sim_mu_range: List[float] = field(default_factory=lambda: [0.025, 0.05, 0.075, 0.1])
    sim_xi_range: List[float] = field(default_factory=lambda: [500, 750, 1000, 1500])


@dataclass
class Project:
    """Represents a project with all its paths and configuration."""
    name: str
    root: Path
    config: ProjectConfig
    
    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"
    
    @property
    def slbl_dir(self) -> Path:
        return self.root / "slbl"
    
    @property
    def simulations_dir(self) -> Path:
        return self.root / "simulations"
    
    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def probability_dir(self) -> Path:
        """Directory containing probability ensemble outputs."""
        return self.root / "probability"

    @property
    def scenarios_dir(self) -> Path:
        """Directory containing scenario polygon shapefiles."""
        return self.inputs_dir / "scenarios"

    def ensure_directories(self):
        """Create all project directories."""
        for d in [self.inputs_dir, self.slbl_dir, self.simulations_dir,
                  self.analysis_dir, self.probability_dir, self.scenarios_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_config(self):
        """Save project configuration."""
        config_path = self.root / "project_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    @classmethod
    def load(cls, name: str) -> 'Project':
        """Load an existing project."""
        root = PROJECTS_ROOT / name
        config_path = root / "project_config.json"
        
        if not config_path.exists():
            raise ValueError(f"Project '{name}' not found")
        
        with open(config_path) as f:
            config_data = json.load(f)
        
        config = ProjectConfig(**config_data)
        return cls(name=name, root=root, config=config)
    
    @classmethod
    def create(cls, name: str, created_by: str, description: str = "", 
               dem_path: str = None) -> 'Project':
        """Create a new project."""
        root = PROJECTS_ROOT / name
        
        if root.exists():
            raise ValueError(f"Project '{name}' already exists")
        
        config = ProjectConfig(
            name=name,
            created=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            dem_path=dem_path or "/mnt/data/dem/dtm2020_final_COG.tif"
        )
        
        project = cls(name=name, root=root, config=config)
        project.ensure_directories()
        project.save_config()
        
        return project


def list_projects() -> List[str]:
    """List all available projects."""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted([
        d.name for d in PROJECTS_ROOT.iterdir() 
        if d.is_dir() and (d / "project_config.json").exists()
    ])


def get_current_project() -> Optional[Project]:
    """Get the currently selected project from session state."""
    project_name = st.session_state.get('current_project')
    if project_name:
        try:
            return Project.load(project_name)
        except Exception:
            return None
    return None


def set_current_project(project_name: str):
    """Set the current project in session state and query params for persistence."""
    st.session_state['current_project'] = project_name
    # Also store in query params so it survives page refresh
    st.query_params['project'] = project_name


def copy_scenarios_between_projects(
    source_project: Project,
    target_project: Project,
    scenario_names: List[str]
) -> List[str]:
    """
    Copy scenario shapefiles from one project to another.

    Parameters
    ----------
    source_project : Project
        Source project to copy from
    target_project : Project
        Target project to copy to
    scenario_names : list
        List of scenario names to copy

    Returns
    -------
    list
        List of successfully copied scenario names
    """
    copied = []
    target_project.scenarios_dir.mkdir(parents=True, exist_ok=True)

    for name in scenario_names:
        try:
            # Copy all shapefile components
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = source_project.scenarios_dir / f"{name}{ext}"
                if src.exists():
                    dst = target_project.scenarios_dir / f"{name}{ext}"
                    shutil.copy(src, dst)
            copied.append(name)
        except Exception as e:
            print(f"Error copying scenario {name}: {e}")

    return copied


def copy_xsections_between_projects(
    source_project: Project,
    target_project: Project,
    xsection_names: List[str]
) -> List[str]:
    """
    Copy cross-section shapefiles from one project to another.

    Parameters
    ----------
    source_project : Project
        Source project to copy from
    target_project : Project
        Target project to copy to
    xsection_names : list
        List of cross-section names to copy

    Returns
    -------
    list
        List of successfully copied cross-section names
    """
    copied = []
    src_dir = source_project.inputs_dir / "xsection_lines"
    dst_dir = target_project.inputs_dir / "xsection_lines"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in xsection_names:
        try:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = src_dir / f"{name}{ext}"
                if src.exists():
                    dst = dst_dir / f"{name}{ext}"
                    shutil.copy(src, dst)
            copied.append(name)
        except Exception as e:
            print(f"Error copying xsection {name}: {e}")

    return copied


def get_project_scenarios(project: Project) -> List[str]:
    """Get list of scenario names in a project."""
    if not project.scenarios_dir.exists():
        return []
    return sorted([
        f.stem for f in project.scenarios_dir.glob("*.shp")
    ])


def get_project_xsections(project: Project) -> List[str]:
    """Get list of cross-section names in a project."""
    xsect_dir = project.inputs_dir / "xsection_lines"
    if not xsect_dir.exists():
        return []
    return sorted([
        f.stem for f in xsect_dir.glob("*.shp")
    ])


def project_selector_sidebar() -> Optional[Project]:
    """
    Display project selector in sidebar and return selected project.
    Persists selection across page refresh using query params.
    """
    with st.sidebar:
        st.subheader("📁 Project")

        projects = list_projects()

        # Project selection
        if projects:
            # Check query params first (for persistence across refresh), then session state
            query_project = st.query_params.get('project')
            session_project = st.session_state.get('current_project')

            # Determine current project: query params > session state > first project
            if query_project and query_project in projects:
                current = query_project
                # Sync to session state if not already set
                if session_project != current:
                    st.session_state['current_project'] = current
            elif session_project and session_project in projects:
                current = session_project
                # Sync query params from session state (for when navigating via sidebar links)
                if query_project != current:
                    st.query_params['project'] = current
            else:
                current = projects[0]
                # Set initial project in both places
                st.session_state['current_project'] = current
                st.query_params['project'] = current

            selected = st.selectbox(
                "Select project:",
                projects,
                index=projects.index(current) if current in projects else 0,
                key="project_selector"
            )

            if selected != st.session_state.get('current_project'):
                set_current_project(selected)
                st.rerun()
        else:
            st.info("No projects yet. Create one below.")
            selected = None

        # Create new project
        with st.expander("➕ New Project", expanded=not projects):
            new_name = st.text_input("Project name:", key="new_project_name")
            new_desc = st.text_area("Description:", key="new_project_desc", height=68)

            if st.button("Create Project", type="primary", disabled=not new_name):
                try:
                    username = st.session_state.get('username', 'unknown')
                    project = Project.create(new_name, username, new_desc)
                    set_current_project(new_name)
                    st.success(f"Created project: {new_name}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.divider()

    return get_current_project()
