"""
Utility functions for the OpenLayers map component.
Handles GeoJSON/Shapefile conversion and project data loading.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json
import shutil


def load_project_scenarios(project) -> gpd.GeoDataFrame:
    """
    Load all scenario polygons from a project.

    Parameters
    ----------
    project : Project
        Project instance with scenarios_dir property

    Returns
    -------
    GeoDataFrame
        Combined scenarios in EPSG:25833
    """
    TARGET_CRS = 25833
    gdfs = []

    if project.scenarios_dir.exists():
        for shp in project.scenarios_dir.glob("*.shp"):
            try:
                gdf = gpd.read_file(shp)
                gdf['name'] = shp.stem
                gdf['source'] = 'project'
                # Transform to target CRS before appending to avoid concat CRS conflicts
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=TARGET_CRS)
                elif gdf.crs.to_epsg() != TARGET_CRS:
                    gdf = gdf.to_crs(epsg=TARGET_CRS)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Warning: Could not load {shp.name}: {e}")

    if gdfs:
        combined = pd.concat(gdfs, ignore_index=True)
        return combined

    return gpd.GeoDataFrame(
        columns=['geometry', 'name', 'source'],
        crs=f'EPSG:{TARGET_CRS}'
    )


def load_project_xsections(project) -> gpd.GeoDataFrame:
    """
    Load all cross-section lines from a project.

    Parameters
    ----------
    project : Project
        Project instance with inputs_dir property

    Returns
    -------
    GeoDataFrame
        Combined cross-sections in EPSG:25833
    """
    TARGET_CRS = 25833
    gdfs = []
    xsect_dir = project.inputs_dir / "xsection_lines"

    if xsect_dir.exists():
        for shp in xsect_dir.glob("*.shp"):
            try:
                gdf = gpd.read_file(shp)
                gdf['name'] = shp.stem
                gdf['source'] = 'project'
                # Transform to target CRS before appending to avoid concat CRS conflicts
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=TARGET_CRS)
                elif gdf.crs.to_epsg() != TARGET_CRS:
                    gdf = gdf.to_crs(epsg=TARGET_CRS)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Warning: Could not load {shp.name}: {e}")

    if gdfs:
        combined = pd.concat(gdfs, ignore_index=True)
        return combined

    return gpd.GeoDataFrame(
        columns=['geometry', 'name', 'source'],
        crs=f'EPSG:{TARGET_CRS}'
    )


def geodataframe_to_geojson(gdf: gpd.GeoDataFrame, target_crs: int = 25833) -> Optional[Dict]:
    """
    Convert GeoDataFrame to GeoJSON dict.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    target_crs : int
        Target EPSG code (default 25833 for UTM33N)

    Returns
    -------
    dict or None
        GeoJSON FeatureCollection or None if empty
    """
    if gdf is None or gdf.empty:
        return None

    # Ensure correct CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=target_crs)
    elif gdf.crs.to_epsg() != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)

    return json.loads(gdf.to_json())


def geojson_to_geodataframe(
    geojson: Dict,
    source_crs: int = 25833,
    target_crs: int = 25833
) -> gpd.GeoDataFrame:
    """
    Convert GeoJSON dict to GeoDataFrame.

    Parameters
    ----------
    geojson : dict
        GeoJSON FeatureCollection
    source_crs : int
        Source EPSG code
    target_crs : int
        Target EPSG code

    Returns
    -------
    GeoDataFrame
        Converted GeoDataFrame
    """
    if not geojson or 'features' not in geojson or not geojson['features']:
        return gpd.GeoDataFrame(
            columns=['geometry', 'name'],
            crs=f'EPSG:{target_crs}'
        )

    gdf = gpd.GeoDataFrame.from_features(geojson['features'], crs=f'EPSG:{source_crs}')

    if source_crs != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)

    return gdf


def save_feature_as_shapefile(
    geojson_feature: Dict,
    output_dir: Path,
    name: str,
    source_crs: int = 25833,
    target_crs: int = 25833
) -> Path:
    """
    Save a single GeoJSON feature as a shapefile.

    Parameters
    ----------
    geojson_feature : dict
        Single GeoJSON feature
    output_dir : Path
        Directory to save shapefile
    name : str
        Filename (without extension)
    source_crs : int
        Source EPSG code
    target_crs : int
        Target EPSG code for output

    Returns
    -------
    Path
        Path to saved shapefile
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create GeoDataFrame from single feature
    gdf = gpd.GeoDataFrame.from_features([geojson_feature], crs=f'EPSG:{source_crs}')

    # Add name attribute
    gdf['name'] = name

    # Drop columns that would be truncated by shapefile format (10 char limit)
    for col in ['featureType']:
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])

    # Reproject if needed
    if source_crs != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)

    # Clean filename
    clean_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)

    # Save
    output_path = output_dir / f"{clean_name}.shp"
    gdf.to_file(output_path)

    return output_path


def save_geojson_as_shapefiles(
    geojson: Dict,
    output_dir: Path,
    source_crs: int = 25833,
    target_crs: int = 25833
) -> List[Path]:
    """
    Save GeoJSON FeatureCollection as individual shapefiles.

    Parameters
    ----------
    geojson : dict
        GeoJSON FeatureCollection
    output_dir : Path
        Output directory
    source_crs : int
        Source EPSG code
    target_crs : int
        Target EPSG code

    Returns
    -------
    list
        List of saved shapefile paths
    """
    if not geojson or 'features' not in geojson:
        return []

    saved_paths = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, feature in enumerate(geojson['features']):
        # Get name from properties or generate one
        props = feature.get('properties', {}) or {}
        name = props.get('name', f'feature_{i+1}')

        path = save_feature_as_shapefile(
            feature, output_dir, name, source_crs, target_crs
        )
        saved_paths.append(path)

    return saved_paths


def delete_shapefile(shp_path: Path) -> bool:
    """
    Delete a shapefile and all its associated files.

    Parameters
    ----------
    shp_path : Path
        Path to .shp file

    Returns
    -------
    bool
        True if deleted successfully
    """
    shp_path = Path(shp_path)

    try:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
            f = shp_path.with_suffix(ext)
            if f.exists():
                f.unlink()
        return True
    except Exception as e:
        print(f"Error deleting shapefile: {e}")
        return False


def get_project_bounds(project, crs: int = 25833) -> Optional[List[float]]:
    """
    Get combined bounds of all project features.

    Parameters
    ----------
    project : Project
        Project instance
    crs : int
        Target EPSG code for bounds

    Returns
    -------
    list or None
        [minx, miny, maxx, maxy] or None if no features
    """
    scenarios = load_project_scenarios(project)
    xsections = load_project_xsections(project)

    gdfs = [gdf for gdf in [scenarios, xsections] if not gdf.empty]

    if not gdfs:
        return None

    combined = pd.concat(gdfs, ignore_index=True)

    if combined.crs.to_epsg() != crs:
        combined = combined.to_crs(epsg=crs)

    bounds = combined.total_bounds
    return bounds.tolist()


def get_project_center(project) -> Optional[Tuple[float, float]]:
    """
    Get center point of project features as (lat, lon) in WGS84.

    Parameters
    ----------
    project : Project
        Project instance

    Returns
    -------
    tuple or None
        (latitude, longitude) or None if no features
    """
    bounds = get_project_bounds(project, crs=25833)
    if bounds is None:
        return None

    # Get center in UTM
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Convert to WGS84
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)

    return (lat, lon)


def calculate_polygon_area(geojson_feature: Dict, source_crs: int = 25833) -> float:
    """
    Calculate area of a polygon feature in km².

    Parameters
    ----------
    geojson_feature : dict
        GeoJSON feature
    source_crs : int
        EPSG code of the feature coordinates

    Returns
    -------
    float
        Area in km²
    """
    gdf = gpd.GeoDataFrame.from_features([geojson_feature], crs=f'EPSG:{source_crs}')

    # For accurate area, use UTM
    if source_crs != 25833:
        gdf = gdf.to_crs(epsg=25833)

    return gdf.geometry.area.iloc[0] / 1e6


def calculate_line_length(geojson_feature: Dict, source_crs: int = 25833) -> float:
    """
    Calculate length of a line feature in km.

    Parameters
    ----------
    geojson_feature : dict
        GeoJSON feature
    source_crs : int
        EPSG code of the feature coordinates

    Returns
    -------
    float
        Length in km
    """
    gdf = gpd.GeoDataFrame.from_features([geojson_feature], crs=f'EPSG:{source_crs}')

    if source_crs != 25833:
        gdf = gdf.to_crs(epsg=25833)

    return gdf.geometry.length.iloc[0] / 1000
