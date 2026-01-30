"""
OpenLayers Map Component for Streamlit
Supports EPSG:25833 (UTM33N) natively for Norwegian GIS data.

Features:
- Drawing tools (polygon, line) with toolbar
- Multiple Norwegian WMS base maps
- Bidirectional communication with Streamlit
- Feature selection and deletion
"""
import streamlit.components.v1 as components
import os
from typing import Optional, Dict, List, Any

# Development mode detection
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "ol_map",
        url="http://localhost:5173",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("ol_map", path=build_dir)


def ol_map(
    center: List[float] = [61.5, 8.5],
    zoom: int = 7,
    basemap: str = "Grunnkart",
    show_hillshade: bool = True,
    enable_drawing: bool = False,
    show_toolbar: bool = True,
    uploaded_features: Optional[Dict] = None,
    scenario_features: Optional[Dict] = None,
    xsection_features: Optional[Dict] = None,
    raster_overlay: Optional[Dict] = None,
    zoom_to_bounds: Optional[List[float]] = None,
    height: int = 600,
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Display an OpenLayers map with Norwegian projections (EPSG:25833).

    Parameters
    ----------
    center : list
        Map center as [latitude, longitude] in WGS84
    zoom : int
        Initial zoom level (3-18)
    basemap : str
        Base layer: 'Grunnkart', 'DTM 10m', 'Satellite (ESRI)'
    show_hillshade : bool
        Show semi-transparent hillshade overlay (DTM relief) on Grunnkart/Satellite
    enable_drawing : bool
        Enable drawing tools
    show_toolbar : bool
        Show the drawing toolbar (only when enable_drawing=True)
    uploaded_features : dict
        GeoJSON FeatureCollection in EPSG:25833 (legacy, for drawings layer)
    scenario_features : dict
        GeoJSON FeatureCollection of scenario polygons in EPSG:25833
    xsection_features : dict
        GeoJSON FeatureCollection of cross-section lines in EPSG:25833
    raster_overlay : dict
        Raster overlay with 'image' (base64) and 'bounds' [minx, miny, maxx, maxy]
    zoom_to_bounds : list
        Bounds to zoom to [minx, miny, maxx, maxy] in EPSG:25833
    height : int
        Map height in pixels
    key : str
        Streamlit component key

    Returns
    -------
    dict or None
        Component state with event type and feature data:
        - type: 'feature_drawn', 'feature_selected', 'feature_deleted', etc.
        - feature: GeoJSON feature (if applicable)
        - allDrawings: All drawings as GeoJSON FeatureCollection
    """
    component_value = _component_func(
        center=center,
        zoom=zoom,
        basemap=basemap,
        showHillshade=show_hillshade,
        enableDrawing=enable_drawing,
        showToolbar=show_toolbar,
        uploadedFeatures=uploaded_features,
        scenarioFeatures=scenario_features,
        xsectionFeatures=xsection_features,
        rasterOverlay=raster_overlay,
        zoomToBounds=zoom_to_bounds,
        height=height,
        key=key,
        default=None
    )

    return component_value
