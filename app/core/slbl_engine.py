"""
SLBL (Sloping Local Base Level) Engine for Web Application.

Adapted from batch processing script for use in Streamlit web application.
Supports both CPU and GPU processing (GPU optional).

Reference:
Jaboyedoff, M., et al. (2019). "Testing a failure surface prediction and 
deposit reconstruction method for a landslide cluster that occurred during 
Typhoon Talas (Japan)". Earth Surf. Dynam., 7, 439–458.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List, Tuple, Dict, Any
from datetime import datetime
import json
import traceback

# Thread caps for multiprocessing compatibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Try to import scipy for faster convolution
try:
    from scipy.ndimage import convolve, binary_dilation
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[SLBL] scipy not available, using fallback implementations")

# GPU support (optional - can be enabled later)
HAVE_GPU = False
cp = None
cpx_nd = None

def init_gpu() -> bool:
    """
    Initialize GPU backend if available.
    Returns True if GPU is available and initialized.
    """
    global HAVE_GPU, cp, cpx_nd
    
    try:
        import cupy as _cp
        import cupyx.scipy.ndimage as _cpx_nd
        
        if _cp.cuda.runtime.getDeviceCount() > 0:
            props = _cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            print(f"[SLBL-GPU] Detected device: {name}")
            HAVE_GPU = True
            cp = _cp
            cpx_nd = _cpx_nd
            return True
    except Exception as e:
        print(f"[SLBL-GPU] Not available: {e}")
    
    return False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SLBLConfig:
    """Configuration for SLBL processing."""
    
    # Mode: "failure" (excavation) or "inverse" (filling)
    mode: str = "failure"
    
    # E-ratio sweep (e = z_max / L_rh)
    # Typical values: 0.05 to 0.25
    e_ratios: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
    
    # Optional max depth constraint (meters)
    max_depths: List[Optional[float]] = field(default_factory=lambda: [None])
    
    # Convergence parameters
    stop_eps: float = 1e-7
    max_iters: int = 3000
    
    # Neighbor rule: 4 (von Neumann) or 8 (Moore)
    neighbours: int = 4
    
    # Z-floor constraint
    use_z_floor: bool = True
    z_floor_buffer_m: float = 0
    
    # Polygon processing
    buffer_pixels: int = 4
    all_touched: bool = True
    
    # Processing
    use_gpu: bool = False
    
    # Output options
    write_geotiff: bool = True
    write_ascii: bool = False  # For AvaFrame compatibility
    
    # Cross-section options
    write_xsections: bool = True
    xsect_lines_source: List[str] = field(default_factory=list)
    xsect_step_m: float = 1.0  # Sampling step along lines
    xsect_clip_to_poly: bool = False  # Clip to polygon boundary
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SLBLConfig':
        return cls(**data)


@dataclass
class SLBLResult:
    """Result from a single SLBL computation."""
    scenario_name: str
    label: str
    e_ratio: float
    max_depth_limit: Optional[float]
    tolerance_C: float

    # Computed metrics
    volume_m3: float = 0.0
    max_depth_m: float = 0.0
    mean_depth_m: float = 0.0
    area_m2: float = 0.0

    # Geometry info
    L_rh_m: float = 0.0
    width_m: float = 0.0

    # Output paths
    thickness_path: str = ""
    base_path: str = ""

    # Status
    success: bool = True
    error: Optional[str] = None

    # Job tracking
    job_id: Optional[str] = None
    job_created_by: Optional[str] = None
    job_created: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Geometry Utilities
# =============================================================================

def compute_poly_dimensions(geom) -> Tuple[float, float]:
    """
    Compute landslide dimensions (L_rh, w) from oriented bounding rectangle.
    
    Returns:
        L_rh: Horizontal length (longest axis)
        w: Width (shortest axis)
    """
    try:
        mbr = geom.minimum_rotated_rectangle
        if mbr.is_empty:
            return np.nan, np.nan
        
        coords = np.asarray(mbr.exterior.coords)
        if coords.shape[0] < 5:
            return np.nan, np.nan
        
        edges = np.sqrt(np.sum(np.diff(coords[:4], axis=0)**2, axis=1))
        sorted_edges = np.sort(edges)[::-1]
        
        return float(sorted_edges[0]), float(sorted_edges[1])
    except Exception:
        return np.nan, np.nan


def compute_tolerance_C(e_ratio: float, L_rh: float, dx: float) -> float:
    """
    Compute SLBL tolerance C from e-ratio.
    
    C = (4 × e / L_rh) × Δx²
    
    Args:
        e_ratio: The e parameter (z_max / L_rh)
        L_rh: Horizontal length of the landslide (meters)
        dx: DEM pixel size (meters)
    """
    if not np.isfinite(e_ratio) or not np.isfinite(L_rh) or L_rh <= 0 or dx <= 0:
        return np.nan
    
    return (4.0 * e_ratio / L_rh) * (dx * dx)


# =============================================================================
# Raster Operations
# =============================================================================

def read_dem_window(dem_path: str, bounds: Tuple[float, float, float, float],
                    buffer_m: float = 500) -> Tuple[np.ma.MaskedArray, dict, rasterio.Affine]:
    """
    Read a windowed portion of a DEM.
    
    Args:
        dem_path: Path to DEM file
        bounds: (minx, miny, maxx, maxy) in DEM CRS
        buffer_m: Buffer around bounds in meters
    
    Returns:
        dem_data, profile, transform
    """
    with rasterio.open(dem_path) as src:
        # Add buffer to bounds
        buffered_bounds = (
            bounds[0] - buffer_m,
            bounds[1] - buffer_m,
            bounds[2] + buffer_m,
            bounds[3] + buffer_m
        )
        
        # Calculate window
        window = rasterio.windows.from_bounds(*buffered_bounds, src.transform)
        
        # Ensure window is within raster bounds
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        
        # Read data
        data = src.read(1, window=window, masked=True).astype('float64')
        transform = src.window_transform(window)
        
        profile = src.profile.copy()
        profile.update(
            width=data.shape[1],
            height=data.shape[0],
            transform=transform
        )
        
        return data, profile, transform


def rasterize_polygon(gdf: gpd.GeoDataFrame, shape: Tuple[int, int],
                      transform: rasterio.Affine, all_touched: bool = True) -> np.ndarray:
    """Rasterize polygon to boolean mask."""
    shapes = ((geom, 1) for geom in gdf.geometry if geom and not geom.is_empty)
    
    mask = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype='uint8'
    ).astype(bool)
    
    return mask


def compute_z_floor(inmask: np.ndarray, dem: np.ma.MaskedArray) -> float:
    """Compute z-floor as minimum elevation around polygon boundary."""
    if not HAVE_SCIPY:
        return np.nan
    
    try:
        ring = binary_dilation(inmask, iterations=1) & ~inmask
        arr = dem.filled(np.nan)
        if np.any(ring):
            return float(np.nanmin(arr[ring]))
        return np.nan
    except Exception:
        return np.nan


def save_geotiff(path: str, arr: np.ndarray, profile: dict, nodata=np.nan):
    """Save array as GeoTIFF with compression."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    prof = profile.copy()
    prof.update(
        dtype='float32',
        count=1,
        compress='deflate',
        predictor=3,
        nodata=nodata
    )
    
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr.astype('float32'), 1)


def save_ascii(path: str, arr: np.ndarray, transform: rasterio.Affine, 
               nodata: float = -9999):
    """Save as ESRI ASCII Grid for AvaFrame compatibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    nrows, ncols = arr.shape
    xllcorner = transform.c
    yllcorner = transform.f + (nrows * transform.e)
    cellsize = abs(transform.a)
    
    arr_out = arr.copy()
    arr_out[~np.isfinite(arr_out)] = nodata
    
    with open(path, 'w') as f:
        f.write(f"ncols        {ncols}\n")
        f.write(f"nrows        {nrows}\n")
        f.write(f"xllcorner    {xllcorner:.6f}\n")
        f.write(f"yllcorner    {yllcorner:.6f}\n")
        f.write(f"cellsize     {cellsize:.6f}\n")
        f.write(f"NODATA_value {nodata}\n")
        
        for row in arr_out:
            f.write(' '.join(f'{val:.6f}' for val in row) + '\n')


# =============================================================================
# SLBL Core Algorithm
# =============================================================================

def _neighbor_mean_cpu(arr: np.ndarray, valid: np.ndarray, mode8: bool = False) -> np.ndarray:
    """Compute mean of neighbors (CPU version)."""
    if not HAVE_SCIPY:
        return _neighbor_mean_fallback(arr, valid, mode8)
    
    if mode8:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    else:
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    
    a = np.where(valid & np.isfinite(arr), arr, 0.0).astype(np.float32)
    m = (valid & np.isfinite(arr)).astype(np.float32)
    
    s = convolve(a, kernel, mode='constant', cval=0.0)
    c = convolve(m, kernel, mode='constant', cval=0.0)
    
    out = np.divide(s, c, out=np.full_like(s, np.nan), where=(c > 0))
    return out.astype(np.float64)


def _neighbor_mean_fallback(arr: np.ndarray, valid: np.ndarray, mode8: bool = False) -> np.ndarray:
    """Fallback neighbor mean without scipy."""
    H, W = arr.shape
    s = np.zeros((H, W), dtype='float64')
    c = np.zeros((H, W), dtype='float64')
    
    shifts4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shifts8 = shifts4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    shifts = shifts8 if mode8 else shifts4
    
    for dy, dx in shifts:
        y0s, y1s = max(0, dy), H + min(0, dy)
        x0s, x1s = max(0, dx), W + min(0, dx)
        y0d, y1d = max(0, -dy), H + min(0, -dy)
        x0d, x1d = max(0, -dx), W + min(0, -dx)
        
        if y0s >= y1s or x0s >= x1s:
            continue
        
        neigh = arr[y0s:y1s, x0s:x1s]
        vmask = valid[y0s:y1s, x0s:x1s] & np.isfinite(neigh)
        s[y0d:y1d, x0d:x1d] += np.where(vmask, neigh, 0.0)
        c[y0d:y1d, x0d:x1d] += vmask.astype('float64')
    
    return np.divide(s, c, out=np.full_like(s, np.nan), where=(c > 0))


def slbl_iterate(dem: np.ma.MaskedArray, inmask: np.ndarray, tol_C: float,
                 config: SLBLConfig, pixel_area: float,
                 progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    SLBL iterative solver.
    
    Args:
        dem: Input DEM (masked array)
        inmask: Boolean mask of the landslide polygon
        tol_C: SLBL tolerance parameter C
        config: SLBL configuration
        pixel_area: Area per pixel for volume calculations
        progress_callback: Optional callback(iteration, max_iters, delta)
    
    Returns:
        base: The computed SLBL surface
        thickness: Difference between DEM and SLBL
    """
    dem_np = dem.filled(np.nan)
    valid_np = np.isfinite(dem_np)
    work_mask = inmask & valid_np
    
    if not np.any(work_mask):
        return dem_np.copy(), np.zeros_like(dem_np)
    
    mode8 = (config.neighbours == 8)
    is_failure = (config.mode.lower() == "failure")
    
    base = dem_np.copy()
    
    for itr in range(config.max_iters):
        # Compute neighbor mean
        mean_nb = _neighbor_mean_cpu(base, valid_np, mode8)
        
        # Compute target surface
        if is_failure:
            target = mean_nb - tol_C
        else:
            target = mean_nb + tol_C
        
        # Update where appropriate
        if is_failure:
            candidate = np.where(work_mask & ~np.isnan(target), target, base)
            new_base = np.where(candidate < base, candidate, base)
        else:
            candidate = np.where(work_mask & ~np.isnan(target), target, base)
            new_base = np.where(candidate > base, candidate, base)
        
        # Check convergence
        diff = np.abs(new_base[work_mask] - base[work_mask])
        delta = float(np.nanmax(diff)) if diff.size and not np.all(np.isnan(diff)) else 0.0
        base[:] = new_base
        
        # Progress callback
        if progress_callback and (itr % 50 == 0 or itr < 5):
            progress_callback(itr, config.max_iters, delta)
        
        if delta < config.stop_eps:
            if progress_callback:
                progress_callback(itr, config.max_iters, delta)
            break
    
    # Compute final thickness
    if is_failure:
        thickness = dem_np - base
    else:
        thickness = base - dem_np
    
    thickness = np.where(work_mask, np.maximum(thickness, 0.0), 0.0)
    
    return base, thickness


# =============================================================================
# Cross-Section Generation
# =============================================================================

def load_crosssection_lines(xsect_source: List[str], target_epsg: int) -> Optional[gpd.GeoDataFrame]:
    """Load all cross-section line shapefiles from given sources."""
    if not xsect_source:
        return None
    
    paths = []
    for source in xsect_source:
        if os.path.isdir(source):
            paths.extend([
                os.path.join(source, f) for f in os.listdir(source)
                if f.lower().endswith('.shp')
            ])
        elif source.lower().endswith('.shp') and os.path.exists(source):
            paths.append(source)
    
    if not paths:
        return None
    
    gdfs = []
    for p in paths:
        try:
            gdf = gpd.read_file(p)
            if gdf.crs and gdf.crs.to_epsg() != target_epsg:
                gdf = gdf.to_crs(epsg=target_epsg)
            gdf = gdf[gdf.geom_type.isin(['LineString', 'MultiLineString'])]
            if not gdf.empty:
                gdfs.append(gdf)
        except Exception as e:
            print(f"[WARN] Failed to load cross-section {p}: {e}")
    
    if not gdfs:
        return None
    
    return pd.concat(gdfs, ignore_index=True)


def densify_line(line, step_m: float) -> List[Tuple[float, float, float]]:
    """Densify LineString to specified step size, returning (x, y, distance) tuples."""
    from shapely.geometry import LineString
    
    L = line.length
    if L <= 0 or step_m <= 0:
        coords = list(line.coords)
        return [(coords[0][0], coords[0][1], 0.0), (coords[-1][0], coords[-1][1], L)]
    
    n = max(1, int(np.floor(L / step_m)))
    dists = [i * step_m for i in range(n + 1)]
    if dists[-1] < L:
        dists.append(L)
    
    pts = []
    for d in dists:
        p = line.interpolate(d)
        pts.append((p.x, p.y, d))
    
    return pts


def sample_rasters_along_line(
    xy_dist_pts: List[Tuple[float, float, float]],
    transform,
    dem_arr: np.ndarray,
    base_arr: np.ndarray,
    thick_arr: np.ndarray,
    inmask: Optional[np.ndarray] = None
) -> List[Tuple]:
    """Sample DEM, base, and thickness along line points."""
    from rasterio.transform import rowcol
    
    xs = [x for x, _, _ in xy_dist_pts]
    ys = [y for _, y, _ in xy_dist_pts]
    rows, cols = rowcol(transform, xs, ys, op=float)
    rows = np.rint(rows).astype(int)
    cols = np.rint(cols).astype(int)
    
    H, W = dem_arr.shape
    out = []
    
    for (x, y, d), r, c in zip(xy_dist_pts, rows, cols):
        if r < 0 or c < 0 or r >= H or c >= W:
            continue
        if inmask is not None and not inmask[r, c]:
            continue
        
        dz = float(dem_arr[r, c]) if np.isfinite(dem_arr[r, c]) else np.nan
        bz = float(base_arr[r, c]) if np.isfinite(base_arr[r, c]) else np.nan
        th = float(thick_arr[r, c]) if np.isfinite(thick_arr[r, c]) else np.nan
        out.append((d, x, y, dz, bz, th))
    
    return out


def infer_line_id(row) -> str:
    """Infer line ID from GeoDataFrame row attributes."""
    for k in ["name", "Name", "NAME", "id", "ID", "Id", "label", "Label", "line_id", "LineID"]:
        if k in row.index and pd.notna(row[k]):
            return str(row[k])
    return f"line{int(row.name)}"


def write_crosssections(
    scenario_name: str,
    label: str,
    xsect_lines_gdf: gpd.GeoDataFrame,
    dem_arr: np.ndarray,
    base_arr: np.ndarray,
    thick_arr: np.ndarray,
    transform,
    inmask: np.ndarray,
    output_dir: str,
    step_m: float = 1.0,
    clip_to_poly: bool = False
):
    """
    Generate cross-section CSV files along lines.
    
    Args:
        scenario_name: Name of scenario
        label: Parameter label
        xsect_lines_gdf: GeoDataFrame with LineString geometries
        dem_arr: DEM array
        base_arr: SLBL base surface array
        thick_arr: Thickness array
        transform: Raster transform
        inmask: Polygon mask
        output_dir: Output directory
        step_m: Sampling step in meters
        clip_to_poly: Whether to clip lines to polygon
    """
    from shapely.geometry import LineString, MultiLineString
    
    xsect_dir = os.path.join(output_dir, "xsections", scenario_name)
    os.makedirs(xsect_dir, exist_ok=True)
    
    for idx, row in xsect_lines_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        line_id = infer_line_id(row)
        
        # Handle MultiLineString
        if isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        elif isinstance(geom, LineString):
            parts = [geom]
        else:
            continue
        
        for part_num, line in enumerate(parts, start=1):
            # Densify line
            xy_dist = densify_line(line, step_m)
            
            # Sample rasters
            samples = sample_rasters_along_line(
                xy_dist, transform, dem_arr, base_arr, thick_arr,
                inmask=inmask if clip_to_poly else None
            )
            
            if not samples:
                continue
            
            # Create DataFrame
            df = pd.DataFrame(samples, columns=[
                'Distance_m', 'X', 'Y', 'DEM_z', 'Base_z', 'Thickness_m'
            ])
            df['LinePart'] = part_num
            
            # Write CSV
            filename = f"{scenario_name}_{label}_xsect_{line_id}_p{part_num}.csv"
            csv_path = os.path.join(xsect_dir, filename)
            df.to_csv(csv_path, index=False, float_format='%.3f')
    
    print(f"[XSECT] Written cross-sections to {xsect_dir}")


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_scenario(
    scenario_path: str,
    dem_path: str,
    output_dir: str,
    e_ratio: float,
    max_depth: Optional[float],
    config: SLBLConfig,
    target_epsg: int = 25833,
    iteration_callback: Optional[Callable] = None
) -> SLBLResult:
    """
    Process a single SLBL scenario with given parameters.
    
    Args:
        scenario_path: Path to scenario polygon shapefile
        dem_path: Path to DEM file
        output_dir: Output directory for results
        e_ratio: E-ratio parameter
        max_depth: Optional maximum depth constraint
        config: SLBL configuration
        target_epsg: Target coordinate system
        iteration_callback: Optional callback for iteration progress
    
    Returns:
        SLBLResult with computed metrics and output paths
    """
    scenario_name = Path(scenario_path).stem
    
    # Create label
    e_label = f"e{e_ratio:.2f}".replace(".", "p")
    md_label = f"md{int(max_depth)}m" if max_depth else "nolimit"
    label = f"{e_label}_{md_label}_{config.mode}"
    
    result = SLBLResult(
        scenario_name=scenario_name,
        label=label,
        e_ratio=e_ratio,
        max_depth_limit=max_depth,
        tolerance_C=0.0
    )
    
    try:
        # Load polygon
        gdf = gpd.read_file(scenario_path)
        if gdf.empty:
            raise ValueError("Empty shapefile")
        
        if gdf.crs is None:
            raise ValueError("Shapefile has no CRS")
        
        if gdf.crs.to_epsg() != target_epsg:
            gdf = gdf.to_crs(epsg=target_epsg)
        
        # Dissolve and buffer
        gdf['_diss'] = 1
        gdf = gdf.dissolve(by='_diss', as_index=False)
        
        if config.buffer_pixels > 0:
            # Get pixel size from DEM
            with rasterio.open(dem_path) as src:
                pixel_size = abs(src.transform.a)
            gdf['geometry'] = gdf.buffer(config.buffer_pixels * pixel_size)
        
        # Get polygon geometry
        from shapely.ops import unary_union
        poly = unary_union(gdf.geometry)
        
        # Compute dimensions
        L_rh, width = compute_poly_dimensions(poly)
        result.L_rh_m = L_rh
        result.width_m = width
        result.area_m2 = poly.area
        
        # Compute tolerance C
        with rasterio.open(dem_path) as src:
            pixel_size = abs(src.transform.a)
        
        tol_C = compute_tolerance_C(e_ratio, L_rh, pixel_size)
        if not np.isfinite(tol_C):
            raise ValueError(f"Invalid tolerance C: e={e_ratio}, L_rh={L_rh}")
        
        result.tolerance_C = tol_C
        
        # Read DEM window
        bounds = poly.bounds
        dem_data, profile, transform = read_dem_window(dem_path, bounds, buffer_m=500)
        pixel_area = pixel_size ** 2
        
        # Rasterize polygon
        inmask = rasterize_polygon(gdf, dem_data.shape, transform, config.all_touched)
        
        if inmask.sum() == 0:
            raise ValueError("Rasterized mask has 0 cells")
        
        # Compute z-floor if enabled
        z_floor = None
        if config.use_z_floor:
            z_floor = compute_z_floor(inmask, dem_data)
            if np.isfinite(z_floor):
                z_floor += config.z_floor_buffer_m
        
        # Apply max_depth constraint to config temporarily
        # (This is a simplified approach - the full script applies it in the iteration)
        
        # Run SLBL
        base, thickness = slbl_iterate(
            dem_data, inmask, tol_C, config, pixel_area,
            progress_callback=iteration_callback
        )
        
        # Apply max_depth constraint post-hoc if specified
        if max_depth is not None:
            thickness = np.minimum(thickness, max_depth)
            base = dem_data.filled(np.nan) - thickness
        
        # Apply z_floor constraint
        if z_floor is not None and np.isfinite(z_floor):
            dem_np = dem_data.filled(np.nan)
            base = np.maximum(base, z_floor)
            thickness = np.where(inmask, np.maximum(dem_np - base, 0.0), 0.0)
        
        # Compute statistics
        pos = thickness > 0
        result.max_depth_m = float(np.nanmax(thickness)) if np.any(pos) else 0.0
        result.mean_depth_m = float(np.nanmean(thickness[pos])) if np.any(pos) else 0.0
        result.volume_m3 = float(np.nansum(thickness) * pixel_area)
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        if config.write_geotiff:
            thick_path = os.path.join(output_dir, f"{scenario_name}_{label}_thickness.tif")
            base_path = os.path.join(output_dir, f"{scenario_name}_{label}_base.tif")
            
            save_geotiff(thick_path, thickness, profile, nodata=0.0)
            save_geotiff(base_path, base, profile, nodata=np.nan)
            
            result.thickness_path = thick_path
            result.base_path = base_path
        
        if config.write_ascii:
            thick_asc = os.path.join(output_dir, f"{scenario_name}_{label}_thickness.asc")
            save_ascii(thick_asc, thickness, transform, nodata=-9999)
        
        # Generate cross-sections if configured
        if config.write_xsections and config.xsect_lines_source:
            xsect_lines = load_crosssection_lines(config.xsect_lines_source, target_epsg)
            if xsect_lines is not None and not xsect_lines.empty:
                dem_np = dem_data.filled(np.nan)
                write_crosssections(
                    scenario_name=scenario_name,
                    label=label,
                    xsect_lines_gdf=xsect_lines,
                    dem_arr=dem_np,
                    base_arr=base,
                    thick_arr=thickness,
                    transform=transform,
                    inmask=inmask,
                    output_dir=output_dir,
                    step_m=config.xsect_step_m,
                    clip_to_poly=config.xsect_clip_to_poly
                )
        
        result.success = True
        
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    return result


def run_slbl_batch(
    scenario_paths: List[str],
    dem_path: str,
    output_dir: str,
    config: SLBLConfig,
    target_epsg: int = 25833,
    progress_callback: Optional[Callable] = None,
    job_id: Optional[str] = None,
    job_created_by: Optional[str] = None,
    job_created: Optional[str] = None
) -> List[SLBLResult]:
    """
    Run SLBL processing for multiple scenarios and parameter combinations.

    Args:
        scenario_paths: List of scenario shapefile paths
        dem_path: Path to DEM file
        output_dir: Output directory
        config: SLBL configuration
        target_epsg: Target coordinate system
        progress_callback: Callback(current, total, message, sub_progress)
        job_id: Optional job ID for tracking
        job_created_by: Optional username who created the job
        job_created: Optional timestamp when job was created

    Returns:
        List of SLBLResult objects
    """
    results = []

    # Build job list
    jobs = []
    for scenario_path in scenario_paths:
        for e_ratio in config.e_ratios:
            for max_depth in config.max_depths:
                jobs.append((scenario_path, e_ratio, max_depth))

    total_jobs = len(jobs)

    for i, (scenario_path, e_ratio, max_depth) in enumerate(jobs):
        scenario_name = Path(scenario_path).stem

        if progress_callback:
            progress_callback(
                i, total_jobs,
                f"Processing {scenario_name} (e={e_ratio:.2f})",
                0.0
            )

        # Iteration callback for sub-progress
        def iter_callback(itr, max_iters, delta):
            if progress_callback:
                sub = min(itr / max_iters, 1.0)
                progress_callback(
                    i, total_jobs,
                    f"{scenario_name}: iter {itr}, δ={delta:.2e}",
                    sub
                )

        result = process_single_scenario(
            scenario_path=scenario_path,
            dem_path=dem_path,
            output_dir=output_dir,
            e_ratio=e_ratio,
            max_depth=max_depth,
            config=config,
            target_epsg=target_epsg,
            iteration_callback=iter_callback
        )

        # Add job tracking info
        result.job_id = job_id
        result.job_created_by = job_created_by
        result.job_created = job_created

        results.append(result)
    
    # Final progress update
    if progress_callback:
        progress_callback(total_jobs, total_jobs, "Complete", 1.0)
    
    return results


def save_batch_summary(results: List[SLBLResult], output_path: str):
    """Save batch results to CSV summary, appending to existing results."""
    rows = [r.to_dict() for r in results]
    new_df = pd.DataFrame(rows)

    # Load existing results and append
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            df = new_df
    else:
        df = new_df

    df.to_csv(output_path, index=False)
    return df
