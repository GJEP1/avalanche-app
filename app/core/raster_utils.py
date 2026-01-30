"""
Shared raster utilities for DEM and thickness raster operations.
"""

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import geopandas as gpd
from shapely.geometry import box


def read_raster(raster_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read a raster file and return data and metadata.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Tuple of (data array, metadata dict)
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    
    metadata = {
        'bounds': bounds,
        'transform': transform,
        'crs': crs,
        'shape': data.shape,
        'nodata': meta.get('nodata'),
        'dtype': data.dtype
    }
    
    return data, metadata


def write_raster(data: np.ndarray, output_path: Path, metadata: dict):
    """
    Write array to raster file.
    
    Args:
        data: 2D numpy array
        output_path: Path for output file
        metadata: Metadata dict with 'transform', 'crs', etc.
    """
    meta = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': metadata.get('crs'),
        'transform': metadata.get('transform'),
        'nodata': metadata.get('nodata', -9999)
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)


def get_raster_bounds(raster_path: Path) -> box:
    """Get bounds of a raster as a Shapely box."""
    with rasterio.open(raster_path) as src:
        return box(*src.bounds)


def raster_matches_shapefile(raster_path: Path, shapefile_path: Path, 
                             tolerance: float = 1.0) -> bool:
    """
    Check if a raster adequately covers a shapefile extent.
    
    Args:
        raster_path: Path to raster
        shapefile_path: Path to shapefile
        tolerance: Tolerance in map units
        
    Returns:
        True if raster covers shapefile extent
    """
    raster_bounds = get_raster_bounds(raster_path)
    
    gdf = gpd.read_file(shapefile_path)
    shp_bounds = box(*gdf.total_bounds)
    
    # Check if shapefile is within raster bounds (with tolerance)
    raster_buffered = raster_bounds.buffer(tolerance)
    
    return raster_buffered.contains(shp_bounds)


def clip_raster_to_extent(input_raster: Path, output_raster: Path, 
                          extent_shapefile: Path, buffer: float = 100.0):
    """
    Clip a raster to the extent of a shapefile with optional buffer.
    
    Args:
        input_raster: Input raster path
        output_raster: Output raster path
        extent_shapefile: Shapefile defining extent
        buffer: Buffer distance in map units
    """
    gdf = gpd.read_file(extent_shapefile)
    bounds = gdf.total_bounds
    
    # Add buffer
    bounds = [
        bounds[0] - buffer,
        bounds[1] - buffer,
        bounds[2] + buffer,
        bounds[3] + buffer
    ]
    
    with rasterio.open(input_raster) as src:
        # Calculate window
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        
        # Read data
        data = src.read(1, window=window)
        
        # Calculate new transform
        transform = rasterio.windows.transform(window, src.transform)
        
        # Write clipped raster
        meta = src.meta.copy()
        meta.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': transform
        })
        
        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(data, 1)


def reproject_raster(input_path: Path, output_path: Path, 
                     target_crs: str = "EPSG:25833"):
    """
    Reproject a raster to a different CRS.
    
    Args:
        input_path: Input raster path
        output_path: Output raster path
        target_crs: Target coordinate reference system
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )


def calculate_raster_statistics(raster_path: Path, 
                                mask_shapefile: Optional[Path] = None) -> dict:
    """
    Calculate statistics for a raster, optionally within a mask.
    
    Args:
        raster_path: Path to raster
        mask_shapefile: Optional shapefile to mask the raster
        
    Returns:
        Dict with min, max, mean, std, etc.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        
        if mask_shapefile:
            import rasterio.mask
            gdf = gpd.read_file(mask_shapefile)
            data, _ = rasterio.mask.mask(src, gdf.geometry, crop=False)
            data = np.ma.masked_array(data[0], mask=(data[0] == src.nodata))
    
    # Calculate statistics on valid data
    valid_data = data.compressed()
    
    if len(valid_data) == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'percentiles': {}
        }
    
    return {
        'count': len(valid_data),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'percentiles': {
            'p10': float(np.percentile(valid_data, 10)),
            'p50': float(np.percentile(valid_data, 50)),
            'p90': float(np.percentile(valid_data, 90)),
        }
    }
