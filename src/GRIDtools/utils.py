## utility functions for use in other GRIDtools methods
# CNB968 - Todd Blythe

import geopandas as gpd
import rasterio as rio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pathlib import Path
from shapely.geometry import Polygon
from shapely.affinity import translate


class RasterClass:
    """
    This is a class for generalizing raster inputs from multiple input sources/types such as from file, rasterio
    loader, and xarray.

    Attributes:
        values(numpy.array) : value array of the raster
        crs(str) : wkt string of the coordinate reference system
        transform(affine.Affine) : Affine transform for the raster dataset
        bounds(tuple) : the dataset bounds (xmin, ymin, xmax, ymax)
        resolution(tuple) : the pixel resolution of the dataset (x_pixel_width, y_pixel_height)
        band_idx(numpy.array): the index labels or values for the bands of the raster (only for xarray objects)
        dim_names(list): the names of the dataset dimensions (only for xarray objects)
    """
    def __init__(self, ds):
        """
        Initialization function for RasterClass that generalizes multiple sources of raster input data to a set of
        common attributes.

        Args:
            ds(str | Path | rasterio.DatasetReader | xarray.DataArray | xarray.Dataset): input raster or
            multidimensional gridded dataset. Must have valid CRS and attributes collable by rioxarray if it is an
            xarray object.
        """
        self.band_idx = None
        self.dim_names = None

        if isinstance(ds, (str, Path)):

            if isinstance(ds, str):
                ds = Path(ds)

            if ds.suffix == '.tif':
                ds = rio.open(ds)

                self.values = ds.read()
                self.crs = ds.crs
                self.transform = ds.transform
                self.bounds = (ds.bounds.left,
                               ds.bounds.bottom,
                               ds.bounds.right,
                               ds.bounds.top)
                self.resolution = ds.res
                ds.close()

            elif ds.suffix == '.nc':
                ds = xr.open_dataset(ds)

                # if there are multiple variables this selects the first for getting grid properties
                if isinstance(ds, xr.Dataset):
                    self.values = ds[list(ds.data_vars)[0]].values
                else:
                    self.values = ds.values
                self.crs = ds.rio.crs
                self.transform = ds.rio.transform()
                self.bounds = ds.rio.bounds()
                self.resolution = ds.rio.resolution()

            else:
                raise ValueError("File type not recognized. Currently only .tif and .nc are supported.")

        elif isinstance(ds, rio.DatasetReader):
            self.values = ds.read()
            self.crs = ds.crs
            self.transform = ds.transform
            self.bounds = (ds.bounds.left,
                           ds.bounds.bottom,
                           ds.bounds.right,
                           ds.bounds.top)
            self.resolution = ds.res
            ds.close()

        elif isinstance(ds, (xr.DataArray, xr.Dataset)):
            if isinstance(ds, xr.Dataset):
                self.values = ds[list(ds.data_vars)[0]].values
            else:
                self.values = ds.values
            self.crs = ds.rio.crs
            self.transform = ds.rio.transform()
            self.bounds = ds.rio.bounds()
            self.resolution = ds.rio.resolution()
            self.dim_names = self._get_xarray_dim_names(ds)
            self.band_idx = ds[self.dim_names[0]].values
        else:
            raise ValueError("The input is not recognized as a file path, rasterio DatasetReader, or xarray object.")

    def _get_xarray_dim_names(self, ds):
        dimnames = []
        for x in self.values.shape:
            dimname = next((name for name, size in dict(ds.sizes).items() if size == x), None)
            dimnames.append(dimname)

        return dimnames


def vectorize_grid(dataset):
    """
    Function to create a vectorized grid from a raster dataset where each pixel is a polygon.

    Args:
        dataset (str | pathlib.Path | rasterio.DatasetReader | xarray.DataArray | xarray.Dataset) : An input gridded
            dataset, can be multidimensional.

    Returns:
        geopandas.GeoDataFrame : Polygons representing the raster grid cells.
    """
    raster = RasterClass(dataset)

    # get grid resolution
    xres, yres = raster.resolution

    # get the array bounds as variables for new geometry
    gminx, gminy, gmaxx, gmaxy = raster.bounds

    # shape of dataset
    nrows = raster.values.shape[1]
    ncols = raster.values.shape[2]

    nshp_cols = list(np.linspace(gminx, gmaxx, ncols + 1))
    nshp_rows = np.linspace(gminy, gmaxy, nrows + 1)
    nshp_rows = np.flip(nshp_rows)

    grid_polygons = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            grid_polygons.append(Polygon([(x, y), (x + xres, y), (x + xres, y + yres), (x, y + yres)]))

    grid_polys = gpd.GeoDataFrame(geometry=grid_polygons, crs=raster.crs)

    return grid_polys


def jitter_geometries(pnt_gdframe, tolerance=0.1):
    """
    Function that applies "jitter" or random noise to geometries in a Geopandas GeoDataFrame.

    Args:
        pnt_gdframe(geopandas.GeoDataFrame): A GeoDataFrame with valid geometry column.
        tolerance(float): The spread of the random noise added (interpreted as the standard-deviation).

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with "jittered" geometries
    """
    gdf = pnt_gdframe

    gdf.loc[:,'geometry'] = gdf['geometry'].apply(lambda point: translate(
        point,
        np.random.normal(0, tolerance / 2),
        np.random.normal(0, tolerance / 2)
    ))

    return gdf


def find_intersections(gdf: gpd.GeoDataFrame):
    """
    Function to find any intersecting geometries in a GeoDataFrame.

    Args:
        gdf(geopandas.GeoDataFrame): Input GeoDataFrame to assess intersections.

    Returns:
        geopandas.GeoDataFrame: Output GeoDataFrame
    """
    # Save geometries to another field
    gdf.loc[:,'geom'] = gdf.geometry

    # Self join
    sj = gpd.sjoin(gdf, gdf,
                   how="inner",
                   predicate="intersects",
                   lsuffix="left",
                   rsuffix="right")

    # Remove geometries that intersect themselves
    sj = sj[sj.index != sj.index_right]

    # Extract the intersecting geometry
    sj['intersection_geom'] = sj['geom_left'].intersection(sj['geom_right'])

    geom_name = sj.active_geometry_name
    # Reset the geometry (remember to set the CRS correctly!)
    sj = sj.set_geometry('intersection_geom', crs=gdf.crs)

    # Drop duplicate geometries
    final_gdf = sj.drop_duplicates(subset=['geometry']).reset_index()

    # Drop intermediate fields
    drops = ['geom_left', 'geom_right', 'index_right', 'index']
    final_gdf = final_gdf.drop(drops, axis=1)

    return final_gdf