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


class RasterClass:
    """
    This is a class for generalizing raster inputs from multiple input sources/types such as from file, rasterio
    loader, and xarray.

    Attributes:

    """
    def __init__(self, ds):
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
                self.transform = ds.rio.transform
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
            self.transform = ds.rio.transform
            self.bounds = ds.rio.bounds()
            self.resolution = ds.rio.resolution()
        else:
            raise ValueError("The input is not recognized as a file path, rasterio DatasetReader, or xarray object.")


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



