"""Utility classes and functions used by the GRIDtools package.

This module contains the following classes:
- RasterClass
This module contains the following functions:
- vectorize_grid()
- find_intersections()

-  **Author(s):** Todd Blythe, MTDNRC, CNB968
-  **Date:** Created 12/1/2025
"""
from pathlib import Path
from typing import Union, Optional

import geopandas as gpd
import rasterio as rio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from shapely.geometry import Polygon
from affine import Affine


class RasterClass:
    """Container for raster datasets.

    Generalizes raster or gridded data by representing it as a standard set of attributes that can be transformed,
    edited, and/or recreated for various end uses. Can be initialized directly by dataset metadata or loaded from
    multiple input sources/types such as from file, rasterio loader, or xarray objects.

    Attributes:
        values:
            A numpy ndarray of raster values
        crs:
            A rasterio CRS object or WKT string of the coordinate reference system
        transform:
            An affine.Affine object representing the Affine transform for the raster dataset
        bounds:
            Tuple of dataset bounds (xmin, ymin, xmax, ymax)
        resolution:
            Tuple containing the pixel resolution of the dataset (x_pixel_width, y_pixel_height)
        band_idx_labels:
            A numpy ndarray of the index labels or values for the bands of the raster (only valid for xarray objects)
        dim_names:
            A list containing the names of the dataset dimensions (only valid for xarray objects)
        variables:
            A string or list of strings representing the variable(s) in the raster dataset
    """
    def __init__(self,
                 data: np.ndarray,
                 crs: rio.crs.CRS,
                 transform: Affine,
                 bounds: tuple,
                 resolution: tuple,
                 var_names: Optional[Union[str, list]] = None,
                 band_labels: Optional[Union[list, np.ndarray]] = None,
                 dim_names: Optional[Union[list, np.ndarray]] = None,
                 ):
        """
        Initializes the instance based on direct raster metadata input.

        Args:
            data:
                Defines the data array associated with the instance.
            crs:
                Defines instance coordinate reference information.
            transform:
                Defines affine transform for the instance.
            bounds:
                Defines the instance geospatial grid bounds.
            resolution:
                Defines the instance's x- and y- resolution or grid distance of each cell.
            var_names:
                The variable names of the instance

                This relates to the axis=0 dimension of the instance data array.
            band_labels:
                Labels for the 3rd dimension of the instance (the bands of the raster).

                Relates to the axis=1 dimension of the data array.
            dim_names:
                The names of the instance dimensions.

                These are equivalent to the labels for the various axes of the data array.
        """
        # TODO: should have each of these as properties with setter methods...this would allow more error checking and
        #   make sure inputs conform to the required format.
        if len(data.shape) < 3:
            self.values = np.stack([data], axis=0)
        else:
            self.values = data
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.resolution = resolution
        self.band_idx_labels = band_labels
        self.dim_names = dim_names
        if var_names is None:
            self.variables = 'Value'
        else:
            self.variables = var_names

    @staticmethod
    def load(ds: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset]):
        """A loader function to create an instance from file or separate raster-like object.

        Args:
            ds: The input raster dataset as a file-path, rasterio, or xarray object.

        Returns:
            An instance of the class, a RasterClass object.
        """
        if isinstance(ds, (str, Path)):

            if isinstance(ds, str):
                ds = Path(ds)

            if ds.suffix == '.tif':
                ds = rio.open(ds)

                values = np.stack([ds.read()], axis=0)
                crs = ds.crs
                transform = ds.transform
                bounds = (ds.bounds.left,
                          ds.bounds.bottom,
                          ds.bounds.right,
                          ds.bounds.top)
                resolution = ds.res
                ds.close()
                variables = None
                band_idx_labels = None
                dim_names = None

            elif ds.suffix == '.nc':
                ds = xr.open_dataset(ds)

                if isinstance(ds, xr.Dataset):
                    variables = list(ds.data_vars)
                    values = np.stack([ds[var] for var in list(ds.data_vars)], axis=0)
                    vdim_list = [ds[v].dims for v in list(ds.data_vars)]
                    dim_names = vdim_list[0]
                    if not all(element == dim_names for element in vdim_list):
                        raise ValueError("The dimensions of data variables in the dataset are not all the same. "
                                         "Datasets with multiple variables are only allowed if all variable dimensions "
                                         "are equal.")
                    dim_names = list(dim_names)
                elif isinstance(ds, xr.DataArray):
                    variables = ds.name
                    values = np.stack([ds.values], axis=0)
                    dim_names = list(ds.dims)
                else:
                    raise ValueError("Datatype not recognized, check input file or object.")
                crs = ds.rio.crs
                transform = ds.rio.transform()
                bounds = ds.rio.bounds()
                resolution = ds.rio.resolution()
                band_idx_labels = ds[dim_names[0]].values
            else:
                raise ValueError("File type not recognized. Currently only .tif and .nc are supported.")

        elif isinstance(ds, rio.DatasetReader):
            values = np.stack([ds.read()], axis=0)
            crs = ds.crs
            transform = ds.transform
            bounds = (ds.bounds.left,
                      ds.bounds.bottom,
                      ds.bounds.right,
                      ds.bounds.top)
            resolution = ds.res
            ds.close()
            variables = None
            band_idx_labels = None
            dim_names = None

        elif isinstance(ds, (xr.DataArray, xr.Dataset)):
            if isinstance(ds, xr.Dataset):
                variables = list(ds.data_vars)
                values = np.stack([ds[var] for var in list(ds.data_vars)], axis=0)
                vdim_list = [ds[v].dims for v in list(ds.data_vars)]
                dim_names = vdim_list[0]
                if not all(element == dim_names for element in vdim_list):
                    raise ValueError("The dimensions of data variables in the dataset are not all the same. "
                                     "Datasets with multiple variables are only allowed if all variable dimensions "
                                     "are equal.")
                dim_names = list(dim_names)
            elif isinstance(ds, xr.DataArray):
                variables = ds.name
                values = np.stack([ds.values], axis=0)
                dim_names = list(ds.dims)
            else:
                raise ValueError("Datatype not recognized, check input file or object.")
            crs = ds.rio.crs
            transform = ds.rio.transform()
            bounds = ds.rio.bounds()
            resolution = ds.rio.resolution()
            band_idx_labels = ds[dim_names[0]].values
        else:
            raise ValueError("The input is not recognized as a file path, rasterio DatasetReader, or xarray object.")

        return RasterClass(data=values,
                           crs=crs,
                           transform=transform,
                           bounds=bounds,
                           resolution=resolution,
                           var_names=variables,
                           band_labels=band_idx_labels,
                           dim_names=dim_names)


def vectorize_grid(dataset: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset, RasterClass]
                   ) -> gpd.GeoDataFrame:
    """Creates vectorized representation of raster dataset grid.

    Args:
        dataset:
            An input gridded dataset, can be multidimensional.

    Returns:
         Polygons representing the raster grid cells.
    """
    if isinstance(dataset, RasterClass):
        raster = dataset
    else:
        raster = RasterClass.load(dataset)

    # get grid resolution
    xres, yres = raster.resolution

    # get the array bounds as variables for new geometry
    gminx, gminy, gmaxx, gmaxy = raster.bounds

    # shape of dataset
    nrows = raster.values.shape[2]
    ncols = raster.values.shape[3]

    nshp_cols = list(np.linspace(gminx, gmaxx, ncols + 1))
    nshp_rows = np.linspace(gminy, gmaxy, nrows + 1)
    nshp_rows = np.flip(nshp_rows)

    grid_polygons = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            grid_polygons.append(Polygon([(x, y), (x + xres, y), (x + xres, y + yres), (x, y + yres)]))

    grid_polys = gpd.GeoDataFrame(geometry=grid_polygons, crs=raster.crs)

    return grid_polys


def find_intersections(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Function to find any intersecting geometries in a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame with valid geometries.

    Returns:
        A new GeoDataFrame of the intersecting geometries from the input GeoDataFrame.
    """
    # Save geometries to another field
    gdf.loc[:, 'geom'] = gdf.geometry

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
