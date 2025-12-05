"""Zonal operations on gridded data using a geometry

The zonal module has some commonly used operations in Hydrologic Science (or Earth Science) including
zonal statistics, where raster (gridded) data are associated with particular geometries based on selected
summary statistics.

This module contains the following functions:
- calc_zonal_stats()
- grid_area_weighted_volume()
- sample_raster_points()

-  **Author(s):** Todd Blythe, MTDNRC, CNB968
-  **Date:** Created 12/1/2025
"""
import warnings
from pathlib import Path
from typing import Union, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import DatasetReader
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from rasterio.transform import rowcol
from shapely.geometry import Polygon

from GRIDtools.utils import vectorize_grid, RasterClass


def calc_zonal_stats(in_geom: Union[str, Path, gpd.GeoDataFrame],
                     in_grid: Union[str, Path, DatasetReader, xr.DataArray, xr.Dataset, RasterClass],
                     method: str = 'groupby',
                     stats: Union[str, list] = 'mean',
                     all_touched: bool = False,
                     output: str = 'pandas',
                     **kwargs) -> Union[pd.DataFrame, xr.Dataset]:
    """Calculates zonal statistics for a set of geometries based input raster data.

    Uses various methods to calculate zonal statistics for set of geometries. This method rasterizes the geometries
    first to calculate zonal statistics. Two methods are available, one requires the additional dependency rasterstats
    package. The default method uses pandas groupby and is compatible with multidimensional (multiband) datsets.

    Args:
        in_geom:
            input file path or GeoDataFrame of geometries to summarize.
        in_grid:
            the input gridded dataset to summarize over the input geometries.
        method:
            'groupby' or 'rasterstats' the default is 'groupby'
        stats:
            The name of the statistic to use for zonal stats, or list of valid statistics - default 'mean.'
            These differ based on the method argument.
            If method == 'rasterstats' then options are any string accepted by that package. See accepted statistics
            strings here: https://pythonhosted.org/rasterstats/manual.html#statistics

            If method == 'groupby' (the default), any statistics function strings accepted by pandas SeriesGroupBy
            descriptive stats. Not all functions may be represented by a string name but their series method can be
            used in the stats argument (e.g., stats=['mean', 'count', pd.Series.mode] where pd.Series.mode returns the
            majority value within the geometry.
        all_touched:
            whether to include all cells intersected by each geometry (True) or only those with center points within
            each geometry (False) - default is False
        output:
            Specifies what type to return, either 'pandas' for pandas.DataFrame or 'xarray' for xarray.Dataset.
            This is ignored if method == 'rasterstats' which will always output a pandas dataframe.
        **kwargs:
            additional key word arguments accepted by rasterstats package

    Returns:
            A multiindex/dimensional DataFrame or dataset with the summary statistics for
            each input geometry and all raster bands/dimensions as data variables or columns

    Raises:
        ValueError: If stats argument is not a string or list
        ValueError: If 'output' argument is neither 'pandas' nor 'xarray'
        ValueError: If 'method' argument is not one of ['groupby', 'rasterstats']

    """
    # get geometry in same reference system
    if isinstance(in_geom, (str, Path)):
        in_geom = gpd.read_file(in_geom)

    geom = in_geom
    if isinstance(in_grid, RasterClass):
        raster = in_grid
    else:
        raster = RasterClass.load(in_grid)

    # check crs
    if geom.crs.to_authority() != raster.crs.to_authority():
        geom = geom.to_crs(raster.crs)

    #TODO - add checks for valid stats
    if isinstance(stats, str):
        stats = [stats]
    elif isinstance(stats, list):
        pass
    else:
        raise ValueError("stats argument must be either valid string or list of accepted methods")

    if method == 'rasterstats':
        try:
            from rasterstats import zonal_stats
        except ImportError as e:
            print(f"The rasterstats package is required for method 'rasterstats' but is not installed: {e}")

        if kwargs.get('layer') is not None:
            layer = kwargs.get('layer')
        else:
            layer = 0

        if kwargs.get('band') is not None:
            band = kwargs.get('band')
        else:
            band = 1

        if kwargs.get('categorical') is not None:
            categorical = kwargs.get('categorical')
        else:
            categorical = False

        if kwargs.get('raster_out') is not None:
            raster_out = kwargs.get('raster_out')
        else:
            raster_out = False

        if kwargs.get('geojson_out') is not None:
            geojson_out = kwargs.get('geojson_out')
        else:
            geojson_out = False

        if kwargs.get('boundless') is not None:
            boundless = kwargs.get('boundless')
        else:
            boundless = True

        # Need to add loop to deal with multiple bands
        zs = zonal_stats(geom.geometry,
                         raster.values[0, 0, :, :],
                         affine=raster.transform,
                         stats=stats,
                         all_touched=all_touched,
                         layer=layer,
                         band=band,
                         nodata=kwargs.get('nodata'),
                         categorical=categorical,
                         category_map=kwargs.get('category_map'),
                         add_stats=kwargs.get('add_stats'),
                         zone_func=kwargs.get('zone_func'),
                         raster_out=raster_out,
                         prefix=kwargs.get('prefix'),
                         geojson_out=geojson_out,
                         boundless=boundless
        )

        fgd = geom.join(pd.DataFrame(zs))

    elif method == 'groupby':
        def rasterized_to_df(rasterized_features,
                             in_raster_values,
                             band_idx,
                             band_name,
                             var_names,
                             stats):
            adj_rzd = rasterized_features - 1
            df_dict = {'FID': np.tile(adj_rzd.ravel(), band_idx.size), band_name: band_idx.repeat(adj_rzd.ravel().size)}

            if isinstance(var_names, str):
                df_dict[var_names] = in_raster_values[0, :, :, :].ravel()
            else:
                for i in range(len(var_names)):
                    df_dict[var_names[i]] = in_raster_values[i, :, :, :].ravel()

            rstrzd_df = pd.DataFrame(
                df_dict
            )

            filtered_df = rstrzd_df.loc[rstrzd_df.FID != -1, :]
            grouped = filtered_df.groupby(['FID', band_name]).agg(stats)
            Fstck = grouped.stack(level=1, future_stack=True)
            new_names = list(Fstck.index.names)[0:2] + ['stat']
            ret_df = Fstck.rename_axis(index=new_names)

            return ret_df

        bands = raster.values.shape[1]
        if raster.band_idx_labels is None:
            band_idx_labels = np.arange(1, bands + 1)
        else:
            band_idx_labels = raster.band_idx_labels

        if raster.dim_names is None:
            band_nm = 'Band'
        else:
            band_nm = raster.dim_names[0]

        geom_value = ((geom, value) for geom, value in zip(geom.geometry, geom.index + 1))
        rasterized = rasterize(
            geom_value,
            out_shape=(raster.values.shape[2], raster.values.shape[3]),
            fill=0,
            transform=raster.transform,
            all_touched=all_touched,
            dtype=np.int64,
            merge_alg=MergeAlg.replace
        )

        result_df = rasterized_to_df(rasterized,
                                     raster.values,
                                     band_idx_labels,
                                     band_nm,
                                     raster.variables,
                                     stats)

        n_rstrzed = np.unique(rasterized)
        gids = n_rstrzed[1:] - 1
        if len(gids) != len(geom.index):
            warnings.warn(
                " ".join(
                    [
                        f"Not all geometries were returned, {len(geom.index) - len(n_rstrzed[1:])} "
                        f"geometries were missed during rasterize. Attempting to rasterize missed geometries...",
                    ]
                ),
                UserWarning,
                stacklevel=2
            )

            missed_polys = geom.loc[~geom.index.isin(gids), :]

            geom_value = ((geom, value) for geom, value in zip(missed_polys.geometry, missed_polys.index + 1))
            m_rasterized = rasterize(
                geom_value,
                out_shape=(raster.values.shape[2], raster.values.shape[3]),
                fill=0,
                transform=raster.transform,
                all_touched=all_touched,
                dtype=np.int64,
                merge_alg=MergeAlg.replace
            )

            m_result_df = rasterized_to_df(m_rasterized,
                                           raster.values,
                                           band_idx_labels,
                                           band_nm,
                                           raster.variables,
                                           stats)
            result_df = pd.concat([result_df, m_result_df])

            n_rstrzed = result_df.index.get_level_values(0).unique()
            if len(n_rstrzed) != len(geom.index):
                print(
                    "Second attempt did not return all geometries, defaulting to point locations for remaining geometries.")

                ## Indexing problem here, fix by finishing point sampler function and then just call that
                #       instead of imbedding separate code here.
                missed_polys = geom.loc[~geom.index.isin(n_rstrzed), :]
                cntrs = missed_polys.centroid
                pnt_df = sample_raster_points(cntrs.x, cntrs.y, raster, pnt_index=missed_polys.index.values, output='pandas_long')
                pnt_df['stat'] = 'single_point'
                pnt_df = pnt_df.set_index(['FID', band_nm, 'stat']).sort_index(level=['FID', band_nm])

                result_df = pd.concat([result_df, pnt_df])

        if len(result_df.index.get_level_values(0).unique()) != len(geom.index):
            warnings.warn(
                " ".join(
                    [
                        f"Point locations failed to capture all geometries, ",
                        "{len(geom.index) - len(result_df.index.get_level_values(0).unique())} unrepresented geometries",
                        "will be skipped..."
                    ]
                ),
                UserWarning,
                stacklevel=2
            )

        fgd = result_df.sort_index(level=['FID', band_nm])

        if output == 'pandas':
            pass
        elif output == 'xarray':
            fgd = fgd.to_xarray()
        else:
            raise ValueError("The output argument is not recognized, choose between 'pandas or 'xarray.'")

    else:
        raise ValueError("The method argument is not recognized, please choose 'rasterstats' or 'groupby.'")
    # return GeoDataFrame with stats added as additional attributes
    return fgd


def grid_area_weighted_volume(dataset: Union[str, Path, DatasetReader, xr.DataArray, xr.Dataset, RasterClass],
                              in_geom: gpd.GeoDataFrame,
                              geom_id_col: Optional[str],
                              data_scale: float = 1.0) -> xr.Dataset:
    """Function to calculate are weighted volume for a geometry from overlapping gridded data.

    In hydrologic or climate sciences, it is common to have depth valued data (e.g., precipitation)
    that needs to be summarized as a volume for a non-uniform area (e.g., a watershed). This function calculates
    the percent of overlap between the underlying area geometry and the gridded dataset grid so that the sum of all
    overlapping grid cells are weighted by their individual overlap. This ensures that the cumulative sum of
    all the averlapping cells is not skewed by grid cells with high or low depth values that only intersect
    a small portion of the underlying area geometry.

    Args:
        dataset:
            Raster, or multiband/multidimensional dataset containing one depth valued data variables (if multiple
            data variables exist in the dataset volumes for only the first variable will be returned).
        in_geom:
            A Geopandas GeoDataFrame with a geometry or geometries to calculate the weighted volume over.

            As of this version, this must be in the same coordinate reference system as the 'dataset' argument. Only
            polygons are accepted.
        geom_id_col:
            An optional string of a column name found in the GeoDataFrame that will be used to index the geometry data
            for the output, this could be a name or location identifier. If None, an integer index will be assigned
            by default.
        data_scale:
            A scaling value to convert the depth valued input data. Default is 1.0, meaning no scaling occurs.

            For example, precipitation data in millimeters, passing a data_scale value of 1000.0 would convert the
            precipitation values to meters.

    Returns:
        An xarray Dataset containing the volume calculated for the data variable in the input dataset. The Dataset
        will also contain coordinate variables for the index/names of each geometry and their area.

    Raises:
        ValueError: If input geometries are not polygons.
        IndexError: if the geom_id_col argument is not found in the input geometry column names
    """

    if (in_geom.geom_type != 'Polygon').any():
        raise ValueError("The input geometry(s) are not all type Polygon. Only Polygons are supported.")

    if isinstance(dataset, RasterClass):
        rast = dataset
    else:
        rast = RasterClass.load(dataset)

    grid_polys = vectorize_grid(rast)
    grid_polys.index.name = 'GridID'
    nrows = rast.values.shape[2]
    ncols = rast.values.shape[3]

    g_proj = in_geom.to_crs(5071)
    grd_proj = grid_polys.to_crs(5071)

    # get in shape areas in Km^2
    ingeom_areas = g_proj.area.values / (1000 ** 2)

    clips = []
    for r in range(len(g_proj.index)):
        clpd = gpd.clip(grd_proj, g_proj.loc[[r]], sort=True)

        if geom_id_col is None:
            clpd['FeatureID'] = r
        elif geom_id_col in in_geom.columns.to_list():
            clpd['FeatureID'] = in_geom.loc[r][geom_id_col]
        else:
            raise IndexError("Argument for geom_id_col not found in input geometry columns.")

        clips.append(clpd)

    clipped_grid_shp = pd.concat(clips)
    clipped_grid_shp['CellArea_sqm'] = clipped_grid_shp.geometry.area
    clipped_grid_shp.index.name = 'GridID'

    # Loop through unique feature ID's calculate the weighted volume for each
    # TODO: this currently just does the first data variable for a multidimensional dataset, need to make this
    #   work for all variables in the RasterClass
    vol_series = []
    for gid in in_geom[geom_id_col].to_list():
        qarea = pd.DataFrame(
            {'CellArea_sqm': clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid]['CellArea_sqm'].values},
            index=clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid].index
        )
        allcells = pd.concat([qarea, grd_proj], axis=1)
        allcells.sort_index(inplace=True)
        area_arry = allcells['CellArea_sqm'].values.reshape(nrows, ncols)
        md_arry = rast.values[0, :, :, :] / data_scale
        vol_grd = area_arry * md_arry
        vol_v = np.nansum(vol_grd, axis=(1, 2))
        vol_v = vol_v.reshape(vol_v.size, 1)
        vol_series.append(vol_v)

    # Create xarray dataset of result
    agg_dset = xr.Dataset(
        {
            "volume": (
            ['time', 'location'], np.hstack(vol_series), {'standard_name': 'Area Weighted Volume',
                                                          'units': 'm^3'}),
        },
        coords={
            "location": (['location'], in_geom[geom_id_col].to_list(), {'long_name': 'location_identifier',
                                                                        'cf_role': 'timeseries_id'}),
            "area": (['location'], ingeom_areas, {'standard_name': 'area',
                                                  'long_name': 'input_shape_area',
                                                  'units': 'km^2'}),
            "time": dataset.indexes['time']
        },
        attrs={
            "featureType": 'timeSeries'
        }
    )

    return agg_dset

# TODO: add a method selection here that allows for an xarray dataset/array to use the nearest selection method as it
#   seems to be more performant
def sample_raster_points(xcoords: Union[list, np.ndarray],
                         ycoords: Union[list, np.ndarray],
                         in_grid: Union[str, Path, DatasetReader, xr.DataArray, xr.Dataset, RasterClass],
                         pnt_index: Optional[Union[list, np.ndarray]],
                         output: str = 'pandas_long') -> Union[pd.DataFrame, xr.Dataset]:
    """Samples raster data at point locations.

    Function takes a list of x and y coordinates and returns a dataframe or dataset of values sampled from a raster
    input dataset.This function returns values across all dimensions of a multidimensional raster. This function does
    not check for matching coordinate reference systems so points must be in the same CRS as the input raster.

    Args:
        xcoords:
            list or 1D-array of x coordinate values.
        ycoords:
            list or 1D-array of y coordinate values.
        in_grid:
            The input gridded dataset to sample from, either a raster or multidimensional/multiband dataset.
        pnt_index: optional
            An input list or 1D-array of index labels to override the default incremental index.
        output: optional
            The type of output to return as a string:
                - 'pandas_long' is a long-form dataframe returned (default)
                - 'pandas_multi' is a multiindex dataframe
                - 'xarray' returns a xarray dataset

    Returns:
        The raster values at the input coordinate locations, indexed by an incremental integer or the pnt_index
        argument.

    Raises:
        ValueError: If the pnt_index argument is not a list or numpy array
        ValueError: If the output argument is not one of ['pandas_long', 'pandas_multi', 'xarray']
    """
    if isinstance(in_grid, RasterClass):
        raster = in_grid
    else:
        raster = RasterClass.load(in_grid)

    bands = raster.values.shape[1]
    if raster.band_idx_labels is None:
        band_idx_labels = np.arange(1, bands + 1)
    else:
        band_idx_labels = raster.band_idx_labels

    if raster.dim_names is None:
        band_nm = 'Band'
    else:
        band_nm = raster.dim_names[0]

    if pnt_index is None:
        pnt_index = np.arange(0, len(xcoords))
    elif isinstance(pnt_index, (list, np.ndarray)):
        pass
    else:
        raise ValueError("The input point indexes are not recognized as list-like or 1D numpy array.")

    rowcol_ids = rowcol(raster.transform, xcoords, ycoords)
    point_values = raster.values[:, :, rowcol_ids[0], rowcol_ids[1]]
    df_dict = {'FID': np.tile(pnt_index, band_idx_labels.size), band_nm: band_idx_labels.repeat(len(pnt_index))}

    if isinstance(raster.variables, str):
        df_dict[raster.variables] = point_values[0, :, :].ravel()
    else:
        for i in range(len(raster.variables)):
            df_dict[raster.variables[i]] = point_values[i, :, :].ravel()

    # Needs to be structured as multiindex dframe with band and variables such that we are sampling
    #   across all dimensions
    out = pd.DataFrame(
        df_dict
    )

    if output == 'pandas_long':
        return out
    elif output == 'pandas_multi':
        out = out.set_index(['FID', band_nm]).sort_index()
        return out
    elif output == 'xarray':
        out = out.set_index(['FID', band_nm]).sort_index()
        out = out.to_xarray()
        return out
    else:
        raise ValueError("The output argument is not recognized. Please choose 'pandas_long', "
                         "'pandas_multi', or 'xarray.'")
