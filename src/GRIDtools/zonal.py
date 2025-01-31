## Zonal operations on gridded data using a geometry
# CNB968 - Todd Blythe

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from rasterio.transform import rowcol
from shapely.geometry import Polygon
import warnings

from utils import vectorize_grid, RasterClass


def calc_zonal_stats(in_geom,
                     in_grid,
                     method='groupby',
                     stats='mean',
                     all_touched=False,
                     output='pandas',
                     **kwargs):
    """
    Uses various methods to calculate zonal statistics for set of geometries. This method rasterizes the geometries
    to calculate zonal statistics.

    Args:
        in_geom (str | pathlib.Path | geopandas.GeoDataFrame):
            input file path or GeoDataFrame of geometries to summarize.

        in_grid (str | pathlib.Path | rasterio.DatasetReader | xarray.DataArray | xarray.Dataset):
            the input gridded dataset to summarize over the input geometries.

        method (str): optional
            'groupby' or 'rasterstats' the default is 'groupby'

        stats (str | list): optional
            The name of the statistic to use for zonal stats, or list of valid statistics - default 'mean.'
            These differ based on the method argument.
            If method == 'rasterstats' then options are any string accepted by that package. See accepted statistics
            strings here: https://pythonhosted.org/rasterstats/manual.html#statistics

            If method == 'groupby' (the default), any statistics function strings accepted by pandas SeriesGroupBy
            descriptive stats. Not all functions may be represented by a string name but their series method can be
            used in the stats argument (e.g., stats=['mean', 'count', pd.Series.mode] where pd.Series.mode returns the
            majority value within the geometry.

        all_touched (bool): optional
            whether to include all cells intersected by each geometry (True) or only those with center points within
            each geometry (False) - default is False

        output (str): optional
            Specifies what type to return, either 'pandas' for pandas.DataFrame or 'xarray' for xarray.Dataset

        **kwargs:
            additional key word arguments accepted by rasterstats package

    Returns:
        pandas.DataFrame | xarray.Dataset:
            A multiindex/dimensional DataFrame or dataset with the summary statistics for
            each input geometry and all raster bands/dimensions
    """
    # get geometry in same reference system
    if isinstance(in_geom, (str, Path)):
        in_geom = gpd.read_file(in_geom)

    geom = in_geom
    raster = RasterClass(in_grid)

    # check crs
    if geom.crs.to_authority() == raster.crs.to_authority():
        pass
    else:
        geom = geom.to_crs(raster.crs)

    #TODO - add checks for valid stats
    if isinstance(stats, str):
        stats = [stats]
    elif isinstance(stats, list):
        pass
    else:
        raise ValueError("stats argument must be either valid string or list of accepted methods")

    if method == 'rasterstats':
        from rasterstats import zonal_stats

        if kwargs.get('layer') is not None:
            layer = kwargs.get('layer')
        else:
            layer=0

        if kwargs.get('band') is not None:
            band = kwargs.get('band')
        else:
            band=1

        if kwargs.get('categorical') is not None:
            categorical=kwargs.get('categorical')
        else:
            categorical=False

        if kwargs.get('raster_out') is not None:
            raster_out=kwargs.get('raster_out')
        else:
            raster_out=False

        if kwargs.get('geojson_out') is not None:
            geojson_out=kwargs.get('geojson_out')
        else:
            geojson_out=False

        if kwargs.get('boundless') is not None:
            boundless=kwargs.get('boundless')
        else:
            boundless=True

        # Need to add loop to deal with multiple bands
        zs = zonal_stats(geom.geometry,
                         raster.values,
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
        def rasterized_to_df(rasterized_features, in_raster_values, band_idx, stats):
            adj_rzd = rasterized_features - 1
            rstrzd_df = pd.DataFrame(
                {'FID': np.tile(adj_rzd.ravel(), band_idx.size), 'Band': band_idx.repeat(adj_rzd.ravel().size),
                 'Value': in_raster_values.ravel()})
            filtered_df = rstrzd_df.loc[rstrzd_df.FID != -1, :]
            grouped = filtered_df.groupby(['FID', 'Band']).agg({'Value': stats})
            Fstck = grouped.stack(level=1, future_stack=True)
            new_names = list(Fstck.index.names)[0:2] + ['stat']
            ret_df = Fstck.rename_axis(index=new_names)

            return ret_df

        bands = raster.values.shape[0]
        if raster.band_idx is None:
            band_idx = np.arange(1, bands + 1)
        else:
            band_idx = raster.band_idx

        geom_value = ((geom, value) for geom, value in zip(geom.geometry, geom.index + 1))
        rasterized = rasterize(
            geom_value,
            out_shape=(raster.values.shape[1], raster.values.shape[2]),
            fill=0,
            transform=raster.transform,
            all_touched=all_touched,
            dtype=np.int64,
            merge_alg=MergeAlg.replace
        )

        result_df = rasterized_to_df(rasterized, raster.values, band_idx, stats)

        n_rstrzed = np.unique(rasterized)
        gids = n_rstrzed[1:] - 1
        if len(gids) != len(geom.index):
            warnings.warn(
                " ".join(
                    [
                        f"Not all geometries were returned, {len(geom.index) - len(n_rstrzed[1:])} geometries were missed during rasterize.",
                        "Attempting to rasterize missed geometries..."
                    ]
                ),
                UserWarning,
                stacklevel=2
            )

            missed_polys = geom.loc[~geom.index.isin(gids), :]

            geom_value = ((geom, value) for geom, value in zip(missed_polys.geometry, missed_polys.index + 1))
            m_rasterized = rasterize(
                geom_value,
                out_shape=(raster.values.shape[1], raster.values.shape[2]),
                fill=0,
                transform=raster.transform,
                all_touched=all_touched,
                dtype=np.int64,
                merge_alg=MergeAlg.replace
            )

            m_result_df = rasterized_to_df(m_rasterized, raster.values, band_idx, stats)
            result_df = pd.concat([result_df, m_result_df])

            n_rstrzed = result_df.index.get_level_values(0).unique()
            if len(n_rstrzed) != len(geom.index):
                print(
                    "Second attempt did not return all geometries, defaulting to point locations for remaining geometries.")

                missed_polys = geom.loc[~geom.index.isin(n_rstrzed), :]
                cntrs = missed_polys.centroid
                rowcol_ids = rowcol(raster.transform, cntrs.x.to_list(), cntrs.y.to_list())
                point_values = raster.values[:, rowcol_ids[0], rowcol_ids[1]]
                point_rasterized = missed_polys.index.values
                pnt_df = rasterized_to_df(point_rasterized + 1, point_values, band_idx, stats)

                result_df = pd.concat([result_df, pnt_df])

        if len(result_df.index.get_level_values(0).unique()) != len(geom.index):
            warnings.warn(
                " ".join(
                    [
                        f"Point locations failed to capture all geometries,",
                        "{len(geom.index) - len(result_df.index.get_level_values(0).unique())} unrepresented geometries",
                        "will be skipped..."
                    ]
                ),
                UserWarning,
                stacklevel=2
            )

        fgd = result_df.sort_index(level=['FID', 'Band'])

    else:
        raise ValueError("The method argument is not recognized, please choose 'rasterstats' or 'groupby.'")
    # return GeoDataFrame with stats added as additional attributes
    return fgd


def grid_area_weighted_volume(dataset, in_geom, geom_id_col=None, data_scale=1):
    """
    Takes a multidimensional (.nc) DataArray and input polygon geometry in the same coordinate reference
    system and returns an area weighted volume for depth valued variables (e.g., precip).
    :param dataset: xarray.DataArray - must have defined spatial_ref and be in units of meters,
    time dimension/coords must be labeled 'time'
    :param in_geom: a GeoDataFrame of input shapefile/geometry in the same spatial_ref as xarray data
    :param out_fp: str - path to save shapefile if save_shp_to_file=True
    :param save_shp_to_file: boolean, default = False, if True will save the grid shapefile to the out filepath
    :return: xarray.Dataset - contains timeseries of area weighted volume for each input geometry.
    """
    if (in_geom.geom_type != 'Polygon').any():
        raise ValueError("The input geometry(s) are not all type Polygon. Only Polygons are supported.")

    rast = RasterClass(dataset)
    grid_polys = vectorize_grid(dataset)
    grid_polys.index.name = 'GridID'
    nrows = rast.values.shape[1]
    ncols = rast.values.shape[2]

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
            raise ValueError("Argument for geom_id_col not found in input geometry columns.")

        clips.append(clpd)

    clipped_grid_shp = pd.concat(clips)
    clipped_grid_shp['CellArea_sqm'] = clipped_grid_shp.geometry.area
    clipped_grid_shp.index.name = 'GridID'

    # Loop through unique feature ID's calculate the weighted volume for each
    vol_series = []
    for gid in in_geom[geom_id_col].to_list():
        qarea = pd.DataFrame(
            {'CellArea_sqm': clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid]['CellArea_sqm'].values},
            index=clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid].index
        )
        allcells = pd.concat([qarea, grd_proj], axis=1)
        allcells.sort_index(inplace=True)
        area_arry = allcells['CellArea_sqm'].values.reshape(nrows, ncols)
        md_arry = dataset.values / data_scale
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