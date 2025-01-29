## Zonal operations on gridded data using a geometry
# CNB968 - Todd Blythe

import geopandas as gpd
import rasterio as rio
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from shapely.geometry import Polygon

from utils import vectorize_grid, RasterClass


def calc_zonal_stats(in_geom, in_grid, method='groupby', **kwargs):
    """
    Uses various methods to calculate zonal statistics for set of geometries. This method rasterizes the geometries
    to calculate zonal statistics.

    Args:
        in_geom:
        in_grid:
        **kwargs:

    Returns:
        GeoDataFrame:
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

    if method == 'rasterstats':
        from rasterstats import zonal_stats

        # Need to add loop to deal with multiple bands
        zs = zonal_stats(geom.geometry, raster.values, affine=raster.transform, **kwargs)
        fgd = geom.join(pd.DataFrame(zs))
    elif method == 'groupby':

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
            "precip_volume": (
            ['time', 'location'], np.hstack(vol_series), {'standard_name': 'Area Weighted Precipitation Volume',
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