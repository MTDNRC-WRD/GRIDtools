## Zonal operations on gridded data using a geometry
# CNB968 - Todd Blythe

import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


def calc_zonal_stats(in_geom, in_grid, **kwargs):
    # get geometry in same reference system
    geom = gpd.read_file(in_geom)
    with rio.open(in_grid) as src:
        affine = src.transform
        array = src.read(1)
        crs = src.crs
        nodata = src.nodata
    array[array == nodata] = np.nan

    # check crs
    if geom.crs.to_authority() == crs.to_authority():
        pass
    else:
        geom = geom.to_crs(crs)

    zs = zonal_stats(geom.geometry, array, affine=affine, **kwargs)
    fgd = geom.join(pd.DataFrame(zs))

    # return GeoDataFrame with stats added as additional attributes
    return fgd

def grid_area_weighted_volume(dataset, in_geom, geom_id_col=None, save_shp_to_file=False, out_fp=None):
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

    # get grid resolution
    xres, yres = dataset.rio.resolution()

    # get the array bounds as variables for new geometry
    gminx, gminy, gmaxx, gmaxy = dataset.rio.bounds()

    # shape of dataset
    nrows = dataset.shape[1]
    ncols = dataset.shape[2]

    nshp_cols = list(np.linspace(gminx, gmaxx, ncols+1))
    nshp_rows = np.linspace(gminy, gmaxy, nrows+1)
    nshp_rows = np.flip(nshp_rows)

    grid_polygons = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            grid_polygons.append(Polygon([(x,y), (x+xres,y), (x+xres, y+yres), (x, y+yres)]))

    grid_polys = gpd.GeoDataFrame(geometry=grid_polygons, crs=4326)

    g_proj = in_geom.to_crs(5071)
    grd_proj = grid_polys.to_crs(5071)

    # get in shape areas in Km^2
    ingeom_areas = r_proj.area.values / (1000 ** 2)

    intersection_geoms = []
    in_shape_id = []
    grid_cell_id = []
    for i, g1 in enumerate(g_proj.geometry.values):
        for j, g2 in enumerate(grd_proj.geometry.values):
            if g1.intersects(g2):
                igeom = g1.intersection(g2)
                intersection_geoms.append(igeom)
                if geom_id_col is None:
                    in_shape_id.append(i)
                elif geom_id_col in in_geom.columns.to_list():
                    in_shape_id.append(in_geom[geom_id_col].iloc[i])
                else:
                    raise ValueError("Argument for geom_id_col not found in input geometry columns.")
                grid_cell_id.append(j)

    clipped_grid_shp = gpd.GeoDataFrame({
        "GridID": grid_cell_id,
        "FeatureID": in_shape_id
    },
        geometry=intersection_geoms,
        crs=5071,
    )
    clipped_grid_shp['CellArea_sqm'] = clipped_grid_shp.geometry.area

    # Loop through unique feature ID's calculate the weighted volume for each
    vol_series = []
    for gid in in_geom[geom_id_col].to_list():
        qarea = pd.DataFrame(clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid]['CellArea_sqKm'],
                             index=clipped_grid_shp.loc[clipped_grid_shp['FeatureID'] == gid].index)
        allcells = pd.concat([qarea, grid_polys], axis=1)
        area_arry = allcells['CellArea_sqKm'].values.reshape(nrows, ncols)
        md_arry = dataset.values
        vol_grd = area_arry * md_arry
        vol_v = np.nansum(vol_grd, axis=(1, 2))
        vol_v = vol_v.reshape(vol_v.size, 1)
        vol_series.append(vol_v)

    # export shapefile if desired
    if save_shp_to_file:
        if out_fp is None:
            raise ValueError("Missing out_fp argument string.")
        else:
            grid_shp.to_crs(4326).to_file(Path(out_fp) / (dataset.name + '_clipped_grid.shp'))

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