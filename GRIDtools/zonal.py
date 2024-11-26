## Zonal operations on gridded data using a geometry
# CNB968 - Todd Blythe

import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import numpy as np
import pandas as pd


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

def grid_area_weighted_volume(dataset, in_geom, out_fp, save_shp_to_file=False):
    """
    Takes raster or multidimensional (.nc) datasets and input geometry in the same coordinate reference
    system and returns an area weighted volume from depth valued datasets (e.g., precip).
    :param dataset: an xarray dataset or dataarray instance, must have defined spatial_ref and be in units of meters,
    time dimension/coords must be labeled 'time'
    :param in_geom: a GeoDataFrame of input shapefile/geometry in the same spatial_ref as xarray data
    :param out_fp: the filepath to output .csv data and shapefile if specified
    :param save_shp_to_file: boolean, default = False, if True will save the grid shapefile to the out filepath
    :return: creates .csv files of volume for the input vector geometries based on the gridded data
    """
    # output path
    out_file_path = Path(out_fp)
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

    polygons = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            polygons.append(Polygon([(x,y), (x+xres,y), (x+xres, y+yres), (x, y+yres)]))

    intersection_areas = []
    main_shape_id = []
    intersecting_shape_id = []
    for i, g1 in enumerate(in_geom.geometry.values):
        for j, g2 in enumerate(polygons):
            if g1.intersects(g2):
                intersection_areas.append(g1.intersection(g2).area)
                main_shape_id.append(i)
                intersecting_shape_id.append(j)

    s_areas = pd.DataFrame({'Cell_Area': intersection_areas}, index=intersecting_shape_id)
    polys = pd.DataFrame({'geometry': polygons})

    # Create shapefile
    grd_shp = gp.GeoDataFrame(pd.concat([s_areas, polys], axis=1))
    grd_shp.set_crs(in_geom.crs.to_epsg(), inplace=True)
    grd_shp.sort_index(inplace=True)
    grd_shp['GID'] = grd_shp.index

    if save_shp_to_file:
        grd_shp.to_file(out_file_path / (dataset.name + '_grid.shp'))

    area_arry = grd_shp['Cell_Area'].values.reshape(nrows, ncols)
    md_arry = dataset.values
    vol_grd = area_arry * md_arry
    vol_v = np.nansum(vol_grd, axis=(1,2))

    if str(dataset.indexes['time'].dtype) != 'datetime64[ns]':
        dttmidx = dataset.indexes['time'].to_datetimeindex()
    else:
        dttmidx = pd.DatetimeIndex(dataset.indexes['time'])

    VOL_SRS = pd.DataFrame({dataset.name + ' volume (cubic meters)': vol_v}, index=dttmidx)
    VOL_SRS.to_csv(out_file_path / (dataset.name + ' Weighted Area Volume_WY{0}.csv'.format(VOL_SRS.index[-1].year)))