from pathlib import Path

import rasterio as rio
import xarray as xr
import rioxarray

from src.GRIDtools.utils import RasterClass

# test from file
rstr_file = Path(r'D:\ArcGIS_Projects\Yellowstone\Upper Yellowstone\prms\MT_hydro_SRTM_30m_clipped.tif')
rc = RasterClass.load(rstr_file)

# test from array and metadata
with rio.open(rstr_file) as ds:
    values = ds.read()
    crs = ds.crs
    transform = ds.transform
    bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    res = ds.res

rc = RasterClass(
    values,
    crs,
    transform,
    bounds,
    res
)

# with explicit naming overriding defaults
rc = RasterClass(
    values,
    crs,
    transform,
    bounds,
    res,
    var_names='Elevation',
    band_labels=[1],
    dim_names=['band','y','x']
)

# test multidimensional from file
nc_file = Path('F:/BOR_UYWS_2025/OpenET_monthly_ensemble_mm_2020_2024/UY_ET_ensemble_monthly_mm_2020_2024.nc')
rc = RasterClass.load(nc_file) # this worked...but this is a huge file so not recommended to work with 17GB

# test GridMET multidimensional with several variables
gmet_file = Path('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/Gridmet_inputs_2020_2024.nc')
rc = RasterClass.load(gmet_file)
# this worked, you can see the multidimensional array, the variable names were successfully mapped as well as the
# shape of the array with axis=0 size of 3 (for 3 variables)
print(rc.variables)
print(rc.values.shape)

xds = xr.open_dataset(gmet_file)
# now try with an xarray Dataset...
rc = RasterClass.load(xds)