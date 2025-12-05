from pathlib import Path
import os

from src.GRIDtools import calc_zonal_stats
from src.GRIDtools.config import runconfig

config_fp = Path("./test/inputs.toml")
config = runconfig(config_fp)

out = calc_zonal_stats(Path(config['zonal_inputs']['in_geom']),
                       Path(config['zonal_inputs']['in_raster']),
                       stats=config['zonal_inputs']['stats']
                       )

out.rename(columns={'min': 'slope_min',
                    'max': 'slope_max',
                    'mean': 'slope_mean',
                    'count': 'slope_count',
                    'std': 'slope_std',
                    'median': 'slope_med',
                    'percentile_25': 'slope_P25',
                    'percentile_75': 'slope_P75',
                    'nodata': 'slope_nodata'}, inplace=True)

out.to_file(Path(config['zonal_inputs']['outfile']))

import geopandas as gpd
flwlns = gpd.read_file(Path('D:/Python Projects/GRIDtools/sample_data/R5_NHD_PROSPER_HUC12.zip'))
keep = ['slope_min', 'slope_max', 'slope_mean', 'slope_count', 'slope_std', 'slope_med', 'slope_P25', 'slope_P75', 'slope_nodata', 'geometry']
slpclms = ['slope_min', 'slope_max', 'slope_mean', 'slope_count', 'slope_std', 'slope_med', 'slope_P25', 'slope_P75', 'slope_nodata', 'index_right']
nw_out = out[keep]

sjF = flwlns.sjoin(nw_out, how='left', predicate='within')
dtid = sjF['slope_med'][~sjF['slope_med'].isna()].index
naid = sjF['slope_med'][sjF['slope_med'].isna()].index
good = sjF.loc[dtid]
nan = sjF.loc[naid].drop(columns=slpclms)
nansj = nan.sjoin(nw_out, how='left')
nansj['ID'] = nansj.index
nansjF = nansj.sort_values(by='slope_med').drop_duplicates('ID', keep='last')
nansjF = nansjF.sort_values(by='ID').drop(columns='ID')
FGDF = pd.concat([good, nansjF]).sort_index()

FGDF.to_file(Path('D:/Python Projects/GRIDtools/sample_data/R5_NHD_PROSPER_SLOPE_zonal.shp'))

if __name__ == '__main__':