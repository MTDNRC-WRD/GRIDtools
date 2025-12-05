import subprocess

gf = 'D:/ArcGIS_Projects/Yellowstone/Upper Yellowstone/prms/UY_prms_grid.shp'
rf = 'D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/insolpy_cfgrid.tif'
o = 'D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/'

env_p = 'C:/Users/CNB968/.conda/envs/GRIDtools/python.exe'
scrpt = 'C:/Users/CNB968/OneDrive - MT/GitHub/GRIDtools/scripts/zonal.py'

subprocess.run([env_p, scrpt, gf, rf, o, 'mean', 'min', 'max', '-all'])
