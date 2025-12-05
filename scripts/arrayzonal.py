"""
This script runs the calc_zonal_stats() function using the groupby method. It accepts arguments to initialize a
raster from an array (passed as a flat list argument with dimensions).

"""
import sys
import os
from pathlib import Path

sys.path.append((Path(__file__).parents[1] / 'src').as_posix())
from GRIDtools.zonal import calc_zonal_stats
#from src.GRIDtools.utils import RasterClass

if __name__ == '__main__':
    print("success")