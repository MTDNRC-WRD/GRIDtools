"""
This script runs the calc_zonal_stats() function. It accepts arguments from file as input.

"""
import argparse
from pathlib import Path
import sys

sys.path.append((Path(__file__).parents[1] / 'src').as_posix())
from GRIDtools.zonal import calc_zonal_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate zonal stats for a shapefile of polygons')
    parser.add_argument('gf', help='file path to geometry shapefile')
    parser.add_argument('rf', help='file path to raster dataset')
    parser.add_argument('o', help='output directory')
    parser.add_argument('s', nargs='+', help='statistics to compute for each geometry')
    parser.add_argument('-all', action='store_true', help='whether to include all raster cells that touch a geometry')
    args = parser.parse_args()

    if args.all:
        allt = True
    else:
        allt = False
    zn = calc_zonal_stats(Path(args.gf), Path(args.rf), stats=args.s, all_touched=allt)
    for stat in args.s:
        s = zn.loc[(slice(None), slice(None), [stat, 'single_point']),:].reset_index().drop(columns=['FID','Band','stat'])
        s.index.name = 'FID'
        s.to_csv(Path(args.o) / f"zonal_{stat}.csv")