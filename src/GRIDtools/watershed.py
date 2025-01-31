"""
Author - Todd Blythe, CNB968

Script to perform watershed delineation actions.
"""

import pynhd
import pandas as pd
from shapely.geometry import Point
from pynhd.exceptions import ServiceError

from GRIDtools.utils import jitter_geometries

def delineate_watershed(in_coords):
    """
    Function that simplifies the work flow of pynhd split_catchment to delineate
    a basin upstream of a list of coordinates.
    :param coords: geopandas.GeoDataFrame - must contain point geometries.
    :return:
    """
    coords = in_coords.copy()

    if (coords.geom_type != 'Point').any():
        raise ValueError("The input geometry(s) are not all type Points. Only Points are supported.")

    if coords.crs is None:
        raise ValueError("The input GeoDataFrame does not have a CRS.")

    if '4326' not in coords.crs.to_authority():
        coords = coords.to_crs(4326)

    trace_attrs = ['req_idx', 'gnis_name', 'comid', 'reachcode', 'intersection_point']
    # add the required columns for pynhd trace_flow and split_catchment
    coords.loc[:,'direction'] = 'up'
    coords.loc[:,'upstream'] = True
    # use flow trace service to find nearest stream feature
    # first make multiple attempts, changing geometry if necessary
    attempts = 0
    tol = 0.001
    while attempts < 5:
        try:
            if attempts == 0:
                mod_coords = coords
            else:
                mod_coords = jitter_geometries(coords, tol)
            trace = pynhd.pygeoapi(mod_coords, "flow_trace")
            break
        except ServiceError:
            if attempts == 0:
                tol = tol
            else:
                tol *= 3
            attempts += 1
            if attempts < 4:
                print("A ServiceError occurred with 'flow_trace', will try altering input geometries.")
            else:
                print("A ServiceError occurred with 'flow_trace', a suitable input geometry was not found.")

    # check if all features returned
    if len(trace.index) != len(coords.index):
        print("Not all points returned valid traces, will try altering geometries.")
        attempts = 0
        tol = 0.001
        while attempts < 5:
            missed_pnts = coords.loc[~coords.index.isin(trace['req_idx']),:]
            try:
                mod_coords = jitter_geometries(missed_pnts, tol)
                miss_trace = pynhd.pygeoapi(mod_coords, "flow_trace")
                trace = pd.concat([trace, miss_trace])
                if len(trace.index) == len(coords.index):
                    break
                else:
                    if attempts == 4:
                        print("Could not find suitable geometries for all missing points, will continue without them.")
                    else:
                        print("Trying again to resolve all points...")
            except ServiceError:
                if attempts == 0:
                    tol = tol
                else:
                    tol *= 3
                attempts += 1
                if attempts < 4:
                    print("A ServiceError occurred with 'flow_trace', will try altering input geometries.")
                else:
                    print("A ServiceError occurred with 'flow_trace', a suitable input geometry was not found.")

    # add index as column for merging attribute fields and retain all but the geometry
    coords.loc[:,'req_idx'] = coords.index
    orig_attrs = coords.columns[coords.columns != 'geometry']
    # update point geometries to reflect hydro-adjusted points
    coords.loc[:,'geometry'] = [Point(x) for x in trace['intersection_point']]
    # delineate watersheds with adjusted points
    watersheds = pynhd.pygeoapi(coords, "split_catchment")

    # separate multi-part features and make valid
    watersheds = watersheds.explode()
    watersheds.geometry = watersheds.make_valid()
    watersheds = watersheds.explode()
    # remove the local catchment and keep only the full delineated area
    watersheds['Area_sqKm'] = watersheds.to_crs('5071').area / (1000 ** 2)
    watersheds.sort_values(by='Area_sqKm', inplace=True)
    watersheds.drop_duplicates(subset='req_idx', keep='last', inplace=True)
    watersheds.reset_index(drop=True, inplace=True)

    # construct final GeoDataFrame with relevant attributes
    watersheds = watersheds.merge(coords[orig_attrs], on='req_idx')
    watersheds = watersheds.merge(trace[trace_attrs], on='req_idx')

    return watersheds