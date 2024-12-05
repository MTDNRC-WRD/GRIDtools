"""
Author - Todd Blythe, CNB968

Script to perform watershed delineation actions.
"""

import pynhd
from shapely.geometry import Point


def delineate_watershed(coords):
    """
    Function that simplifies the work flow of pynhd split_catchment to delineate
    a basin upstream of a list of coordinates.
    :param coords: geopandas.GeoDataFrame - must contain point geometries.
    :return:
    """

    if (coords.geom_type != 'Point').any():
        raise ValueError("The input geometry(s) are not all type Polygon. Only Polygons are supported.")

    trace_attrs = ['req_idx', 'gnis_name', 'comid', 'reachcode', 'intersection_point']
    # add the required columns for pynhd trace_flow and split_catchment
    coords['direction'] = ['up'] * coords.shape[0]
    coords['upstream'] = [True] * coords.shape[0]
    # use flow trace service to find nearest stream feature
    trace = pynhd.pygeoapi(coords, "flow_trace")
    # add index as column for merging attribute fields and retain all but the geometry
    coords['req_idx'] = coords.index
    orig_attrs = coords.columns[coords.columns != 'geometry']
    # update point geometries to reflect hydro-adjusted points
    coords.geometry = [Point(x) for x in trace['intersection_point']]
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