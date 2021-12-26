import logging
import math
import os
from configparser import SafeConfigParser
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import tobac
import xarray as xr
from scipy.spatial import ConvexHull


class Config(dict):
    def __init__(self, filename):
        ''' Reads in the settings.txt file
        Example 'settings.txt':
        '''

        curr_dir = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
        config = SafeConfigParser(inline_comment_prefixes=';')
        config.read(filename)
        num_keys = ['lat_min', 'delta_time',
                    'min_threshold', 'max_threshold', 'step_threshold',
                    'n_workers', 'threads_per_worker']
        for key, value in config.items('ALL'):
            if 'dir' in key:
                self[key] = os.path.join(curr_dir, value)
            elif key in num_keys:
                self[key] = float(value)
            else:
                self[key] = value


def validate_date(date_in):
    '''Validate input date'''
    try:
        datetime.strptime(date_in, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def validate_path(path_in, var_name):
    '''Validate input path'''
    if not isinstance(path_in, str):
        raise ValueError('{} must be a string'.format(var_name))
    elif not os.path.isdir(path_in):
        os.makedirs(path_in)
        # raise ValueError('{} ({}) does not exist'.format(var_name, path_in))


def PointsInCircum(eachPoint, r, n=100):
    '''Check points in Circum'''
    return [(eachPoint[0] + math.cos(2*math.pi/n*x)*r,eachPoint[1] + math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]


def bufferPoints(inPoints, stretchCoef, n):
    '''Stretch the points to a larger area'''
    # https://stackoverflow.com/questions/33831516/incrementing-area-of-convex-hull
    newPoints = []
    for eachPoint in inPoints:
        newPoints += PointsInCircum(eachPoint, stretchCoef, n)
    newPoints = np.array(newPoints)
    newBuffer = ConvexHull(newPoints)

    return newPoints[newBuffer.vertices]


def get_large_hull(hulls, stretchCoef=1.2):
    '''Get a larger hull in case some fires are nearby'''
    large_hull = []
    for hull in hulls:
        stretchCoef = stretchCoef
        pointsStretched = bufferPoints(hull.points, stretchCoef, n=10)
        # ring = LinearRing(list(zip(pointsStretched[:, 1], pointsStretched[:, 0])))
        large_hull.append(pointsStretched[:, [1,0]])

    return large_hull


def get_finite_scn(scn):
    '''Get the finite scn by drop and interpolation'''
    # convert the no2 unit
    no2_vcd = (scn['nitrogendioxide_tropospheric_column']*scn['nitrogendioxide_tropospheric_column'].attrs['multiplication_factor_to_convert_to_molecules_percm2'])

    # assign the coords of x and y, which can be used to filter the lightning no2 later
    no2_vcd = no2_vcd.assign_coords(y=range(len(no2_vcd.y)), x=range(len(no2_vcd.x)))

    # drop nan values
    no2_part_finite = no2_vcd.where(np.isfinite(no2_vcd), drop=True)

    # because segmentation don't permit nan values, we can interpolate data first and set nan to nan at the end
    no2_interp = no2_part_finite.rio.write_crs(4326).rio.write_nodata(np.nan).rio.interpolate_na(method='linear')

    # shrink by one pixel in x dimension in case rioxarray doesn't work well at the boundary
    no2_interp = no2_interp.isel(x=slice(1, -1))

    # get the array full of finite value
    finite_mask = xr.ufuncs.isfinite(no2_interp).all(dim='x')
    no2_finite = no2_interp.where(finite_mask, drop=True).expand_dims('time')

    return no2_finite


def feature(threshold, target='maximum',
            position_threshold='weighted_diff',
            coord_interp_kind='nearest',
            sigma_threshold=0.5,
            min_distance=0,
            n_erosion_threshold=0,
            n_min_threshold=50):
    '''Set keyword arguments for the feature detection step'''
    parameters_features = {}
    parameters_features['target'] = target

    # diff between specific value and threshold for weighting when finding the center location (instead of just mean lon/lat)
    parameters_features['position_threshold'] = position_threshold

    # we want to keep the original x/y instead of interpolated x/y
    # https://github.com/climate-processes/tobac/pull/51
    parameters_features['coord_interp_kind'] = coord_interp_kind

    # for slightly smoothing (gaussian filter)
    parameters_features['sigma_threshold'] = sigma_threshold

    # Minumum number of cells above/below threshold in the feature to be tracked
    # parameters_features['min_num'] = 4

    # K, step-wise threshold for feature detection
    parameters_features['threshold'] = threshold

    # minimum distance between features
    parameters_features['min_distance'] = min_distance

    # pixel erosion (for more robust results)
    parameters_features['n_erosion_threshold'] = n_erosion_threshold

    # minimum number of contiguous pixels for thresholds
    parameters_features['n_min_threshold'] = n_min_threshold

    return parameters_features


def segmentation(threshold, target='maximum', method='watershed'):
    '''Set keyword arguments for the segmentation step'''
    parameters_segmentation = {}
    parameters_segmentation['target'] = target
    parameters_segmentation['method'] = method
    # until which threshold the area is taken into account
    parameters_segmentation['threshold'] = threshold

    return parameters_segmentation


def feature_mask(no2_finite, min_threshold=4e14, max_threshold=1e15, step_threshold=2e14):
    '''Detect the features and get the masks using tobac'''
    # set the grid size to 5 km because the resolution of pixel is 3.6 km * 5.6 km
    dxy = 5000  # data resolution; Unit: m

    # calculate the range of threshold
    threshold = np.arange(min_threshold, max_threshold, step_threshold)  # multi-thresholds for tracking

    # set parameters_features
    parameters_features = feature(threshold)

    # drop the x and y coordinates and assign at the end
    # this is related to some bug in tobac, Xin just uses this trick to escape ...
    x_coord = no2_finite.coords['x']
    y_coord = no2_finite.coords['y']
    no2_finite = no2_finite.drop_vars(['x', 'y'])

    # get features
    features = tobac.themes.tobac_v1.feature_detection_multithreshold(no2_finite, dxy, **parameters_features)

    # set parameters_segmentation
    parameters_segmentation = segmentation(np.min(threshold))

    # get masks and paired features
    if features is not None:
        masks_no2, features_no2 = tobac.themes.tobac_v1.segmentation(features, no2_finite, dxy, **parameters_segmentation)

        masks_no2 = masks_no2.where(masks_no2>0).rename({'dim_0': 'y', 'dim_1': 'x'})
        masks_no2 = masks_no2.assign_coords({'x': x_coord, 'y':y_coord})

        return masks_no2
    else:
        return None


# def predict_loc(data, lon, lat):
def predict_loc(lon, lat, wdir, wspd, wdelta):
    # https://stackoverflow.com/a/7835325/7347925
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    R = 6378.1  # Radius of the Earth
    brng = np.deg2rad(wdir)  # Bearing is radians.
    d = wspd*wdelta/1e3  # Distance in km

    lat2 = math.asin(math.sin(lat)*math.cos(d/R) + math.cos(lat)*math.sin(d/R)*math.cos(brng))

    lon2 = lon + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat),
                math.cos(d/R)-math.sin(lat)*math.sin(lat2))

    lat2 = np.rad2deg(lat2)
    lon2 = np.rad2deg(lon2)

    return lon2, lat2


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    https://stackoverflow.com/a/16898636/7347925
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def convert_cluster(scn, df, lightning_mask, kind='clean'):
    '''Convert the lightning cluster DataFrame to Dataset which has the same shape of NO2'''
    # rename the label
    df = df.rename(columns={'label':'lightning_label'})
    # set label as index which is the coordinate of xarray Dataset
    df.set_index(['lightning_label'], inplace=True)
    # convert to Dataset
    ds = concat_pred(df.to_xarray())

    # get the pixel center points
    pixel_points = np.vstack((scn['nitrogendioxide_tropospheric_column'].longitude.stack(z=('x', 'y')),
                              scn['nitrogendioxide_tropospheric_column'].latitude.stack(z=('x', 'y')))
                              ).T

    labels = list(set(ds.lightning_label.values))

    for index, label in enumerate(labels):
        # iterate each label and update the mask
        dfq = ds.sel(lightning_label=label)
        lon_lat = dfq[['longitude_pred', 'latitude_pred']].stack(all=("level", "lightning_label"))
        lightning_points = lon_lat.to_array().transpose('all', ...)

        mask = in_hull(pixel_points, lightning_points).reshape(scn['nitrogendioxide_tropospheric_column'].shape, order='F')

        # get the overplapped label
        overlapped_label = np.delete(np.unique(lightning_mask.where(xr.DataArray(mask, dims=['y', 'x']), 0)), 0)

        if len(overlapped_label) == 0:
            if kind == 'clean':
                # clean lightning 1abel: 1, 2, ....
                lightning_mask = xr.where(mask, index+1, lightning_mask)
            elif kind == 'polluted':
                # polluted lightning 1abel: -1, -2, ....
                lightning_mask = xr.where(mask, -index-1, lightning_mask)
        elif len(overlapped_label) == 1:
            # set to the only overlapped value
            if kind == 'clean':
                lightning_mask = xr.where(mask, overlapped_label[0], lightning_mask)
            elif kind == 'polluted':
                lightning_mask = xr.where(mask, overlapped_label[0], lightning_mask)
        else:
            # assign minimum label to related labels
            min_label = np.min(overlapped_label)
            lightning_mask = xr.where(mask, min_label, lightning_mask)
            for rest_label in np.delete(overlapped_label, min_label):
                lightning_mask = lightning_mask.where(lightning_mask==rest_label, min_label)

    return ds, lightning_mask.rename('lightning_mask')


def concat_pred(ds):
    '''Concatenate predicted lon and lat by pressure level'''
    # get the DataArray names of predicted lon and lat
    lon_names = [var for var in ds.data_vars if 'longitude_pred' in var]
    lat_names = [var for var in ds.data_vars if 'latitude_pred' in var]
    levels = [int(name.split('_')[-1]) for name in lon_names]

    # concatenate into one array
    ds['longitude_pred'] = ds[lon_names].to_array(dim='lon_level', name='longitude_pred').assign_coords(lon_level=levels).rename({'lon_level':'level'})
    ds['latitude_pred'] = ds[lat_names].to_array(dim='lat_level', name='longitude_pred').assign_coords(lat_level=levels).rename({'lat_level':'level'})
    ds.coords['level'].attrs['units'] = 'hPa'

    # add description and units
    ds['longitude_pred'].attrs['description'] = 'Longitude of lightning at different pressure levels predicted by ERA5 data'
    ds['latitude_pred'].attrs['description'] = 'Latitude of lightning at different pressure levels predicted by ERA5 data'

    # drop useless variables
    ds = ds.drop_vars(lon_names+lat_names)

    return ds
