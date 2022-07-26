import logging
import math
import os
from configparser import SafeConfigParser
from datetime import datetime

import geopandas
import geopy
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import tobac
import xarray as xr
from geopy.distance import geodesic
from matplotlib.path import Path
from metpy.units import units
from pyresample import kd_tree
from pyresample.geometry import SwathDefinition
from scipy.interpolate import interpn
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiPoint, Point


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
    return [(eachPoint[0] + math.cos(2*math.pi/n*x)*r, eachPoint[1] + math.sin(2*math.pi/n*x)*r) for x in range(0, n+1)]


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
        large_hull.append(pointsStretched[:, [1, 0]])

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

        masks_no2 = masks_no2.where(masks_no2 > 0).rename({'dim_0': 'y', 'dim_1': 'x'})
        masks_no2 = masks_no2.assign_coords({'x': x_coord, 'y': y_coord})

        return masks_no2
    else:
        return None


def get_delta_time(begin_time, end_time):
    # because the ds is hourly data, we need to create the hourly time step
    times = np.concatenate(([begin_time.to_pydatetime().replace(tzinfo=None)],
                            pd.date_range(begin_time.ceil('h').replace(tzinfo=None),
                            end_time.floor('h').replace(tzinfo=None), freq='H').to_pydatetime(),
                            [end_time.replace(tzinfo=None).to_pydatetime()]
                            ))

    # calculate the time delta (seconds)
    time_step = [t.total_seconds() for t in np.diff(times)]

    return time_step


def calc_wind(row, coords, u, v, time_pred, lon_column, lat_column):
    '''Calculate the wspd and wdir'''
    # interpolate the u and v fields
    # clip lon and lat in case it's outside of the ERA5 file (please take care!)
    interp_lat = max(coords[1].min(), min(row[lat_column], coords[1].max()))
    interp_lon = max(coords[2].min(), min(row[lon_column], coords[2].max()))

    interp_points = [xr.DataArray(time_pred).astype('float').values, interp_lat, interp_lon]
    u_interp = interpn(coords, u, interp_points)
    v_interp = interpn(coords, v, interp_points)

    # wind speed (m/s)
    wspd = mpcalc.wind_speed(u_interp * units.meters / units.second,
                             v_interp * units.meters / units.second).magnitude

    # wind direction (degree)
    wdir = mpcalc.wind_direction(u_interp * units.meters / units.second,
                                 v_interp * units.meters / units.second,
                                 convention='to').magnitude

    return wspd, wdir


def predict_loc(row, level, coords, u, v):
    '''Predict the location using wind data'''
    # set the column names for saving predicted location
    lon_column = f'longitude_pred_{level}'
    lat_column = f'latitude_pred_{level}'

    # copy the initial location
    row[lon_column] = row['longitude']
    row[lat_column] = row['latitude']
    time_pred = row['time']

    for delta in row['time_step']:
        # interpolate the wind info from era5
        wspd, wdir = calc_wind(row, coords, u, v, time_pred, lon_column, lat_column)

        # predict the location at next time
        # https://stackoverflow.com/a/40645383/7347925
        # validation website: https://www.fcc.gov/media/radio/find-terminal-coordinates
        # this method has issue with pandarallel
        dest = geodesic(kilometers=(wspd*delta/1e3))\
            .destination(geopy.Point(row[lat_column], row[lon_column]), wdir)
        lat2 = dest[0]
        lon2 = dest[1]

        # save the predicted location
        row[lon_column] = lon2
        row[lat_column] = lat2

        # update time to the next step
        time_pred = time_pred + pd.Timedelta(seconds=delta)

        # # --- bak up: method without geopy ---
        # # https://stackoverflow.com/a/7835325/7347925
        # lon, lat = np.deg2rad(row[lon_column]), np.deg2rad(row[lat_column])

        # brng = np.deg2rad(wdir)  # Bearing is radians.
        # d = wspd*delta/1e3  # Distance in km
        # R = 6378.1  # Radius of the Earth

        # lat2 = math.asin(math.sin(lat)*math.cos(d/R) + math.cos(lat)*math.sin(d/R)*math.cos(brng))

        # lon2 = lon + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat),
        #            math.cos(d/R)-math.sin(lat)*math.sin(lat2))

        # lat2 = np.rad2deg(lat2)
        # lon2 = np.rad2deg(lon2)

    return row


def mask_label(scn, ds, lightning_mask):
    '''Generate the lightning label based on mask value'''
    # define the swaths
    swath_tropomi = SwathDefinition(lons=scn['nitrogendioxide_tropospheric_column'].longitude,
                                    lats=scn['nitrogendioxide_tropospheric_column'].latitude)
    swath_lightning = SwathDefinition(lons=ds['longitude_pred'].isel(level=0),
                                      lats=ds['latitude_pred'].isel(level=0))

    # assign the mask value as lightning label using the nearest resample
    lightning_label = kd_tree.resample_nearest(swath_tropomi, lightning_mask.values,
                                               swath_lightning, radius_of_influence=10000, epsilon=0.5)
    ds['lightning_label'] = xr.DataArray(lightning_label, coords=[ds['cluster_label'],], dims=['cluster_label'])
    
    # use the lightning label as the dims
    ds = ds.swap_dims({'cluster_label': 'lightning_label'})

    return ds


def convert_cluster(scn, df, lightning_mask, kind='clean'):
    '''Convert the lightning cluster DataFrame to Dataset which has the same shape of NO2'''
    # rename the label
    df = df.rename(columns={'label': 'cluster_label'})
    # set label as index which is the coordinate of xarray Dataset
    df.set_index(['cluster_label'], inplace=True)
    # convert to Dataset
    ds = concat_pred(df.to_xarray())

    # get the pixel center points
    pixel_points = np.vstack((scn['nitrogendioxide_tropospheric_column'].longitude.stack(z=('x', 'y')),
                              scn['nitrogendioxide_tropospheric_column'].latitude.stack(z=('x', 'y')))
                             ).T

    labels = list(set(ds.cluster_label.values))

    for index, label in enumerate(labels):
        # iterate each label and update the mask
        dfq = ds.sel(cluster_label=label)

        # we use the lightning prediction over different pressure levels to create mask
        lon_lat = dfq[['longitude_pred', 'latitude_pred']].stack(all=("level", "cluster_label"))
        lightning_points = lon_lat.to_array().transpose('all', ...)

        # convert longitude from -180 ~ 180 to 0 ~ 360
        lightning_points[:, 0] %= 360

        # the cross line
        intersecting_line = LineString(((180, -90), (180, 90)))

        # combine the points into convex hull
        hull = MultiPoint(lightning_points).convex_hull

        # if the convex hull doesn't corss the 180 line,
        #   convert the longitude back to -180 ~ 180,
        #   and recreate the convex hull
        if not hull.intersects(intersecting_line):
            lightning_points[:, 0] = (lightning_points[:, 0] + 180) % 360 - 180
            hull = MultiPoint(lightning_points).convex_hull
        else:
            # convert the TROPOMI longitude to 0 ~ 360
            pixel_points[:, 0] %= 360

        # use matplotlib path instead of for loop to check the pixel points inside the convex hull
        #   https://iotespresso.com/find-which-points-of-a-set-lie-within-a-polygon/
        path_p = Path(hull.boundary)

        # get the mask where predicted lightning is inside
        mask = path_p.contains_points(pixel_points).reshape(scn['nitrogendioxide_tropospheric_column'].shape, order='F')

        # get the overplapped DBSCAN label
        overlapped_label = np.delete(np.unique(lightning_mask.where(xr.DataArray(mask, dims=['y', 'x']), 0)), 0)

        if len(overlapped_label) == 0:
            # no overlapped labels mean that only one DBSCAN cluster is inside
            if kind == 'clean':
                # clean lightning 1abel: 1, 2, ....
                lightning_mask = xr.where(mask, index+1, lightning_mask)
            elif kind == 'polluted':
                # polluted lightning 1abel: -1, -2, ....
                lightning_mask = xr.where(mask, -index-1, lightning_mask)
        elif len(overlapped_label) == 1:
            # set the mask value to the overlapped value
            lightning_mask = xr.where(mask, overlapped_label[0], lightning_mask)
        else:
            # this condition indicates there're >=2 DBSCAN clusters
            # get the minimum label
            min_label = np.min(overlapped_label)
            # set the mask where lightning happens to minimum label
            lightning_mask = xr.where(mask, min_label, lightning_mask)
            # update overlapped masks with minimum min label
            for rest_label in np.delete(overlapped_label, np.where(overlapped_label == min_label)):
                lightning_mask = lightning_mask.where(lightning_mask != rest_label, min_label)

    # generate label based on mask values
    ds = mask_label(scn, ds, lightning_mask)

    return ds, lightning_mask.rename('lightning_mask')


def concat_pred(ds):
    '''Concatenate predicted lon and lat by pressure level'''
    # get the DataArray names of predicted lon and lat
    lon_names = [var for var in ds.data_vars if 'longitude_pred' in var]
    lat_names = [var for var in ds.data_vars if 'latitude_pred' in var]
    levels = [int(name.split('_')[-1]) for name in lon_names]

    # concatenate into one array
    ds['longitude_pred'] = ds[lon_names].to_array(dim='lon_level', name='longitude_pred').assign_coords(lon_level=levels).rename({'lon_level': 'level'})
    ds['latitude_pred'] = ds[lat_names].to_array(dim='lat_level', name='longitude_pred').assign_coords(lat_level=levels).rename({'lat_level': 'level'})
    ds.coords['level'].attrs['units'] = 'hPa'

    # add description and units
    ds['longitude_pred'].attrs['description'] = 'Longitude of lightning at different pressure levels predicted by ERA5 data'
    ds['latitude_pred'].attrs['description'] = 'Latitude of lightning at different pressure levels predicted by ERA5 data'

    # drop useless variables
    ds = ds.drop_vars(lon_names+lat_names)

    return ds
