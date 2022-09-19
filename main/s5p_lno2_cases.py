'''
INPUT:
    - Processed S5P (TROPOPMI) L2 product files by `s5p_lno2_main.py`

OUTPUT:
    - csv file (lno2_cases_yyyymmdd-yyyymmdd.csv) which saves the consecutive swaths data
        Columns: case, label, filename, fresh_lightning
            - case: the index of cases
            - label: the lightning_label in the processed S5P L2 product
            - filename: the filename of processed S5P L2 files
            - fresh_lightning: how many lightning is detected during the 100-min period before the TROPOMI overpass

UPDATE:
    Xin Zhang:
        2022-07-13: basic version
'''

import logging
import os
from copy import deepcopy
from glob import glob
from itertools import groupby
# Disable warnings
from warnings import filterwarnings

import numpy as np
import pandas as pd
import trackpy as tp
import xarray as xr
from pyresample import kd_tree
from pyresample.geometry import GridDefinition, SwathDefinition

from s5p_lno2_utils import Config

filterwarnings("ignore")
# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)

# logging.basicConfig(level=logging.DEBUG)


def resample_grid(lon_min, lon_max, lat_min, lat_max):
    '''Define the resampled grid with 0.125 degree resolution'''
    grid_lon = np.arange(lon_min, lon_max+0.125, 0.125)
    grid_lat = np.arange(lat_min, lat_max+0.125, 0.125)
    lons, lats = np.meshgrid(grid_lon, grid_lat)
    target_grid = GridDefinition(lons=lons, lats=lats)

    return target_grid


def set_parameters():
    """Set the keyword arguments for linking step"""
    parameters_linking = {}
    parameters_linking['dxy'] = 5000  # m
    parameters_linking['dt'] = 1.5*3600  # s
    parameters_linking['vmax'] = 50  # m/s, actually this doesn't stand for the vmax because the NO2 mask is larger than the lightning/cloud region
    parameters_linking['stubs'] = 2
    parameters_linking['order'] = 1
    parameters_linking['extrapolate'] = 1
    parameters_linking['memory'] = 0
    parameters_linking['adaptive_stop'] = 0.2
    parameters_linking['adaptive_step'] = 0.95
    parameters_linking['subnetwork_size'] = 100
    parameters_linking['method_linking'] = 'predict'

    return parameters_linking


def read_feature(frame, filename, target_grid):
    '''Get the features of TROPOMI swath segmentation'''
    try:
        ds = xr.open_dataset(filename, group='S5P').isel(time=0)
        ds_lightning = xr.open_dataset(filename, group='Lightning')
    except:
        print('!!! Empty group data !!!')
        return None

    labels = np.unique(ds['lightning_mask'])
    labels = np.delete(labels, 0)

    lon_centers = []
    lat_centers = []
    fresh_lightnings = []

    # # check if all labels of lightning_mask are available in lightning_label
    # #   if not, then there's something wrong in the pairing, especially at the place crossing 180 deg
    # pair_cond = all(item in np.unique(ds_lightning['lightning_label']) for item in labels)

    # if not pair_cond:
    #     return None

    for label in labels:
        # calculate the center of lightning mask
        mask = ds['lightning_mask'] == label
        lon_mean = ds['longitude'].where(mask, drop=True).mean()
        lat_mean = ds['latitude'].where(mask, drop=True).mean()

        # calcualte how many fresh lightning are labelled
        fresh_lightning = (ds_lightning['delta'].where(ds_lightning['lightning_label'] == label) > -100).sum().data

        if (lat_mean > lat_min) & (lat_mean < lat_max) & (lon_mean > lon_min) & (lon_mean < lon_max):
            lon_centers.append(lon_mean)
            lat_centers.append(lat_mean)
            fresh_lightnings.append(fresh_lightning)
        else:
            # not in the target
            labels = np.delete(labels, label)

    # no valid case available
    if len(lon_centers) == 0:
        return None

    # define the swath based on lightning mask centers
    swath_center = SwathDefinition(lons=lon_centers, lats=lat_centers)

    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    _, _, index_array, distance_array = kd_tree.get_neighbour_info(source_geo_def=target_grid,
                                                                   target_geo_def=swath_center,
                                                                   radius_of_influence=10000,
                                                                   neighbours=1)

    # get the y and x index
    y, x = np.unravel_index(index_array, target_grid.shape)

    features = xr.Dataset({'frame': (['index'], [frame]*len(y)),
                           'hdim_1': (['index'], y),
                           'hdim_2': (['index'], x),
                           'longitude': (['index'], lon_centers),
                           'latitude': (['index'], lat_centers),
                           'label': (['index'], labels),
                           'fresh_lightning': (['index'], fresh_lightnings),
                           # 'filename': (['index'], [os.path.basename(filename)]*len(y))
                           'filename': (['index'], [filename]*len(y))
                           })

    return features


def get_features(filenames):
    '''Get the lightning mask features for linking'''
    features = []

    for frame, filename in enumerate(filenames):
        logging.debug(f'Processing features of {filename} (Frame {frame})')
        feature = read_feature(frame, filename, target_grid)
        if feature is not None:
            features.append(feature)

    # combine features into one Dataset
    if len(features) > 0:
        features = xr.concat(features, dim='index')

    return features


def get_tracks(features):
    '''Link features into tracks'''
    pred = tp.predict.NearestVelocityPredict(span=1)
    tp.linking.Linker.MAX_SUB_NET_SIZE = parameters_linking['subnetwork_size']

    trajectories = pred.link_df(deepcopy(features.to_dataframe()),
                                search_range=int(parameters_linking['dt'] * parameters_linking['vmax'] / parameters_linking['dxy']),
                                memory=parameters_linking['memory'],
                                pos_columns=["hdim_1", "hdim_2"],
                                t_column="frame",
                                neighbor_strategy="KDTree",
                                link_strategy="auto",
                                adaptive_step=parameters_linking['adaptive_step'],
                                adaptive_stop=parameters_linking['adaptive_stop'])

    # assigning incremental values based on an unique value of a column
    trajectories["cell"] = pd.factorize(trajectories["particle"])[0] + 1
    trajectories.drop(columns=["particle"], inplace=True)

    # remove segmentations which only have one time step linking
    trajectories = trajectories.groupby('cell').filter(lambda x: len(x) > 1).to_xarray()

    return trajectories


def read_data(traj):
    """Read the processed TROPOMI data"""
    filename = traj['filename'].values[0]
    ds_tropomi = xr.open_dataset(filename, group='S5P').isel(time=0) \
                    [['nitrogendioxide_tropospheric_column', 'qa_value', 'processing_quality_flags',
                      'solar_zenith_angle', 'cloud_radiance_fraction_nitrogendioxide_window',
                      'apparent_scene_pressure', 'cloud_pressure_crb', 'nitrogendioxide_segmentation', 'lightning_mask']]

    ds_lightning = xr.open_dataset(filename, group='Lightning')

    return ds_tropomi, ds_lightning


def link_swaths(swaths, case_index, index_col, label_col, filename_col, fresh_lightning_col):
    """iterate through each case and get the features/tracks"""
    logging.info(swaths)
    features = get_features(swaths)
    if len(features) == 0:
        # no features are detected
        return case_index

    trajectories = get_tracks(features)

    if len(trajectories['cell']) > 0:
        # features are linked at least two swaths
        for _, grp in trajectories.groupby('cell'):
            # append vairables to lists
            index_col.extend([case_index]*len(grp['filename']))
            filename_col.extend(grp['filename'].values)
            label_col.extend(grp['label'].values)
            fresh_lightning_col.extend(grp['fresh_lightning'].values)
            # update the case index
            case_index += 1

    return case_index


def main():
    # get all filenames based on requested date range
    pattern = os.path.join(clean_dir, '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
    filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], [])

    # get the basenames of the filelist
    basenames = list(sorted(map(os.path.basename, filelist)))

    # every element of consecutive_swaths is successive swaths data
    consecutive_swaths = []

    # group basenames by consecutive swath number
    #   https://stackoverflow.com/q/72035356/7347925
    for _, g in groupby(enumerate(basenames), lambda k: int(k[1].split('_')[-3]) - k[0]):
        consecutive_swaths.append([os.path.join(clean_dir, v.split('_')[-5][:6], v) for _, v in g])

    # drop list length equal to 1 (only one single swath is available)
    consecutive_swaths = [t for t in consecutive_swaths if len(t) > 1]
    logging.info(f'Found {len(consecutive_swaths)} consecutive swaths.')

    # count cases
    case_index = 0

    # save data
    index_col = []
    label_col = []
    filename_col = []
    fresh_lightning_col = []

    for swath in consecutive_swaths:
        case_index = link_swaths(swath, case_index, index_col, label_col, filename_col, fresh_lightning_col)

    # combine lists into one DataFrame
    df = pd.DataFrame([index_col, label_col, filename_col, fresh_lightning_col],
                      index=['case', 'label', 'filename', 'fresh_lightning']).T
    # sort by case index and filename
    df = df.sort_values(['case', 'filename'])
    logging.info(df)

    # export data
    st = req_dates[0].strftime('%Y%m%d')
    et = req_dates[-1].strftime('%Y%m%d')
    savename = cfg['output_data_dir']+'/lno2_cases_' + st + '_' + et + '.csv'
    logging.info(f'Saved to {savename}')
    df.to_csv(savename, index=False)


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)

    overwrite = cfg.get('overwrite', 'True') == 'True'
    clean_dir = cfg['output_data_dir'] + '/clean_lightning/'  # directory where clean lightning cases are saved

    # generate data range
    req_dates = pd.date_range(start=cfg['start_date'],
                              end=cfg['end_date'],
                              freq='D')

    # define the target grid which is used to resample swaths to the same grid for linking data
    lon_min = -180; lon_max = 180; lat_min = 60; lat_max = 90
    target_grid = resample_grid(lon_min, lon_max, lat_min, lat_max)

    parameters_linking = set_parameters()

    main()
