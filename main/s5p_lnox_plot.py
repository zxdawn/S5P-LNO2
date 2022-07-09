import gc
import logging
import os
from copy import deepcopy
from glob import glob
from itertools import groupby
from multiprocessing import Pool

import numpy as np
import pandas as pd
import proplot as pplt
import trackpy as tp
import xarray as xr
from pyresample import kd_tree
from pyresample.geometry import GridDefinition, SwathDefinition

from s5p_lnox_utils import Config

pplt.rc.reso = 'med'

# Disable warnings
from warnings import filterwarnings

filterwarnings("ignore")


def resample_grid(lon_min, lon_max, lat_min, lat_max):
    '''Define the resampled grid with 0.125 degree resolution'''
    grid_lon = np.arange(lon_min, lon_max, 0.125)
    grid_lat = np.arange(lat_min, lat_max, 0.125)
    lons, lats = np.meshgrid(grid_lon, grid_lat)
    target_grid = GridDefinition(lons=lons, lats=lats)

    return target_grid


def set_parameters():
    """Set the keyword arguments for linking step"""
    parameters_linking = {}
    parameters_linking['dxy'] = 5000  # m
    parameters_linking['dt'] = 1.5*3600  # s
    parameters_linking['vmax'] = 100  # m/s, actually this doesn't stand for the vmax because the NO2 mask is larger than the lightning/cloud region
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
    ds = xr.open_dataset(filename, group='S5P').isel(time=0)

    # swath_tropomi = SwathDefinition(lons=ds['longitude'], lats=ds['latitude'])
    # da_resample = xr.DataArray(kd_tree.resample_nearest(swath_tropomi, ds['lightning_mask'].values,
    #                                                     arctic_grid, radius_of_influence=50000, epsilon=0.5),
    #                         dims=['latitude', 'longitude'], coords=[grid_lat, grid_lon])

    labels = np.unique(ds['lightning_mask'])
    labels = np.delete(labels, 0)

    lon_centers = []
    lat_centers = []

    for label in labels:
        mask = ds['lightning_mask']==label
        # mask_region = ds['lightning_mask'].where(mask, drop=True)
        lon_mean = ds['longitude'].where(mask, drop=True).mean()
        lat_mean = ds['latitude'].where(mask, drop=True).mean()
        if (lat_mean>lat_min) & (lat_mean<lat_max) & (lon_mean>lon_min) & (lon_mean<lon_max):
            lon_centers.append(lon_mean)
            lat_centers.append(lat_mean)
        else:
            # not in the target
            labels = np.delete(labels, label)

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
                           #'filename': (['index'], [os.path.basename(filename)]*len(y))
                           'filename': (['index'], [filename]*len(y))
                           })

    return features


def get_features(filenames):
    '''Get the lightning mask features for linking'''
    features = []

    for frame, filename in enumerate(filenames):
        logging.info(f'Processing features of {filename} (Frame {frame})')
        try:
            features.append(read_feature(frame, filename, target_grid))
        except:
            print('!!! Empty group data !!!')

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
    trajectories = trajectories.groupby('cell').filter(lambda x : len(x)>1).to_xarray()

    return trajectories


def read_data(traj):
    filename = traj['filename'].values[0]
    ds_tropomi = xr.open_dataset(filename, group='S5P').isel(time=0) \
                    [['nitrogendioxide_tropospheric_column', 'qa_value', 'processing_quality_flags',
                      'solar_zenith_angle', 'cloud_radiance_fraction_nitrogendioxide_window',
                      'apparent_scene_pressure', 'cloud_pressure_crb', 'nitrogendioxide_segmentation', 'lightning_mask']]

    ds_lightning = xr.open_dataset(filename, group='Lightning')

    return ds_tropomi, ds_lightning


def plot_scatter(ax, ds_lightning, level, cmap='Browns5_r', loc='r'):
    s = ax.scatter(ds_lightning['longitude_pred'].sel(level=level),
                   ds_lightning['latitude_pred'].sel(level=level),
                   marker="$\u25EF$", cmap=cmap,
                   # cmap_kw = {'left': 0.05, 'right': 1},
                   c=ds_lightning['delta'], s=6,
                   vmax=0,
                   # alpha=0.5
                   )
    ax.colorbar(s, loc=loc, label=f'Relative Flash Time ({level} hPa)')
    return s


def plot_func(ds_tropomi, ds_lightning, case_index, traj):
    """Plot data

    case_index:
        the index of linked swaths

    cell:
        No. of cell because swaths can be linked by several cells

    frame:
        the time frame of cells
    """
    #for mask_label in np.unique(ds_tropomi['lightning_mask']):
    #    if mask_label >= 1:
    mask_label = traj['label'].values[0]
    # get the data inside each labeled mask
    lightning_mask = ds_tropomi['lightning_mask'] == mask_label
    ds_lightning_mask = ds_tropomi.where(lightning_mask, drop=True)
    lon_mask = ds_lightning_mask.coords['longitude']
    lat_mask = ds_lightning_mask.coords['latitude']

    lon_min = lon_mask.min().values
    lon_max = lon_mask.max().values
    lat_min = lat_mask.min().values
    lat_max = lat_mask.max().values

    # create another mask for plotting NO2 in case of the bug of pcolormesh cross the pole
    ds_mask = ds_tropomi.where((ds_tropomi.coords['longitude'] >= lon_min) &
                               (ds_tropomi.coords['longitude'] <= lon_max) &
                               (ds_tropomi.coords['latitude'] >= lat_min) &
                               (ds_tropomi.coords['latitude'] <= lat_max),
                               drop=True)

    # general setting
    plot_set = {'x': 'longitude', 'y': 'latitude',
                'xlim': (lon_min, lon_max), 'ylim': (lat_min, lat_max),
                'discrete': False}

    fig, axs = pplt.subplots(nrows=3, ncols=3, span=False, sharey=3, sharex=3)

    ax = axs[0]
    # in case there's almost no valid data
    if ds_lightning_mask['nitrogendioxide_tropospheric_column'].sizes['x'] <= 1 or \
        ds_lightning_mask['nitrogendioxide_tropospheric_column'].sizes['y'] <= 1:
        print('no valid data ...')
        return
    ds_mask['nitrogendioxide_tropospheric_column'].plot(**plot_set,
                                                        vmin=0, vmax=4e-5,
                                                        cmap='Thermal',
                                                        cbar_kwargs={"label": "[mol m$^{-2}$]",
                                                                     # 'orientation':'horizontal',
                                                                     'loc': 'bottom'},
                                                        ax=ax)
    ax.format(title='NO$_2$ VCD')

    ax = axs[1]
    pqf_mask = 0b1111111

    pqf = ds_mask['processing_quality_flags'].to_masked_array().astype('uint32') & pqf_mask
    ax.pcolormesh(ds_mask['longitude'], ds_mask['latitude'], (pqf != 0), cmap='tab10', levels=2)
    ax.format(title='Processing Quality Flag \n (Blue is 0)')

    # ds_mask['qa_value'].plot(**plot_set, vmin=0, vmax=1,
    #                                            cmap='Paired',
    #                                            cbar_kwargs={"label": ""},
    #                                            ax=ax)
    # ax.format(title='qa_value')

    ax = axs[2]
    ds_mask['solar_zenith_angle'].plot(**plot_set,
                                       cmap='Thermal',
                                       cbar_kwargs={"label": "[degree]"},
                                       ax=ax)
    ax.format(title='Solar Zenith Angle')

    ax = axs[3]
    ds_mask['nitrogendioxide_segmentation'].where(ds_mask['nitrogendioxide_segmentation'] > 0)\
        .plot(**plot_set, cmap='Accent', add_colorbar=False, ax=ax)
    ax.format(title='Segmentation of NO$_2$ VCD')

    ax = axs[4]
    ds_mask['cloud_radiance_fraction_nitrogendioxide_window']\
        .plot(**plot_set,
              vmin=0.5, vmax=1,
              cmap='Spectral',
              # cmap_kw={'left':0.2, 'right':0.95},
              cbar_kwargs={"label": ""},
              extend='max',
              ax=ax
              )
    ax.format(title='Cloud Radiance Fraction')

    ax = axs[5]
    #(ds_mask['cloud_pressure_crb']/1e2).plot(**plot_set,
    (ds_mask['apparent_scene_pressure']/1e2).plot(**plot_set,
                                                  vmin=150, vmax=700,
                                                  cmap='Spectral_r',
                                                  # cmap_kw={'left':0.05, 'right':0.8},
                                                  cbar_kwargs={"label": "[hPa]"},
                                                  ax=ax
                                                  )
    #ax.format(title='Cloud Pressure')
    ax.format(title='Apparent Scene Pressure')

    ax = axs[6]
    # plot transported lightning tracer at three pressure levels
    # plot_scatter(ax, ds_lightning, 650, cmap='Blue2', loc='l')
    plot_scatter(ax, ds_lightning, 500, cmap='Blue2', loc='l')
    # plot_scatter(ax, ds_lightning, 450, cmap='Purples2')
    # plot_scatter(ax, ds_lightning, 350, cmap='Browns3', loc='r')
    plot_scatter(ax, ds_lightning, 300, cmap='Browns3', loc='r')

    # plot the contour of mask
    ax.contour(lightning_mask.longitude, lightning_mask.latitude, lightning_mask,
               levels=[1], colors=['red5'])

    recent_lightning = (ds_lightning['delta'] > -100).sum().values
    ax.format(title=f"Horizontal transport of lightning tracer \n Flash Count in 100 min : {recent_lightning} \n Mask label: {mask_label}")

    ax = axs[7]
    ds_lightning_mask['nitrogendioxide_tropospheric_column'].plot\
        (**plot_set,
         vmin=0, vmax=ds_lightning_mask['nitrogendioxide_tropospheric_column'].max(),
         cmap='Thermal',
         cbar_kwargs={"label": "[mol m$^{-2}$]"},
         ax=ax
         )
    ax.format(title='Tracked NO$_2$ VCD')

    ax = axs[8]
    ds_lightning_mask['nitrogendioxide_tropospheric_column'].plot(**plot_set,
                                                                  cmap='Thermal',
                                                                  cbar_kwargs={"label": "[mol m$^{-2}$]"},
                                                                  ax=ax
                                                                  )
    ax.format(title='Tracked NO$_2$ VCD')

    # in case the point cross 180E
    if lon_max - lon_min > 180:
        lon_min = min([n for n in ds_lightning['longitude'] if n>0]).values

    axs.format(xlim=(lon_min, lon_max), ylim=(lat_min, lat_max),
               suptitle=ds_tropomi.s5p_filename,
               xformatter='deglat', yformatter='deglon',
               xlabel='', ylabel='',
               )

    savedir = os.path.join(cfg['output_fig_dir'], ds_tropomi.attrs['s5p_filename'][20:26])

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            # sometimes it tries to create the dir at the same time
            pass

    cell = traj['cell'].values[0]
    frame = int(traj['frame'].values[0])
    savename = os.path.join(savedir, f"Case{case_index}_Cell{cell}_Frame{frame}_"+ds_tropomi.attrs['s5p_filename'][:-19]+f'_{mask_label}.jpg')
    print(f'Saving to {savename}')
    fig.savefig(savename, dpi=300)

    del ds_tropomi, ds_lightning
    gc.collect()


def plot_data(trajectories, case_index):
    print(trajectories)
    for frame, traj in trajectories.groupby('frame'):
        ds_tropomi, ds_lightning = read_data(traj)
        plot_func(ds_tropomi, ds_lightning, case_index, traj)


def plot_swaths(case_index, swaths):
    # iterate through each case and get the features
    features = get_features(swaths)
    trajectories = get_tracks(features)
    if len(trajectories['cell']) > 0:
        for _, grp in trajectories.groupby('cell'):
            plot_data(grp, case_index=case_index)

def main():
    # get all filenames based on requested date range
    pattern = os.path.join(clean_dir, '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
    filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], [])

    # get the basenames of the filelist
    basenames = list(sorted(map(os.path.basename, filelist)))

    # group basenames by consecutive swath number
    #   https://stackoverflow.com/q/72035356/7347925
    consecutive_swaths = []
    for _, g in groupby(enumerate(basenames), lambda k: int(k[1].split('_')[-3]) - k[0]):
        consecutive_swaths.append([os.path.join(clean_dir, v.split('_')[-5][:6],v) for _, v in g])

    # drop list length equal to 1 (only one single swath is available)
    consecutive_swaths = [t for t in consecutive_swaths if len(t)>1]
    logging.info(f'Found {len(consecutive_swaths)} consecutive swaths.')


    with Pool(processes=int(cfg['nb_worker'])) as pool:
        pool.starmap(plot_swaths, enumerate(consecutive_swaths))


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