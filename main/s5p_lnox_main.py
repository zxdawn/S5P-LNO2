'''
INPUT:
    - S5P (TROPOPMI) L2 product files
    - ENGLN lightning flash csv files or VAISALA lightning density/flash files

OUTPUT:
    S5P data with lightning data and masks

Steps:
    1) Read each swath of TROPOMI
    2) Get both detected and predicted lightning flashes on different pressure levels
            <delta_time> minutes before TROPOMI overpass
    3) Cluster clean and polluted lightning into `lightning_mask` , with the help of VIIRS fire data
    4) Sementation of high NO2 pixels (`nitrogendioxide_segmentation`) via `tobac`
    5) Save the S5P, Lightning, and Fire variables in individual groups of NetCDF files

UPDATE:
    Xin Zhang:
       05/12/2021: Basic version
       19/12/2021: Export masks (lightning_mask and nitrogendioxide_segmentation) instead of overlapped lightning
'''

import functools
import logging
import os
from concurrent.futures import ProcessPoolExecutor as Pool
from warnings import filterwarnings

import numpy as np
import pandas as pd
import xarray as xr
from s5p_lnox_utils import *

filterwarnings("ignore")
# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger('satpy').setLevel(logging.ERROR)


def process_data(filename, cfg):
    '''Process data and save data related to LNO2'''
    # generate the output filename
    output_file = os.path.join(cfg['output_data_dir'],
                               os.path.basename(filename)[20:26],
                               os.path.basename(filename)[:-19]+'.nc'
                               )
    basename = os.path.basename(filename)

    # whether continue processing data
    if not check_log(cfg, basename):
        return

    # load s5p and era5 data
    logging.info(f'    Processing {filename} ...')
    scn, vnames, t_overpass, ds_era5 = load_s5p_era5([filename], cfg)

    # load lightning and fire data
    df_lightning, df_viirs = load_lightning_fire(scn, t_overpass, cfg)

    # create lightning mask initialized with zero
    mask = xr.full_like(scn['nitrogendioxide_tropospheric_column'], 0, dtype=int).copy().rename('no2_mask')

    if df_lightning.empty:
        logging.info(' '*4+f'No lightning flashes are found for {os.path.basename(filename)}')
    else:
        # cluster lightning data
        df_cluster, cluster_labels, hulls = get_cluster(df_lightning)

        if df_cluster.empty:
            logging.info(' '*4+f'No lightning clusters are found for {os.path.basename(filename)}')
        else:
            # classify lightning clusters into clean and polluted (fire) categories
            clean_cluster, polluted_cluster = classify_lightning(df_cluster, df_viirs, cluster_labels, hulls)

            if not clean_cluster.empty:
                logging.info(' '*4+'Calculate the transported clean lighting cluster ...')
                clean_cluster = pred_cluster(clean_cluster, t_overpass, ds_era5, wind_levels, cfg)

            if not polluted_cluster.empty:
                logging.info(' '*4+'Calculate the transported polluted lighting cluster ...')
                polluted_cluster = pred_cluster(polluted_cluster, t_overpass, ds_era5, wind_levels, cfg)

            # create the pixel masks overlapped with the ConvexHull of each cluster
            #   and convert the clean_cluster into Dataset
            mask, clean_cluster = lightning_mask(scn, clean_cluster, polluted_cluster)

            # we only apply the segmentation for clean lightning to save time
            #      if you need it for all the conditions, please feel free to modify the code.
            if clean_cluster is not None:
                # segmentate the high NO2 pixels by `tobac`
                #   take care of the threshold, the default values are suitable for clean region like the Arctic
                masks_scn = segmentation(scn, min_threshold=cfg['min_threshold'],
                                         max_threshold=cfg['max_threshold'],
                                         step_threshold=cfg['step_threshold'])

                # beceause `segmentation` delete some scanlines with nan value,
                #   we need to pair the masks_scn with the swath pixels
                no2_segmentation = segmentation_mask(scn, masks_scn)

                # export datasets to NetCDF file
                save_data(clean_dir, filename, scn, vnames, cfg, mask, df_viirs, clean_cluster, no2_segmentation)
            else:
                # in case users want to use fire polluted data, the dataset is exported instead of saving the empty one
                #   Note taht the lightning_mask contains the fire mask, we don't need to save extra polluted cluster
                save_data(fire_dir, filename, scn, vnames, cfg, mask, df_viirs)


def main():
    # get all filenames based on requested date range
    pattern = os.path.join(cfg['s5p_dir'], '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
    filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], [])

    with Pool(max_workers=int(cfg['num_pool'])) as pool:
        # data process in parallel
        # we don't use multiprocessing.Pool because it's non-daemonic
        #  https://stackoverflow.com/a/61470465/7347925
        try:
            pool.map(functools.partial(process_data, cfg=cfg), filelist)
        except Exception as exc:
            logging.info(exc)


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)

    overwrite = cfg.get('overwrite', 'True') == 'True'
    clean_dir = cfg['output_data_dir'] + '/clean_lightning/'  # directory to only save clean lightning cases
    fire_dir = cfg['output_data_dir'] + '/fire_lightning/'  # directory to only save fire lightning cases
    wind_levels = np.arange(700, 100, -200)  # pressure levels used to calculate the location of transported LNO2 air parcel

    # generate data range
    req_dates = pd.date_range(start=cfg['start_date'],
                              end=cfg['end_date'],
                              freq='D')

    main()
