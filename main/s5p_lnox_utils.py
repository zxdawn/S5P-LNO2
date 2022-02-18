'''
Some utils for s5p_lnox_main.py

UPDATE:
    Xin Zhang:
       05/12/2021: Basic
'''

import functools
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor as Pool
from datetime import timedelta
from glob import glob
from itertools import compress

import geopandas as gpd
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
# from geopy.distance import geodesic
from metpy.units import units
from s5p_lnox_functions import *
from satpy import Scene
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')


def validation(cfg):
    '''Validate inputs'''
    validate_date(cfg['start_date'])
    validate_date(cfg['end_date'])
    validate_path(cfg['s5p_dir'], 's5p_dir')
    validate_path(cfg['lightning_dir'], 'lightning_dir')
    validate_path(cfg['fire_dir'], 'fire_dir')
    validate_path(cfg['output_data_dir'], 'output_data_dir')
    validate_path(cfg['output_fig_dir'], 'output_fig_dir')


def load_s5p_era5(f_s5p, cfg):
    '''Load s5p data'''
    logging.debug(' '*4+f'Reading {f_s5p} ...')
    scn = Scene(f_s5p, reader='tropomi_l2')

    vnames = ['time_utc', 'qa_value',
              'processing_quality_flags', 'geolocation_flags',
              # 'latitude', 'longitude',
              # 'latitude_bounds', 'longitude_bounds',
              'assembled_lon_bounds', 'assembled_lat_bounds',
              'nitrogendioxide_slant_column_density',
              'nitrogendioxide_tropospheric_column', 'nitrogendioxide_stratospheric_column',
              'nitrogendioxide_total_column', 'nitrogendioxide_ghost_column',
              'cloud_pressure_crb', 'cloud_radiance_fraction_nitrogendioxide_window',
              'cloud_fraction_crb_nitrogendioxide_window',
              'air_mass_factor_stratosphere', 'air_mass_factor_troposphere',
              'air_mass_factor_total',
              'air_mass_factor_clear', 'air_mass_factor_cloudy',
              'scene_albedo', 'snow_ice_flag', 'solar_azimuth_angle', 'solar_zenith_angle',
              'viewing_azimuth_angle', 'viewing_zenith_angle',
              'surface_albedo_nitrogendioxide_window', 'surface_pressure', 'apparent_scene_pressure',
              'tm5_constant_a', 'tm5_constant_b', 'tm5_tropopause_layer_index', 'averaging_kernel',
              ]

    logging.debug(' '*4 + f'Reading vnames: {vnames}')
    scn.load(vnames)

    # get the mean overpass time
    t_overpass = pd.to_datetime(scn['time_utc'].where(scn['nitrogendioxide_tropospheric_column'].latitude.mean('x') >= cfg['lat_min'], drop=True)).mean()

    # set global attrs
    scn.attrs['s5p_filename'] = os.path.basename(f_s5p[0])

    # read hourly era5 data
    ds_era5 = xr.open_dataset(os.path.join(cfg['era5_dir'],
                              f"era5_{scn['nitrogendioxide_tropospheric_column'].time.dt.strftime('%Y%m').values}.nc")
                              )

    return scn, vnames, t_overpass, ds_era5


def load_lightning_fire(scn, t_overpass, cfg):
    '''Read lightning and fire data related to the S5P NO2 Scene'''
    # get lightning data on overpass day and one day before
    day_now = scn.attrs['end_time']
    day_pre = scn.attrs['end_time']-timedelta(days=1)
    lightning_list = [day.strftime(f"{cfg['lightning_dir']}/%Y%m/xin_Arctic_pulse%Y%m%d.csv") for day in [day_pre, day_now]]
    # drop not existed filename
    lightning_list = [filename for filename in lightning_list if os.path.exists(filename)]

    # read ENGLN and VIIRS data
    logging.debug(f'    Reading {lightning_list} ...')
    df_lightning = pd.concat(map(pd.read_csv, lightning_list))
    df_lightning = df_lightning[df_lightning.latitude >= cfg['lat_min']]
    logging.debug(f"    Reading {glob(cfg['fire_dir']+'/*.csv')} ...")
    df_viirs = pd.concat((pd.read_csv(f, dtype={'acq_time': 'str'}) for f in glob(cfg['fire_dir']+'/*.csv')))

    # get lightning dots during the several hours before the mean overpass time
    df_lightning['timestamp'] = pd.to_datetime(df_lightning['timestamp'], utc=True)
    delta = df_lightning['timestamp'] - t_overpass
    df_lightning['delta'] = delta.dt.total_seconds()/60
    subset = (-cfg['delta_time'] < df_lightning['delta']) & (df_lightning['delta'] < 0)
    df_lightning = df_lightning[subset]

    df_viirs['time'] = pd.to_datetime(df_viirs['acq_date'] + ' ' + df_viirs['acq_time'], utc=True)
    delta = df_viirs['time'] - t_overpass
    df_viirs['delta'] = delta.dt.total_seconds()/60
    subset = (-cfg['delta_time'] < df_viirs['delta']) & (df_viirs['delta'] < 0)
    df_viirs = df_viirs[subset]

    return df_lightning, df_viirs


def get_cluster(df_lightning):
    '''Get the cluster of lightning points

    Ref: https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/

    '''
    coords = df_lightning[['latitude', 'longitude']].values
    kms_per_radian = 6371.0088

    # search for 40km around each lightning dots
    epsilon = 40/kms_per_radian
    logging.info(' '*4 + 'Cluster lightning by DBSCAN ...')
    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_

    df_cluster = pd.DataFrame({'time': df_lightning['timestamp'].values,
                               'longitude': df_lightning['longitude'].values,
                               'latitude': df_lightning['latitude'].values,
                               'type': df_lightning['type'].values,
                               'peakcurrent': df_lightning['peakcurrent'].values,
                               'delta': df_lightning['delta'].values,
                               'label': cluster_labels})

    # drop noise data (-1)
    df_cluster = df_cluster[df_cluster.label != -1]

    # more than 20 points
    v = df_cluster.label.value_counts()
    df_cluster = df_cluster[df_cluster.label.isin(v.index[v.gt(20)])]
    cluster_labels = list(set(df_cluster['label']))
    # num_clusters = len(cluster_labels)

    hulls = []

    for label in cluster_labels:
        dfq = df_cluster[df_cluster['label'] == label]
        Y = np.array(dfq[['longitude', 'latitude']])
        hull = ConvexHull(Y)
        hulls.append(hull)

    large_hulls = get_large_hull(hulls)

    return df_cluster, cluster_labels, large_hulls


def classify_lightning(df_cluster, df_viirs, cluster_labels, hulls):
    '''Drop lightning cluster where fire is inside'''
    # https://stackoverflow.com/questions/49298505/find-polygons-that-contain-at-least-one-of-a-list-of-points-in-python
    logging.info(' '*4 + 'Clean lightning ...')
    polys = gpd.GeoSeries([MultiPoint(hull.tolist()).convex_hull for hull in hulls])
    points = gpd.GeoSeries(df_viirs[['longitude', 'latitude']].apply(Point, axis=1).values)
    masks = polys.apply(lambda x: ~points.within(x).any()).values

    clean_label = list(compress(cluster_labels, masks))
    clean_cluster = df_cluster[df_cluster['label'].isin(clean_label)]
    polluted_cluster = df_cluster[~df_cluster['label'].isin(clean_label)]

    # change index type to float which `apply` function relies on
    clean_cluster.index = clean_cluster.index.map(float)
    polluted_cluster.index = polluted_cluster.index.map(float)

    return clean_cluster, polluted_cluster


def pred_cluster(df_cluster, t_overpass, ds_era5, wind_levels, cfg):
    '''Predict the location of transported lighting clusters

    1) Get the interped wind direction and wind speed at each lightning location from hourly ERA5 data
    2) Predict the location of lightning air at the TROPOMI overpass time by the wind info

    Note that the source of high NO2 could originate frome different pressure levels.
    So, we checked chose 700 hPa ~ 300 hPa for prediction.

    Because `parallel_apply` sometimes raise "can't locate memory" error.
        we split the cluster into chunks whose row number is <= 100 and apply pool map manually
    Ref:    https://stackoverflow.com/a/41447373
            https://stackoverflow.com/a/44729807
    If you have large memory, you can delete the loop in `pred_cluster_chunk()` and use `parallel_apply()`:
        `for level in wind_levels:
            clean_cluster.parallel_apply(pred_cluster_chunk, axis=1, args=(t_overpass, ds_era5, level))`
    Ofc, `dask` is another option which needs tricky settings of chunk size.
    Ref:    https://stackoverflow.com/a/37979876
            https://gdcoder.com/speed-up-pandas-apply-function-using-dask-or-swifter-tutorial/
    Let's stick to the manual chunk by `np.array_split()` at this stage.

    If someone figure out how to vectorise the process, please feel free to PR.
    '''
    for level in wind_levels:
        chunk_df = np.array_split(df_cluster.reset_index(drop=True), np.arange(0, len(df_cluster), 100))
        with Pool(max_workers=int(cfg['num_cpus'])) as pool:
            pred_lons, pred_lats = zip(*pool.map(functools.partial(pred_cluster_chunk, t_overpass=t_overpass,
                                                                   ds_era5=ds_era5, level=level),
                                                 chunk_df))
            pred_lons = sum([*pred_lons], [])
            pred_lats = sum([*pred_lats], [])

        df_cluster[f'longitude_pred_{level}'] = pred_lons
        df_cluster[f'latitude_pred_{level}'] = pred_lats

    return df_cluster


def pred_cluster_chunk(df, t_overpass, ds_era5, level):
    lons, lats = [], []

    for _, row in df.iterrows():
        times = np.concatenate(([row.time.to_pydatetime().replace(tzinfo=None)],
                                pd.date_range(row.time.ceil('h').replace(tzinfo=None),
                                t_overpass.floor('h').replace(tzinfo=None), freq='H').to_pydatetime(),
                                [t_overpass.replace(tzinfo=None).to_pydatetime()]
                                ))

        delta_wind = [t.total_seconds() for t in np.diff(times)]

        lat, lon = row.latitude, row.longitude

        for t_index, time in enumerate(times[:-1]):
            data = ds_era5.interp(time=time, longitude=lon, latitude=lat, level=level)

            # if there's nan value then just skip it.
            # This situation sometimes happen at the high level (> 250 hPa)
            if np.isnan(data['u']) or np.isnan(data['v']):
                continue

            data['wspd'] = mpcalc.wind_speed(data['u'] * units.meters / units.second,
                                             data['v'] * units.meters / units.second).rename('wspd')
            data['wdir'] = mpcalc.wind_direction(data['u'] * units.meters / units.second,
                                                 data['v'] * units.meters / units.second,
                                                 convention='to').rename('wdir')

            # # https://stackoverflow.com/a/40645383/7347925
            # # validation website: https://www.fcc.gov/media/radio/find-terminal-coordinates
            # # this method has issue with pandarallel
            # dest = geodesic(kilometers=(data['wspd']*delta_wind[t_index]/1e3).metpy.dequantify())\
            #                 .destination(geopy.Point(lat, lon), data['wdir'].metpy.dequantify())
            # lat = dest[0]
            # lon = dest[1]

            # manual method works well with pandarallel
            lon, lat = predict_loc(lon, lat, data['wdir'].metpy.dequantify(),
                                   data['wspd'].metpy.dequantify(), delta_wind[t_index])
        lons.append(lon)
        lats.append(lat)

    return lons, lats


def segmentation(scn, min_threshold=4e14, max_threshold=1e15, step_threshold=2e14):
    '''Segmentation the NO2 field by tobac

    Tobac: https://gmd.copernicus.org/articles/12/4551/2019/

    '''
    logging.info(' '*4 + 'Segmentation of NO2 ...')
    # get the finite no2 data
    no2_finite = get_finite_scn(scn)  # unit: molec./cm2

    # Detect the features and get the masks using tobac
    masks_scn = feature_mask(no2_finite, min_threshold=min_threshold,
                             max_threshold=max_threshold, step_threshold=step_threshold)

    return masks_scn


def lightning_mask(scn, clean_cluster, polluted_cluster):
    '''Create NO2 mask (0: no lightning, 1: lightning with fire, >=2: clean lightning with detected high NO2 ) '''
    logging.info(' '*4 + 'Creating lightning mask ...')

    # initialize the masks
    clean_lightning_mask = xr.full_like(scn['nitrogendioxide_tropospheric_column'], 0, dtype=int).load().copy()
    polluted_lightning_mask = clean_lightning_mask.copy()

    if not clean_cluster.empty:
        clean_cluster, clean_lightning_mask = convert_cluster(scn, clean_cluster, clean_lightning_mask, 'clean')
    else:
        clean_cluster = None

    if not polluted_cluster.empty:
        _, polluted_lightning_mask = convert_cluster(scn, polluted_cluster, polluted_lightning_mask, 'polluted')

    lightning_mask = clean_lightning_mask + polluted_lightning_mask
    lightning_mask.attrs['description'] = '<0: labeled lightning with fire; 0: no lightning; >0: labeled lightning without fire'

    return lightning_mask, clean_cluster


def segmentation_mask(scn, masks_scn):
    '''Create the segmentation mask from labels and set values (0: no high NO2, others: high NO2)

    Note that these masks are detected by tobac and don't represent the correct lightning NO2 region.
    So, it is better to use the `lightning_label` stored in `clean_cluster` and `polluted cluster`.
    '''
    logging.info(' '*4 + 'Creating clean LNO2 segmentation mask ...')

    # create the mask filled with -1
    #   x and y are only used to assign values
    no2_segmentation = xr.full_like(scn['nitrogendioxide_tropospheric_column'], 0, dtype=int)\
                         .assign_coords({'y': range(len(scn['nitrogendioxide_tropospheric_column'].y)),
                                         'x': range(len(scn['nitrogendioxide_tropospheric_column'].x))
                                         }).copy()

    no2_segmentation.load()
    no2_segmentation.loc[dict(y=masks_scn.coords['y'], x=masks_scn.coords['x'])] = masks_scn

    # delete the manually assigned coordinates
    no2_segmentation = no2_segmentation.drop_vars(['x', 'y'])

    # clean attributes
    no2_segmentation.attrs = []
    no2_segmentation.attrs['description'] = '0: no high NO2; >=1: labeled high NO2'

    return no2_segmentation.rename('nitrogendioxide_segmentation')


def save_data(savedir, filename, scn, vnames, cfg, lightning_mask, df_viirs=None, clean_cluster=None, no2_segmentation=None):
    '''Saving all data

    1) lightning data (1D, coord: lightning_label)
        time, longitude, latitude, delta, wdir,
        wspd, longitude_pred, latitude_pred

    2) lightning cluster (1D, coord: no2_label)
        lightning_label, no2_label, geometry

    3) TROPOMI L2 subset data with no2_label (2D, coord:[y, x])

    '''
    # get the saving path
    output_file = os.path.join(savedir,
                               os.path.basename(filename)[20:26],
                               os.path.basename(filename)[:-19]+'.nc'
                               )

    # create the dir if not existed
    if not os.path.isdir(savedir+os.path.basename(filename)[20:26]):
        os.makedirs(savedir+os.path.basename(filename)[20:26])

    logging.info(' '*4 + f'Saving to {output_file}')

    if no2_segmentation is None:
        s5p_vnames = vnames + ['lightning_mask']
    else:
        s5p_vnames = vnames + ['lightning_mask', 'nitrogendioxide_segmentation']
        scn['nitrogendioxide_segmentation'] = no2_segmentation

    scn['lightning_mask'] = lightning_mask

    # set group attributes
    group_attrs = {'s5p_filename': scn.attrs['s5p_filename'],
                   'description': 'Subset of official TROPOMI L2 data'
                   }

    # set compression
    comp = dict(zlib=True, complevel=7)

    logging.info(' '*8 + f'Saving S5P products ...')
    scn.save_datasets(filename=output_file,
                      datasets=s5p_vnames,
                      groups={'S5P': s5p_vnames},
                      compute=True,
                      group_attrs=group_attrs,
                      writer='cf',
                      engine='netcdf4',
                      compression=comp,
                      )

    if clean_cluster is not None:
        # set data types
        float_names = ['longitude', 'latitude', 'delta', 'longitude_pred', 'latitude_pred']
        for name in float_names:
            clean_cluster[name] = clean_cluster[name].astype('float32')

        # add description
        clean_cluster['lightning_label'].attrs['description'] = 'Clustered lightning labeled by DBSCAN'
        clean_cluster['delta'].attrs['description'] = 'The time difference between detected lightning and TROPOMI overpass time'
        clean_cluster['delta'].attrs['units'] = 'minute'

        enc = {var: comp for var in clean_cluster.data_vars}
        clean_cluster.attrs['description'] = f"Clean lighting point data grouped by lightning_label, {cfg['delta_time']} minutes before TROPOMI overpass"

        logging.info(' '*8 + f'Saving clustered clean lightning ...')
        clean_cluster.to_netcdf(path=output_file,
                                group='Lightning',
                                engine='netcdf4',
                                encoding=enc,
                                mode='a')

    if df_viirs is not None:
        # subset fire DataFrame to Dataset
        ds_viirs = df_viirs.set_index('time')[['longitude', 'latitude', 'type']].to_xarray()

        # set data types
        ds_viirs['longitude'] = ds_viirs['longitude'].astype('float32')
        ds_viirs['latitude'] = ds_viirs['latitude'].astype('float32')

        enc = {var: comp for var in ds_viirs.data_vars}
        ds_viirs.attrs['description'] = f"SNPP/VIIRS fire point data, {cfg['delta_time']} minutes before TROPOMI overpass"

        logging.info(' '*8 + f'Saving Fire products ...')
        ds_viirs.to_netcdf(path=output_file,
                           group='Fire',
                           engine='netcdf4',
                           encoding=enc,
                           mode='a')
