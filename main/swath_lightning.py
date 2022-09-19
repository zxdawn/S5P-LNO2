'''
INPUT:
    - S5P (TROPOPMI) L2 product files
    - ENGLN lightning flash daily csv files or daily VAISALA lightning density/flash csv files

OUTPUT:
    NetCDF file:
        lightning happened in TROPOMI swaths where latitude > <lat_min>
            during <delta_time> minutes before TROPOMI overpass

UPDATE:
    Xin Zhang:
        2022-06-17: basic version
        2022-06-25: add variables: tropopause pressure and cloud pressure
'''

import gc
import logging
import os
from datetime import timedelta
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from pyresample.geometry import GridDefinition
from satpy import Scene
from scipy import stats
from scipy.optimize import linprog

from s5p_lno2_amf import cal_tropo
from s5p_lno2_utils import Config

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger('satpy').setLevel(logging.ERROR)


def load_s5p(filename):
    """Read TROPOMI data"""
    scn = Scene([filename], reader='tropomi_l2')
    scn.load(['time_utc', 'longitude', 'latitude',
              'tm5_constant_a', 'tm5_constant_b', 'tm5_tropopause_layer_index',
              'surface_pressure', 'cloud_pressure_crb', 'apparent_scene_pressure',
              'cloud_radiance_fraction_nitrogendioxide_window', 'processing_quality_flags'])

    overpass_time = pd.to_datetime(scn['time_utc'].where(scn['latitude'].mean('x') >= cfg['lat_min'], drop=True)).mean()

    # calculate pressure levels
    a = scn['tm5_constant_a']
    b = scn['tm5_constant_b']
    psfc = scn['surface_pressure']

    low_p = (a[:, 0] + b[:, 0]*psfc)/1e2

    itropo = scn['tm5_tropopause_layer_index']

    scn['ptropo'] = cal_tropo(low_p, itropo)
    scn['apparent_scene_pressure'] /= 100

    # calculate pqf
    pqf_mask = 0b1111111
    pqf = scn['processing_quality_flags'].to_masked_array().astype('uint32') & pqf_mask
    scn['pqf'] = scn['ptropo'].copy(deep=True)
    scn['pqf'].values = pqf

    # pick valid data
    cloud_mask = (scn['cloud_radiance_fraction_nitrogendioxide_window']>=crf_low) & (scn['pqf'] == 0)
    scn['cloud_pressure_crb'] = scn['cloud_pressure_crb'].where(cloud_mask)
    scn['ptropo'] = scn['ptropo'].where(cloud_mask)

    return scn, overpass_time


def load_lightning(scn, t_overpass):
    """Read lightning data"""
    day_now = t_overpass
    day_pre = day_now-timedelta(days=1)

    lightning_list = [day.strftime(f"{cfg['lightning_dir']}/%Y%m/%Y%m%d.csv") for day in [day_pre, day_now]]
    # drop not existed filename
    lightning_list = [filename for filename in lightning_list if os.path.exists(filename)]

    # read lightning and VIIRS data
    logging.debug(f'    Reading {lightning_list} ...')
    df_lightning = pd.concat(map(pd.read_csv, lightning_list))
    df_lightning = df_lightning[df_lightning.latitude >= cfg['lat_min']]

    # get lightning dots during the several hours before the mean overpass time
    df_lightning['timestamp'] = pd.to_datetime(df_lightning['timestamp'], utc=True)
    delta = df_lightning['timestamp'] - t_overpass
    df_lightning['delta'] = delta.dt.total_seconds()/60
    subset = (-cfg['delta_time'] < df_lightning['delta']) & (df_lightning['delta'] < 0)
    df_lightning = df_lightning[subset]

    # # if the lightning data is density data, which have column name including "count",
    # #   then add the count info to new column
    # boolean_count = df_lightning.columns.str.lower().str.contains('count')
    # if any(boolean_count):
    #     # read "count" column
    #     count_colname = df_lightning.columns[boolean_count]
    #     df_lightning['count'] = df_lightning[count_colname].values.astype('int64')

    #     # in case there's something wrong with count
    #     df_lightning['count'].values[df_lightning['count'] < 1] = 0

    #     # duplicate the rows if the lightning data is density data, which have the "count" column
    #     df_lightning = df_lightning.loc[df_lightning.index.repeat(df_lightning['count'])]

    #     del df_lightning['count']

    return df_lightning[['timestamp', 'longitude', 'latitude']]


def in_hull(points, x):
    # https://stackoverflow.com/a/43564754/2912349
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)

    return lp.success


def resample_scn(scn):
    """Create the mask of swath related to defined grid"""
    # create array full of 1 to create mask
    scn['mask'] = xr.full_like(scn['longitude'], 1)

    # define the arctic grid
    lats, lons = np.meshgrid(lat_center, lon_center)
    arctic_grid = GridDefinition(lons=lons, lats=lats)

    # resample to arctic grid
    scn_resample = scn.resample(arctic_grid)

    return scn_resample


def process_data(filename):
    """Process TROPOMI and lightning data"""
    logging.info(f'    Processing {filename} ...')
    # load tropomi data
    scn, t_overpass = load_s5p(filename)

    if pd.isnull(t_overpass):
        t_overpass = pd.to_datetime(scn['longitude'].attrs['end_time'], utc=True)

    # load lightning and fire data
    logging.debug('    Reading lightning data ...')
    df_lightning = load_lightning(scn, t_overpass)

    # check whether the lightning point is within the subset of tropomi swath
    # https://stackoverflow.com/q/72657458/7347925
    logging.debug('    Pick the lightning within swath ...')

    scn_resample = resample_scn(scn)
    scn_resample['mask'].load()

    if len(df_lightning['longitude']) > 0:
        lightning_counts = xr.DataArray(stats.binned_statistic_2d(df_lightning['longitude'], df_lightning['latitude'], None,
                                        'count', bins=[lon_bnd, lat_bnd]).statistic,
                                        dims=['y', 'x']).rename('lightning_counts').astype('int')

        pcld = xr.DataArray(stats.binned_statistic_2d(scn['longitude'].values.ravel(), scn['latitude'].values.ravel(),
                                        scn['cloud_pressure_crb'].values.ravel()/100, # hPa
                                        'min', bins=[lon_bnd, lat_bnd]).statistic,
                                        dims=['y', 'x']).rename('pcld')

        ptropo = xr.DataArray(stats.binned_statistic_2d(scn['longitude'].values.ravel(), scn['latitude'].values.ravel(),
                                        scn['ptropo'].values.ravel(), # hPa
                                        'min', bins=[lon_bnd, lat_bnd]).statistic,
                                        dims=['y', 'x']).rename('ptropo')

        psc = xr.DataArray(stats.binned_statistic_2d(scn['longitude'].values.ravel(), scn['latitude'].values.ravel(),
                                        scn['apparent_scene_pressure'].values.ravel(), # hPa
                                        'min', bins=[lon_bnd, lat_bnd]).statistic,
                                        dims=['y', 'x']).rename('psc')

        # pick the data in swath mask
        lightning_counts = lightning_counts.where(scn_resample['mask'].drop_vars('crs').fillna(0), 0)
        pcld = pcld.where(scn_resample['mask'].drop_vars('crs').fillna(0))
        ptropo = ptropo.where(scn_resample['mask'].drop_vars('crs').fillna(0))
        psc = psc.where(scn_resample['mask'].drop_vars('crs').fillna(0))

    else:
        lightning_counts = xr.full_like(scn_resample['mask'].drop_vars('crs'), 0).rename('lightning_counts').astype('int')
        pcld = xr.full_like(scn_resample['mask'].drop_vars('crs'), np.nan).rename('pcld')
        ptropo = xr.full_like(scn_resample['mask'].drop_vars('crs'), np.nan).rename('ptropo')
        psc = xr.full_like(scn_resample['mask'].drop_vars('crs'), np.nan).rename('psc')

    # merge into Dataset
    ds = xr.merge([lightning_counts, pcld, ptropo, psc])
    # remove attributes
    ds.attrs = []
    ds['lightning_counts'].attrs = []
    ds['pcld'].attrs = []
    ds['ptropo'].attrs = []
    ds['psc'].attrs = []

    ds['lightning_counts'].attrs['description'] = f"Lightning count during {cfg['delta_time']} minutes before TROPOMI overpass"
    ds['pcld'].attrs['description'] = f"Minimum cloud pressure in grids"
    ds['ptropo'].attrs['description'] = f"Minimum tropopause pressure in grids"
    ds['psc'].attrs['description'] = f"Minimum apparent scene pressure in grids"

    # add time dim
    ds = ds.expand_dims('time').assign_coords(time=('time', [t_overpass]))
    ds['time'] = ds['time'].astype('datetime64[ns]')

    # add swath coord
    swath = os.path.basename(filename).split('_')[-4]
    lightning_counts.coords['swath'] = swath

    # add lon/lat dim
    ds = ds.rename({'y': 'longitude', 'x': 'latitude'})
    ds.coords['longitude'] = lon_center
    ds.coords['latitude'] = lat_center


    # remove data
    del scn, scn_resample['mask'], scn_resample
    gc.collect()

    return ds


def main():
    # get all filenames based on requested date range
    pattern = os.path.join(cfg['s5p_dir'], '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
    filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], [])

    # for file in filelist:
    #     print(process_data(file))
    #     break

    with Pool(3) as p:
        res = p.map(process_data, filelist)

    output = xr.concat(res, 'time')

    # set encoding
    comp = dict(zlib=True, complevel=7)
    enc = {var: comp for var in output.data_vars}

    # export file
    st = req_dates[0].strftime('%Y%m%d')
    et = req_dates[-1].strftime('%Y%m%d')

    savename = cfg['output_data_dir'] + f'/swath_lightning_crf{str(int(crf_low*100))}_' + st + '_' + et + '.nc'
    logging.info(f'export to {savename}')
    output.to_netcdf(path=savename,
                     engine='netcdf4',
                     encoding=enc)


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)

    overwrite = cfg.get('overwrite', 'True') == 'True'

    # set the low limit of cloud radiance fraction (CRF)
    crf_low = 0

    # generate data range
    req_dates = pd.date_range(start=cfg['start_date'],
                              end=cfg['end_date'],
                              freq='D')

    # create target grid, increase the resolution higher than 0.05 will get more accurate results
    lon_resolution = 0.5
    lat_resolution = 0.5
    lon_bnd = np.arange(-180, 180+lon_resolution, lon_resolution)
    lat_bnd = np.arange(60, 90+lat_resolution, lat_resolution)
    lon_center = np.convolve(lon_bnd, np.ones(2), 'valid') / 2
    lat_center = np.convolve(lat_bnd, np.ones(2), 'valid') / 2

    main()
