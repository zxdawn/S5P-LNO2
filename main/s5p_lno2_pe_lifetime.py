'''
INPUT:
    - Product file named "S5P_LNO2_production.nc" or "S5P_LNO2_lifetime.nc",
        generated by `s5p_lno2_product.py`
    - "kind" keywords
        - production: file for calculating production efficiency
                      Read "S5P_LNO2_production.nc"
        - lifetime: file for calculting lighting NO2 lifetime using swaths containing but no recent lightning
                    Read "S5P_LNO2_lifetime.nc"

OUTPUT:
    CSV file with S5P LNO2 infos

UPDATE:
    Xin Zhang:
       2022-05-19: Basic version
'''

import os
import numpy as np
import pandas as pd
import logging
import xarray as xr
from netCDF4 import Dataset
from s5p_lno2_utils import Config
from functools import partial
from multiprocessing import Pool
import cartopy.io.shapereader as shpreader
import shapely.vectorized
from shapely.ops import unary_union
from shapely.prepared import prep


# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


def get_geom(name, category='physical', resolution='50m'):
    # https://stackoverflow.com/questions/47894513/
    #    checking-if-a-geocoordinate-point-is-land-or-ocean-with-cartopy
    shp_fname = shpreader.natural_earth(name=name,
                                        resolution=resolution,
                                        category=category
                                       )
    geom = unary_union(list(shpreader.Reader(shp_fname).geometries()))

    return prep(geom)


def calc_pe(filename, case, swaths, tau=6):
    """Calculate production efficiency

    t0: time of reference swath
    t1: time of selected swath
    """
    res = []
    for index, swath in enumerate(swaths):
        logging.info(f'Processing {swath}')

        # load data
        ds_s5p_t1 = xr.open_dataset(filename, group=case+'/'+swath+'/S5P')
        ds_lightning_t1 = xr.open_dataset(filename, group=case+'/'+swath+'/Lightning')
        area_t1 = ds_s5p_t1['area']
        area = area_t1.sum()
        overpass_t1 = ds_s5p_t1['time']
        pasp = ds_s5p_t1['apparent_scene_pressure']

        # calculate the summation of no2 (mol)
        no2_t1 = (ds_s5p_t1['nitrogendioxide_tropospheric_column']*area_t1).sum()
        lno2_t1 = (ds_s5p_t1['lno2']*area_t1).sum()
        try:
            lno2_geo_t1 = (ds_s5p_t1['lno2geo']*area_t1).sum()
        except:
            lno2_geo_t1 = (ds_s5p_t1['lno2_geo']*area_t1).sum()
        lno2_vis_t1 = (ds_s5p_t1['lno2vis']*area_t1).sum()

        # calculate the mean lon/lat of LNO2 pixels
        lon_center = ds_s5p_t1['longitude'].where(ds_s5p_t1['lno2'].notnull().isel(time=0)).mean().values
        lat_center = ds_s5p_t1['latitude'].where(ds_s5p_t1['lno2'].notnull().isel(time=0)).mean().values

        if shapely.vectorized.contains(land_geom, lon_center, lat_center):
            region = 'land'
        else:
            region = 'ocean'

        # get previous overpass time
        if index > 0:
            # read previous swath's data
            logging.info(' '*4+f'Reading previous swath: {swaths[index-1]}')
            ds_s5p_t0 = xr.open_dataset(filename, group=case+'/'+swaths[index-1]+'/S5P')
            ds_lightning_t0 = xr.open_dataset(filename, group=case+'/'+swaths[index-1]+'/Lightning')
            overpass_t0 = ds_s5p_t0['time'].values
            area_t0 = ds_s5p_t0['area']

            # subset lightning to period between two swaths
            ds_lightning_t1['delta'][:] = (ds_lightning_t1['time'] - overpass_t0) / np.timedelta64(1, 'h')
            ds_lightning_t1.load()
            delta_t1 = ds_lightning_t1.where(ds_lightning_t1['delta'] > 0, drop=True)['delta'].to_numpy()[:,np.newaxis]

            # how many lightning happened between swaths
            nlightning = delta_t1.size

            # efold_lifetime is the sum of factor
            efold_lifetime = np.exp(-delta_t1/tau).sum()

            # calculate the summation of lno2
            lno2_t0 = (ds_s5p_t0['lno2']*area_t0).sum()
            try:
                lno2_geo_t0 = (ds_s5p_t0['lno2geo']*area_t0).sum()
            except:
                lno2_geo_t0 = (ds_s5p_t0['lno2_geo']*area_t0).sum()
            lno2_vis_t0 = (ds_s5p_t0['lno2vis']*area_t0).sum()

            # calculate production efficiency
            if nlightning > 0:
                logging.info(' '*4+'Calculating production efficiency ...')
                pe_lno2geo = (lno2_geo_t1 - lno2_geo_t0 * np.exp(-(overpass_t1 - overpass_t0)/ np.timedelta64(1, 'h')/tau)) / efold_lifetime
                pe_lno2vis = (lno2_vis_t1 - lno2_vis_t0 * np.exp(-(overpass_t1 - overpass_t0)/ np.timedelta64(1, 'h')/tau)) / efold_lifetime
                pe_lno2 = (lno2_t1 - lno2_t0 * np.exp(-(overpass_t1 - overpass_t0)/ np.timedelta64(1, 'h')/tau)) / efold_lifetime

                pe_lno2geo = pe_lno2geo.item()
                pe_lno2vis = pe_lno2vis.item()
                pe_lno2 = pe_lno2.item()
            else:
                pe_lno2 = np.nan
                pe_lno2geo = np.nan
                pe_lno2vis = np.nan
        else:
            # how many lightning happened 100 mins before the overpass
            nlightning = sum(ds_lightning_t1['delta'] > -100).values

            pe_lno2 = np.nan
            pe_lno2geo = np.nan
            pe_lno2vis = np.nan

        res.append([overpass_t1.dt.strftime('%Y-%m-%dT%H:%M').values[0], case, swath, lon_center, lat_center,
                    region, nlightning, area.values, pasp.min().values,
                    no2_t1.values, lno2_geo_t1.values, lno2_vis_t1.values, lno2_t1.values,
                    pe_lno2geo, pe_lno2vis, pe_lno2])

    df = pd.DataFrame(res, columns=['time', 'case', 'swath', 'longitude', 'latitude', 'region', 'nlightning', 'area', \
                                    'apparent_scene_pressure', 'no2', 'lno2geo', 'lno2vis', 'lno2', \
                                    'pe_lno2geo', 'pe_lno2vis', 'pe_lno2'])

    return df


def process_data(filename, res, case):
    ds_nc = Dataset(filename)
    # get the group names of swath inside Case
    swaths = list(sorted(ds_nc[case].groups.keys()))
    df = calc_pe(filename, case, swaths, tau=6)
    #res.append(df)
    return df


def main():
    filename = os.path.join(data_dir, nc_file)
    ds_nc = Dataset(filename)

    # get the group names of Cases
    cases = list(sorted(ds_nc.groups.keys()))

    res = []
    func = partial(process_data, filename, res)
    #with Pool(processes=int(cfg['num_pool'])) as pool:
    with Pool(processes=8) as pool:
        output = pd.concat(pool.map(func, cases), axis=0)
        print(output)
        pool.close()
        pool.join()

    #for case in cases:
    #    # get the group names of swath inside Case
    #    swaths = list(sorted(ds_nc[case].groups.keys()))
    #    df = calc_pe(filename, case, swaths)
    #    res.append(df)

    # combine into one DataFrame
    # output = pd.concat(res, axis=0)
    # print(output.to_string())
    savename = f'{savedir}/S5P_LNO2_{kind}.csv'
    #savename = f'{savedir}/S5P_LNO2_{kind}_tau8.csv'
    #savename = f'{savedir}/S5P_LNO2_{kind}_bkgd10.csv'
    logging.info(f'Saved to {savename}')
    output.to_csv(f'{savename}', index=False)

if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)
    kind = 'production' # 'production' or 'lifetime'

    if kind == 'production':
        nc_file = 'S5P_LNO2_production.nc'
    elif kind == 'lifetime':
        nc_file = 'S5P_LNO2_lifetime.nc'

    data_dir = cfg['output_data_dir']
    savedir = cfg['output_data_dir']

    land_geom = get_geom('land')
    
    main()
