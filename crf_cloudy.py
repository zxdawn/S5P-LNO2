'''
INPUT:
    - S5P (TROPOPMI) L2 product files

OUTPUT:
    - csv file of cloudy data, which can be used to generate histogram

    variables: lon, lat, sza, cp, and crf varibales
    conditions:
        - lat > 0
        - crf >= 0.7
        - sza < 80

UPDATE:
    Xin Zhang:
        2022-11-08: basic version
'''


import logging
import os
import time
from glob import glob
from multiprocessing import Pool

import pandas as pd
from satpy import Scene

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger('satpy').setLevel(logging.ERROR)

num_pool = 8
sdate = '2019-06-01'
edate = '2019-08-31'
s5p_dir = '../data/tropomi/'
save_dir = '../data/tropomi_crf/'
req_dates = pd.date_range(start=sdate, end=edate, freq='D')


def load_s5p(f_s5p):
    logging.debug(' '*4+f'Reading {f_s5p} ...')
    scn = Scene([f_s5p], reader='tropomi_l2')

    #vnames = ['latitude', 'longitude', 'cloud_radiance_fraction_nitrogendioxide_window', 'cloud_pressure_crb', 'solar_zenith_angle']
    vnames = ['cloud_radiance_fraction_nitrogendioxide_window', 'cloud_pressure_crb', 'solar_zenith_angle']

    logging.debug(' '*4 + f'Reading vnames: {vnames}')
    scn.load(vnames)

    # pick crf >= 0.7
    ds = scn.to_xarray_dataset()
    ds = ds.where(scn['cloud_radiance_fraction_nitrogendioxide_window']>=0.7, drop=True)

    # stack and filter again
    ds = ds.stack(z=['y', 'x'])
    ds = ds.where((ds['cloud_radiance_fraction_nitrogendioxide_window']>=0.7)&(ds['solar_zenith_angle']<80)&(ds['latitude']>0), drop=True)

    return ds.drop_vars(['crs', 'y', 'x', 'z']).to_dataframe()

def save_data(df, filename):
    output_file = os.path.join(save_dir,
                               os.path.basename(filename)[20:26],
                               os.path.basename(filename)[:-19]+'.csv'
                               )
    if not os.path.isdir(save_dir+os.path.basename(filename)[20:26]):
        os.makedirs(save_dir+os.path.basename(filename)[20:26])

    logging.info(f'Saving to {output_file}')
    df.to_csv(output_file, index=False)

def process_tropomi(filename):
    df = load_s5p(filename)
    save_data(df, filename)


def main():
    # get all filenames based on requested date range
    pattern = os.path.join(s5p_dir, '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
    filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], [])
    
    # for filename in filelist:
    #     process_tropomi(filename)
    #     break

    start_time = time.time()
    
    # multiprocessing
    pool = Pool(num_pool)
    pool.map(process_tropomi, filelist)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    pool.close()

if __name__ == '__main__':
    main()
