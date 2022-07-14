'''
INPUT:
    - S5P (TROPOPMI) L2 product files
    - S5P CTMANA data
    - fresh_lightning_cases.csv (default) or nolightning_cases.csv generated by `s5p_lno2_cases_sum.py`

OUTPUT:
    Append two variables 'no2_vmr' and 'temperature' to S5P (TROPOPMI) L2 product files

UPDATE:
    Xin Zhang:
       2022-05-19: Basic version
'''


import os
import logging
from glob import glob

import pandas as pd
import xarray as xr
from datetime import timedelta
from s5p_lnox_utils import Config
from TM5_profile import extract_orbit

# read config file
cfg = Config('settings.txt')
logging.info(cfg)

ctmana_dir = cfg['ctmana_dir']
output_data_dir = cfg['output_data_dir']

# get the consecutive swaths
filelist = pd.read_csv(output_data_dir+'/fresh_lightning_cases.csv')['filename']  # or nolightning_cases.csv

# # if you want to extract CTMANA for all files, uncomment these lines below
#clean_dir = cfg['output_data_dir'] + '/clean_lightning/'  # directory to only save clean lightning cases
#
## generate data range
#req_dates = pd.date_range(start=cfg['start_date'],
#                          end=cfg['end_date'],
#                          freq='D')
#
#pattern = os.path.join(clean_dir, '{}{:02}', 'S5P_*_L2__NO2____{}{:02}{:02}T*')
#filelist = sum([glob(pattern.format(date.year, date.month, date.year, date.month, date.day)) for date in req_dates], []) 

for file in filelist:
    print('Adding NO2 and Temperature profiles to ', file)
    try:
        with xr.open_dataset(file, group='S5P') as ds:
            st = pd.to_datetime(ds.attrs['time_coverage_start'])
            et = pd.to_datetime(ds.attrs['time_coverage_end'])

        # get the middle time of orbit
        mt = st + (et - st)/2
        print('Middle time of time_coverage: ', mt)
        
        # if middle time >= 23:45, use the CTMANA on the next day
        if (mt.hour == 23) & (mt.minute >= 45):
            mt = mt + timedelta(days=1)

        # set datetime string
        prefix = mt.strftime('%Y%m%d')
        suffix = (mt + timedelta(days=1)).strftime('%Y%m%d')
        ctm_name = glob(ctmana_dir+mt.strftime('/%Y%m/')+f'S5P_OPER_AUX_CTMANA_{prefix}T000000_{suffix}T000000_*.nc')[0]
        print('Used CTMANA file: ', ctm_name)
        import time
        start_time = time.time()
        extract_orbit(ctm=ctm_name, species='no2', l2_file=file, verbose=1)
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print('!!!Empty dataset!!!')
