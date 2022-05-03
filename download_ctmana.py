'''
INPUT:
    - Date files (yyyymmdd, in each row) generated by s5p_lnox_link.py or your own script
OUTPUT:
    S5P CTMANA file
UPDATE:
    Xin Zhang:
       2022-05-03: Basic version
'''

import pandas as pd
from datetime import datetime, timedelta
from pytropomi.downs5p import downs5p
from s5p_lnox_utils import Config

cfg = Config('settings.txt')
ctmana_dir = cfg['ctmana_dir']

# generated by s5p_lnox_link.py
df = pd.read_csv('./ctmana_dates.csv', names=['date'], dtype={'date': 'str'}, parse_dates=['date'])

for index, row in df.iterrows():
    # download daily CTMANA file
    beginPosition = datetime(row.dt.year.values[0], row.dt.month.values[0], row.dt.day.values[0], 0)
    endPosition = beginPosition + timedelta(days=1)
    savepath = ctmana_dir+beginPosition.strftime('%Y%m')
    downs5p(producttype='AUX_CTMANA', beginPosition=beginPosition, endPosition=endPosition, savepath=savepath)

