'''
INPUT:
    Dates

OUTPUT:
    S5P CTMANA file
'''

import pandas as pd
from glob import glob
from datetime import datetime, timedelta
from pytropomi.downs5p import downs5p

ctmana_dir = './'

sdate = '2020-06-01'
edate = '2020-08-31'

dates = pd.date_range(sdate, edate, freq='1D')
dates_1day = dates+timedelta(days=1)

for index, date in enumerate(dates):
    savepath = ctmana_dir + date.strftime('%Y%m')
    file = savepath + date.strftime('/S5P_OPER_AUX_CTMANA_%Y%m%dT*')
    if len(glob(file)) > 0:
        print(f'{file} exists ...')
    else:
        downs5p(producttype='AUX_CTMANA', beginPosition=date, endPosition=dates_1day[index], savepath=savepath)
