'''
INPUT:
    - VAISALA GLD360 gridded lightning stroke data (gld-stroke-count-m0p1.asc)
OUTPUT:
    - The daily GLD360 data in csv format (<lightning_dir>/<yyymm>/<yyyymmdd>.csv)
UPDATE:
    Xin Zhang:
       2022-06-03: Basic version
'''


import logging
import pandas as pd
from pathlib import Path
from s5p_lno2_utils import Config
logging.basicConfig(level=logging.INFO)
from gld360_summer import read_gld360

cfg = Config('settings.txt')
lightning_dir = cfg['lightning_dir']

filename = f'{lightning_dir}/gld-stroke-count-m0p1.asc'
df, _, _ = read_gld360(filename)

logging.info(f'Exporting daily files to {lightning_dir} now ...')

for name, group in df.groupby(df.index.date):
    logging.debug(name)
    # set the export path and filename
    save_dir = f"{lightning_dir}/{name.strftime('%Y%m')}/"
    save_name = f"{name.strftime('%Y%m%d.csv')}"

    # create the savedir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # optional: duplicate the row based on the eventCount
    #   it can help us track lightning NO2 later,
    # https://stackoverflow.com/a/47336762
    duplicated_group = pd.DataFrame(group.values.repeat(group.eventCount, axis=0), columns=group.columns,
                                    index=group.index.repeat(group.eventCount))
    duplicated_group.index.names = ['timestamp']
    duplicated_group[['longitude','latitude','eventCount']].to_csv(save_dir+save_name, float_format='%.1f')

    # # if you just care the density, then it's fine to use `group` directly
    # group[['longitude','latitude','eventCount']].to_csv(save_dir+save_name, float_format='%.1f')