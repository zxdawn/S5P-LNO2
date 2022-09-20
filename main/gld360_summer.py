'''
INPUT:
    - VAISALA GLD360 gridded lightning stroke data (gld-stroke-count-m0p1.asc)
OUTPUT:
    - The original GLD360 data in human-readable format
    - Summation of summertime lightning data (gld_stroke_summer.csv)
UPDATE:
    Xin Zhang:
       2022-06-03: Basic version
'''

import logging
import pandas as pd
import numpy as np
from s5p_lno2_utils import Config
logging.basicConfig(level=logging.INFO)

cfg = Config('settings.txt')
lightning_dir = cfg['lightning_dir']


def read_section(df):
    """Read the sections"""
    # get the index of each section
    idx_meta = df.index[df[0] == 'METADATA'][0]
    idx_dimension = df.index[df[0] == 'DIMENSIONS'][0]
    idx_fields = df.index[df[0] == 'FIELDS'][0]
    idx_data = df.index[df[0] == 'DATA-SPARSE'][0]

    # read the data (meta_data, dimensions, fields, data) between sections
    meta_data = df.loc[idx_meta+1:idx_dimension-1]
    dimensions = df.loc[idx_dimension+1:idx_fields-1]
    dimensions.columns = ['dimensions']
    fields = df.loc[idx_fields+1:idx_data-1]
    fields.columns = ['fields']
    data = df.loc[idx_data+1:]
    data.columns = ['data']

    return dimensions, fields, data


def read_dim(dimensions, varname):
    """Read the dimension info"""
    # get the data and delete the varname
    #   start_value, end_value, number of bins
    var_list = dimensions[dimensions['dimensions'].str.startswith(varname)]['dimensions'].values[0].split(',')[1:]
    # convert into float type
    var_list = list(map(float, var_list))

    if varname != 'epochSeconds':
        # lon and lat
        resolution = (var_list[1]-var_list[0])/var_list[2]
        return np.linspace(var_list[0]+resolution/2, var_list[1]-resolution/2,  int(var_list[2]))
    else:
        # time
        st = var_list[0]
        dt = (var_list[1]-var_list[0])/var_list[2]
        return st, dt


def sparse_to_human(df, lon, lat, st, dt):
    """convert the sparse data into human-readable data"""
    # copy the bin index for group bins later
    df['latitude_index'] = df['latitude']
    df['longitude_index'] = df['longitude']
    # convert into lon and lat
    df['latitude'] = df['latitude'].apply(lambda x: lat[int(x)])
    df['longitude'] = df['longitude'].apply(lambda x: lon[int(x)])
    # convert into datetime
    df['time'] = pd.to_datetime(df['epochSeconds'].apply(lambda x: st+dt*int(x)), unit='s')

    # clean DataFrame
    df.drop(columns='epochSeconds', inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df.astype({'eventCount': int, 'latitude_index': int, 'longitude_index': int})

    return df

def read_gld360(filename):
    logging.info('Reading original file')
    df = pd.read_csv(filename, sep="\t", header=None)

    # read sections
    logging.info('Reading sections')
    dimensions, fields, data = read_section(df)

    # read dim infos
    logging.info('Reading dimensions')
    lat = read_dim(dimensions, 'latitude')
    lon = read_dim(dimensions, 'longitude')
    st, dt = read_dim(dimensions, 'epochSeconds')

    # get the column names: dimensions + fields
    data_columns = list(dimensions['dimensions'].str.split(',').str[0]) + list(fields['fields'])

    # split the data column and save
    data[data_columns] = data['data'].str.split(',',expand=True)

    # drop the useless data and reset the index
    logging.info('Convert to human-readable format')
    df = sparse_to_human(data.drop(columns='data').reset_index(drop=True), lon, lat, st, dt)

    return df, lon, lat


if __name__ == '__main__':
    # read gld360 data
    filename = f'{lightning_dir}/gld-stroke-count-m0p1.asc'
    df, lon, lat = read_gld360(filename)

    # group monthly data by lon/lat bins defined by GLD360
    logging.info('Group data')
    geo_sum_monthly = df.groupby([pd.Grouper(freq='1M'), 'longitude_index', 'latitude_index'])['eventCount'].sum().to_xarray()

    # fill no lightning grid with nan
    # https://stackoverflow.com/q/68207994/7347925
    geo_sum_monthly = geo_sum_monthly.reindex({'longitude_index': range(0, len(lon)),
                                            'latitude_index': range(0, len(lat))},
                                            fill_value=np.nan)

    # use the lon/lat as coordinates instead of index 
    geo_sum_monthly = geo_sum_monthly.rename({'longitude_index': 'longitude', 'latitude_index': 'latitude'})
    geo_sum_monthly.coords['longitude'] = lon
    geo_sum_monthly.coords['latitude'] = lat

    # set compression
    comp = dict(zlib=True, complevel=7)
    enc = {geo_sum_monthly.name: comp}

    # export summer netcdf file
    savename = f'{lightning_dir}/gld_stroke_summer.nc'
    logging.info(f'Export summertime data to {savename}')
    geo_sum_monthly.to_netcdf(savename, engine='netcdf4', encoding=enc)

    # export the original file in human-readable format
    savename = f'{lightning_dir}/gld-stroke-count-m0p1.csv'
    logging.info(f'Export original data to human-readable csv file: {savename}')
    df.to_csv()#, index=False)