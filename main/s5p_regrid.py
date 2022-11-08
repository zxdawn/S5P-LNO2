'''
INPUT:
    - S5P (TROPOPMI) L2 product files
OUTPUT:
   Regridded  NetCDF file
UPDATE:
    Xin Zhang:
        2022-09-04: basic version
'''

import os
import harp
import time
import numpy as np
from glob import glob
from datetime import datetime
from multiprocessing import Pool
from s5p_lno2_utils import Config

cfg = Config('settings.txt')

num_pool = 8
tropomi_dir = cfg['s5p_dir']+'/20**/'
save_dir = './tropomi_regrid_qa75/'

lat_bins = np.linspace(60, 90, 301)
lon_bins = np.linspace(-180, 180, 3601)
#lat_bins = np.arange(60, 90.1, 0.1)
#lon_bins = np.arange(-180, 180.1, 0.1)
bin_spatial = f'bin_spatial({tuple(lat_bins)}, {tuple(lon_bins)})'


def pre_process(filename):
    '''Subset valid data in the Arctic'''
    operations = ';'.join(['latitude >= 60[degree_north]',
                           'tropospheric_NO2_column_number_density_validity > 75',
                           #'scene_pressure - 0.98*surface_pressure > 0',
                           'keep(datetime_start, orbit_index, latitude, longitude, \
                                 validity, solar_zenith_angle,  cloud_fraction, \
                                 cloud_pressure, tropospheric_NO2_column_number_density)',
                           ])

    try:
        product = harp.import_product(filename, operations)
        return product

        #pqf = product.validity.data.astype('uint32') & 0b1111111
        #product.pqf = harp.Variable(pqf, ["time", ])
        #product_pqf = harp.execute_operations(product, operations='pqf==0')

        #return product_pqf

    except harp.NoDataError:
        print(f'No valid data for {filename}')
        return None


def regrid(product):
    '''Regrid products'''
    operations = ';'.join(['keep(datetime_start, orbit_index, latitude, longitude, \
                                 solar_zenith_angle, cloud_fraction, cloud_pressure, tropospheric_NO2_column_number_density)',
                           bin_spatial,
                           'derive(latitude {latitude})',
                           'derive(longitude {longitude})',
                           ])

    regridded_product = harp.execute_operations(product, operations=operations)

    return regridded_product


def save_data(product, filename):
    '''Export data to new NetCDF files'''
    file_base = os.path.basename(filename)
    # create export path
    year = file_base[20:24]
    month = file_base[24:26]
    export_path = save_dir+year+month+'/'
    # create dir
    os.makedirs(export_path, exist_ok=True)
    # set savename
    prefix = '_'.join(file_base.split('_')[:-1])
    savename = prefix+datetime.now().strftime('_%Y%m%dT%H%M%S.nc')
    print(f'Exporting to {export_path+savename}')
    harp.export_product(product, export_path+savename, file_format='netcdf')


def process_tropomi(filename):
    product = pre_process(filename)
    if product:
        regrid_product = regrid(product)
        save_data(regrid_product, filename)


files = sorted(glob(f'{tropomi_dir}*.nc', recursive=True))

start_time = time.time()

# multiprocessing
pool = Pool(num_pool)
pool.map(process_tropomi, files)

print("--- %s seconds ---" % (time.time() - start_time))
pool.close()
