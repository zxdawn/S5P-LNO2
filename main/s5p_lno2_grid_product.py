'''
INPUT:
    - S5P LNO2 NetCDF product <- s5p_lno2_product.py
OUTPUT:
    One netcdf file named "S5P_LNO2_grid_production.nc" which has gridded 0.125 x 0.125 variables
UPDATE:
    Xin Zhang:
       2023-01-14: Basic
'''

import os
import re
import logging
import numpy as np
import xarray as xr
from scipy import stats
from netCDF4 import Dataset
from pyresample.geometry import GridDefinition
from pyresample import kd_tree, geometry
from s5p_lno2_utils import Config

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


class s5p_lno2_grid:
    '''Plot the tropomi no2 with lightning images'''
    def __init__(self, filename):
        self.filename = filename
        self.ds = Dataset(self.filename)

        # get the group names of Cases
        self.cases = list(sorted(self.ds.groups.keys()))

        # define the output grid
        self.lon_bnd = np.arange(lon_min, lon_max+resolution, resolution)
        self.lat_bnd = np.arange(lat_min, lat_max+resolution, resolution)
        self.lon_center = np.convolve(self.lon_bnd, np.ones(2), 'valid') / 2 
        self.lat_center = np.convolve(self.lat_bnd, np.ones(2), 'valid') / 2 
        lons, lats = np.meshgrid(self.lon_center, self.lat_center)
        self.grid_def = GridDefinition(lons=lons, lats=lats)

    def process_data(self, case):
        # get the group names of swath inside Case
        self.swaths = list(sorted(self.ds[case].groups.keys()))

        lons, lats = [], []
        ds_list = []

        for swath in self.swaths:
            logging.info(f'Processing {case} {swath}')
            self.ds_orbit = self.read_s5p(case, swath)
            self.read_lightning(case, swath)
            orbit = '{:02}'.format(int(re.sub('[^0-9]','', case))) + '_' + re.sub('[^0-9]','', swath)
            self.ds_orbit = self.ds_orbit.expand_dims(dim={'orbit': [orbit]})
            ds_list.append(self.ds_orbit)

        ds_grid = xr.merge(ds_list)
        ds_grid['orbit'].attrs['description'] = '<case_number>_<s5p_orbit_number>'

        return ds_grid

    def read_s5p(self, case, swath):
        # load data
        ds_s5p = xr.open_mfdataset(self.filename, group=case+'/'+swath+'/S5P')
        # ds_lightning = xr.open_mfdataset(filename, group=case+'/'+swath+'/Lightning')

        no2 = ds_s5p['nitrogendioxide_tropospheric_column']
        lno2 = ds_s5p['lno2']
        pcld = ds_s5p['cloud_pressure_crb']
        lightning_label = np.unique(ds_s5p['lightning_mask'].values)
        self.lightning_label = lightning_label[~np.isnan(lightning_label)]

        # define swath
        self.swath_def = geometry.SwathDefinition(lons=ds_s5p['longitude'], lats=ds_s5p['latitude'])

        # resample data
        no2_grid = self.resample_2d(no2)
        lno2_grid = self.resample_2d(lno2)
        pcld_grid = self.resample_2d(pcld)

        ds = xr.merge([no2_grid, lno2_grid, pcld_grid])
        ds.attrs = ''

        return ds

    def read_lightning(self, case, swath):
        # load data
        ds_lightning = xr.open_mfdataset(filename, group=case+'/'+swath+'/Lightning')
        ds_lightning = ds_lightning.where(ds_lightning['lightning_label']==self.lightning_label, drop=True)

        res_lightning = []
        for lev in ds_lightning['level']:
            ds_level = ds_lightning.sel(level=lev)
            grid_lightning = stats.binned_statistic_2d(ds_level['latitude_pred'].values, ds_level['longitude_pred'].values, None, \
                                                       'count', bins=[self.lat_bnd, self.lon_bnd]).statistic
            da = xr.DataArray(grid_lightning, dims=['latitude', 'longitude'])
            res_lightning.append(da)
        
        da_lightning = xr.concat(res_lightning, dim='level').assign_coords(level=ds_lightning['level'], latitude=self.lat_center, longitude=self.lon_center)
        da_lightning['level'].attrs['units'] = 'hPa'

        # set nan pixels according to s5p data
        da_lightning = da_lightning.where(~self.ds_orbit['no2'].isnull().values)

        # add to Dataset
        self.ds_orbit['lightning_counts'] = da_lightning

    def resample_2d(self, data):
        '''resample 2d S5P data'''
        grid_coords = {'latitude': self.lat_center, 'longitude': self.lon_center}
        gridded_data = kd_tree.resample_nearest(self.swath_def, data.values, self.grid_def, radius_of_influence=10000, fill_value=None)
        da = xr.DataArray(gridded_data, dims=['latitude', 'longitude'], coords=grid_coords)
        da = da.rename(data.name)
        if da.name == 'nitrogendioxide_tropospheric_column':
            # use abbreviation
            da = da.rename('no2')
        if da.name == 'lno2':
            # in case the unit is not available
            units = 'mol m-2'
        else:
            units = data.attrs['units']
        da.attrs['units'] = units

        return da

def save_data(ds):
    '''export Dataset to NetCDF file'''
    # set compression
    comp = dict(zlib=True, complevel=7)
    enc = {var: comp for var in ds.data_vars}

    output_file = os.path.join(datadir, savename)

    logging.info(f'Saving to {savename}')
    ds.to_netcdf(path=output_file,
                 engine='netcdf4',
                 encoding=enc,
                )

def main():
    data = s5p_lno2_grid(filename)

    output = []

    for index in range(len(data.cases)):
        # get case number
        case = data.cases[index]

        # process data and plot
        ds_grid = data.process_data(case)
        output.append(ds_grid)

    # export to NetCDF
    ds = xr.concat(output, 'orbit').sortby('orbit')
    save_data(ds)


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)

    datadir = cfg['output_data_dir']
    filename = f'{datadir}/S5P_LNO2_production.nc'
    savename = 'S5P_LNO2_grid_product.nc'

    # define the resample region
    lon_min = -180
    lon_max = 180
    lat_min = 60
    lat_max = 90
    resolution = 0.1


    main()
