'''
INPUT:
    - S5P (TROPOPMI) L2 product files
OUTPUT:
    S5P pixel area file (s5p_pixel_area.nc)
UPDATE:
    Xin Zhang:
       2022-05-18: Basic version
'''

import xarray as xr
from glob import glob
from satpy import Scene
from pyproj import Geod
import numpy as np
from functools import lru_cache


def load_s5p(f_s5p):
    """Load S5P longitude and latitude bounbary data"""
    scn = Scene(f_s5p, reader='tropomi_l2')
    scn.load(['assembled_lon_bounds', 'assembled_lat_bounds'])

    return scn


@lru_cache(maxsize=2**10)
def calc_area():
    """Calculate area of each pixel"""
    geod = Geod(ellps="WGS84")

    len_x = lon_bnds.shape[0]-1
    len_y = lon_bnds.shape[1]-1
    area = np.full((len_x, len_y), 0)

    for x in range(len_x):
        for y in range(len_y):
            # get the corner coordinates
            lons = lon_bnds[x:x+2, y:y+2].ravel()
            lats = lat_bnds[x:x+2, y:y+2].ravel()

            # clockwise direction
            lons[-2], lons[-1] = lons[-1], lons[-2]
            lats[-2], lats[-1] = lats[-1], lats[-2]

            # get the area
            poly_area, poly_perimeter = geod.polygon_area_perimeter(lons, lats)
            area[x, y] = poly_area

    return area

def main():
    # get example file of low resolution and high resolution
    #   because of the along-track pixel size reduction after 6 August 2019
    scn_low = load_s5p(glob('./tropomi/201908/*___20190805T193408*'))
    scn_high = load_s5p(glob('./tropomi/201908/*___20190806T191502*'))

    # lru_cache needs global variable   
    global lon_bnds, lat_bnds 

    lon_bnds = scn_low['assembled_lon_bounds'].values
    lat_bnds = scn_low['assembled_lat_bounds'].values
    area_low = xr.DataArray(calc_area(), dims=['y_low', 'x_low']).rename('area_low')
    area_low.attrs['units'] = 'm2'
    area_low.attrs['description'] = 'Area of TROPOMI low-resolution pixels before 6 August 2019'

    # clear the cache because the lon_bnds and lat_bnds are updated below
    calc_area.cache_clear()

    lon_bnds = scn_high['assembled_lon_bounds'].values
    lat_bnds = scn_high['assembled_lat_bounds'].values
    area_high = xr.DataArray(calc_area(), dims=['y_high', 'x_high']).rename('area_high')
    area_high.attrs['units'] = 'm2'
    area_high.attrs['description'] = 'Area of high-resolution TROPOMI pixels after 6 August 2019'

    # merge into one Dataset
    ds = xr.merge([area_low, area_high])
    ds.attrs['description'] = 'Area of TROPOMI pixels'

    # set encoding
    comp = dict(zlib=True, complevel=7)
    enc = {var: comp for var in ds.data_vars}

    # export file
    ds.to_netcdf(path='s5p_pixel_area.nc',
                 engine='netcdf4',
                 encoding=enc)

if __name__ == '__main__':
    main()

