'''
INPUT:
    - CSV file which has S5P (TROPOPMI) filenames
       generated by s5p_lno2_cases_sum.py
    - S5P Look-up table (LUT)
    - "kind" keywords
        - production: file for calculating production efficiency
                      Read "fresh_lightning_cases.csv"
        - lifetime: file for calculting lighting NO2 lifetime using swaths containing but no recent lightning
                    Read "no_lightning_cases.csv"

OUTPUT:
    One netcdf file named "S5P_LNO2_production.nc" or "S5P_LNO2_lifetime.nc"
        which has S5P data and lightning NO2 variables

UPDATE:
    Xin Zhang:
       2022-05-16: Basic
       2022-07-21: Set the 30th percentile of SCD_Trop as SCD_bkgd
'''

import logging
import os
from glob import glob
from warnings import filterwarnings

import numpy as np
import pandas as pd
import tobac
import xarray as xr
from distfit import distfit
# from pyproj import Geod
# from shapely.geometry import Polygon

from s5p_lnox_amf import cal_amf, cal_bamf, cal_tropo, scene_mode
from s5p_lnox_utils import Config

filterwarnings("ignore")
# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


def update_vars(ds):
    ds['amf_geo'] = 1/np.cos(np.radians(ds['solar_zenith_angle'])) + 1/np.cos(np.radians(ds['viewing_zenith_angle']))

    ds['SCD_Strato'] = ds['nitrogendioxide_stratospheric_column'] * ds['air_mass_factor_stratosphere']
    ds['SCD_Trop'] = ds['nitrogendioxide_slant_column_density'] - ds['SCD_Strato']

    ds['SCD_Strato'] = ds['SCD_Strato'].rename('SCD_Strato')
    ds['SCD_Trop'] = ds['SCD_Trop'].rename('SCD_Trop')
    ds['SCD_Strato'].attrs['units'] = 'mol m-2'
    ds['SCD_Trop'].attrs['units'] = 'mol m-2'

    # calculate pressure levels
    a = ds['tm5_constant_a']
    b = ds['tm5_constant_b']
    psfc = ds['surface_pressure']

    low_p = (a[:, 0] + b[:, 0]*psfc)/1e2
    high_p = (a[:, 1] + b[:, 1]*psfc)/1e2

    ds['bot_p'] = low_p
    ds['top_p'] = high_p

    itropo = ds['tm5_tropopause_layer_index']

    ds['ptropo'] = cal_tropo(ds['bot_p'], itropo)


# def pixel_area(ds_tropomi, row):
#     """Calculate the area (using one pixel to represent all picked pixels in the mask)"""
#     pixel = ds_tropomi['longitude'].where(ds_tropomi['lightning_mask']==row['label'], drop=True).isel(x=[0, 1], y=[0,1])
#     corners = np.column_stack([pixel.coords['longitude'].values.ravel(), pixel.coords['latitude'].values.ravel()])
#     corners[[-2,-1]] = corners[[-1,-2]]
#     geod = Geod(ellps="WGS84")
#     area, poly_perimeter = geod.geometry_area_perimeter(Polygon(corners))
#     return area


def feature(threshold, target='maximum',
            position_threshold='weighted_diff',
            coord_interp_kind='nearest',
            sigma_threshold=0,
            min_distance=0,
            n_erosion_threshold=0,
            n_min_threshold=20):
    '''Set keyword arguments for the feature detection step'''
    parameters_features = {}
    parameters_features['target'] = target

    # diff between specific value and threshold for weighting when finding the center location (instead of just mean lon/lat)
    parameters_features['position_threshold'] = position_threshold

    # we want to keep the original x/y instead of interpolated x/y
    # https://github.com/climate-processes/tobac/pull/51
    parameters_features['coord_interp_kind'] = coord_interp_kind

    # for slightly smoothing (gaussian filter)
    parameters_features['sigma_threshold'] = sigma_threshold

    # Minumum number of cells above/below threshold in the feature to be tracked
    # parameters_features['min_num'] = 4

    # K, step-wise threshold for feature detection
    parameters_features['threshold'] = threshold

    # minimum distance between features
    parameters_features['min_distance'] = min_distance

    # pixel erosion (for more robust results)
    parameters_features['n_erosion_threshold'] = n_erosion_threshold

    # minimum number of contiguous pixels for thresholds
    parameters_features['n_min_threshold'] = n_min_threshold

    return parameters_features


def get_feature(ds_mask, alpha_high):
    # fill few nan values
    scd_no2 = ds_mask['SCD_Trop'].rio.write_crs(4326).rio.write_nodata(np.nan).rio.interpolate_na(method='linear').drop_vars('spatial_ref').rename('scdTrop')
    scd_no2.load()

    # normalize the SCD data
    scd_no2_norm = (scd_no2 - scd_no2.min()) / (scd_no2.max() - scd_no2.min())

    # Initialize model
    dist = distfit(distr='lognorm', alpha=alpha_high)

    # Find best theoretical distribution for empirical data X
    dist.fit_transform(scd_no2_norm.stack(z=['y', 'x'])[~np.isnan(scd_no2_norm.stack(z=['y', 'x']))].values, verbose=0)

    dxy = 5000  # data resolution; Unit: m

    min_threshold = dist.model['CII_max_alpha']
    max_threshold = 1
    num_threshold = 10

    # calculate the range of threshold
    threshold = np.linspace(min_threshold, max_threshold, num_threshold)  # multi-thresholds for tracking

    # set parameters_features
    parameters_features = feature(threshold)

    # get features
    features = tobac.themes.tobac_v1.feature_detection_multithreshold(scd_no2_norm.expand_dims('time'), dxy, **parameters_features)

    return threshold, scd_no2, scd_no2_norm, features


def segmentation(threshold, target='maximum', method='watershed'):
    '''Set keyword arguments for the segmentation step'''
    parameters_segmentation = {}
    parameters_segmentation['target'] = target
    parameters_segmentation['method'] = method
    # until which threshold the area is taken into account
    parameters_segmentation['threshold'] = threshold

    return parameters_segmentation


def calc_lno2vis(scd_no2, scd_no2_norm, ds_mask, ds_amf, threshold, alpha_bkgd, features, crf_min):
    """Get the segmentation of NO2 and calculate LNO2Vis"""
    # set parameters_segmentation
    parameters_segmentation = segmentation(np.min(threshold))

    # get masks and paired features
    dxy = 5000  # data resolution; Unit: m
    masks_no2, features_no2 = tobac.themes.tobac_v1.segmentation(features, scd_no2_norm.expand_dims('time'), dxy, **parameters_segmentation)

    masks_no2 = masks_no2.where(masks_no2 > 0).rename({'dim_0': 'y', 'dim_1': 'x'})

    # set the 30th percentile of SCD_Trop over the non-LNO2 region as background NO2
    scd_no2_bkgd = scd_no2.where(xr.ufuncs.isnan(masks_no2)).quantile(0.3)

    lno2_vis = (ds_mask['SCD_Trop'] - scd_no2_bkgd) / ds_amf['amfTropVis']
    lno2_geo = (ds_mask['SCD_Trop'] - scd_no2_bkgd) / ds_mask['amf_geo']
    lno2_vis.load()
    # lno2_vis = xr.DataArray(lno2_vis, dims=['y', 'x']).assign_coords(longitude=ds_mask['longitude'])

    # lno2_mask = ds_mask['lightning_mask']>0
    # lno2_mask = (ds_mask['lightning_mask']>0)&(ds_mask['cloud_radiance_fraction_nitrogendioxide_window'] > crf_min)

    # update crf
    scene_mode(ds_mask)
    lno2_mask = (ds_mask['cloud_radiance_fraction_nitrogendioxide_window'] > crf_min) & ~xr.ufuncs.isnan(masks_no2)

    return lno2_mask, scd_no2_bkgd, lno2_vis, lno2_geo


def read_file(row, lut, tau=6, crf_min=0, alpha_high=0.2, alpha_bkgd=0.5, peak_width=60, peak_offset=0):
    # read processed S5P data file
    filename = row['filename']
    ds_tropomi = xr.open_dataset(filename, group='S5P').isel(time=0)
    ds_lightning = xr.open_dataset(filename, group='Lightning')

    # update some useful variables
    update_vars(ds_tropomi)

    # pick the labeled part
    lightning_mask = ds_tropomi['lightning_mask'] == row['label']

    # read the pixel areas
    ds_area = xr.open_dataset('s5p_pixel_area.nc')
    if ds_tropomi.sizes['y'] > 4000:
        # 3245 or 3246 scanlines (4172 or 4173 after the along-track pixel size reduction)
        area = ds_area['area_high'][:ds_tropomi.sizes['y'], :ds_tropomi.sizes['x']].rename({'y_high': 'y', 'x_high': 'x'})
    elif ds_tropomi.sizes['y'] > 3300:
        # sometimes it is 3736
        area = ds_area['area_medium'][:ds_tropomi.sizes['y'], :ds_tropomi.sizes['x']].rename({'y_medium': 'y', 'x_medium': 'x'})
    else:
        area = ds_area['area_low'][:ds_tropomi.sizes['y'], :ds_tropomi.sizes['x']].rename({'y_low': 'y', 'x_low': 'x'})

    varnames = ['surface_albedo_nitrogendioxide_window', 'cloud_albedo_crb', 'surface_pressure', 'cloud_pressure_crb',
                'snow_ice_flag', 'cloud_fraction_crb_nitrogendioxide_window', 'cloud_radiance_fraction_nitrogendioxide_window',
                'solar_zenith_angle', 'viewing_zenith_angle', 'solar_azimuth_angle', 'viewing_azimuth_angle', 'amf_geo',
                'apparent_scene_pressure', 'scene_albedo', 'bot_p', 'top_p',
                'no2_vmr', 'temperature', 'tm5_tropopause_layer_index',
                'nitrogendioxide_tropospheric_column', 'air_mass_factor_troposphere', 'SCD_Trop', 'lightning_mask']

    # Dataset in the lightning mask
    ds_mask = ds_tropomi[varnames].where(lightning_mask, drop=True)

    # add area to Dataset
    ds_mask['area'] = area.where(lightning_mask, drop=True)

    # calculate pressure levels
    ds_mask_pclr = xr.concat([ds_mask['bot_p'], ds_mask['top_p'][-1, ...]], dim='layer')
    ptropo = cal_tropo(ds_mask_pclr, ds_mask['tm5_tropopause_layer_index'])

    # calculate mean time as overpass time
    t_overpass = pd.to_datetime(xr.broadcast(ds_tropomi['time_utc'], lightning_mask)[0]
                                .where(lightning_mask, drop=True).stack(z=['y', 'x']).dropna(dim='z')).mean()
    t_overpass = t_overpass.to_datetime64()

    # calculate box-AMF
    s5p_origin, bAmfClr, bAmfCld, del_lut = cal_bamf(ds_mask, lut)

    # calculate AMFs
    ds_amf = cal_amf(ds_mask, s5p_origin,
                     xr.merge([ds_mask['no2_vmr'].rename('no2'), ds_mask['temperature'].rename('tk')]),
                     bAmfClr, bAmfCld)

    # detect the feature using the masked SCD NO2
    threshold, scd_no2, scd_no2_norm, features = get_feature(ds_mask, alpha_high)

    if not features:
        # no features are detected
        # copy dataarray
        lno2_mask = ptropo.copy().rename('lno2_mask')
        scd_no2_bkgd = ptropo.copy().rename('scdBkgd')
        amf_lno2 = ptropo.copy().rename('amflno2')
        lno2_vis = ptropo.copy().rename('lno2vis')
        lno2_geo = ptropo.copy().rename('lno2geo')
        lno2 = ptropo.copy().rename('lno2')

        # set value to nan
        lno2_mask[:] = 0  # zero means no lightning NO2 is detected
        scd_no2_bkgd[:] = np.nan
        amf_lno2[:] = np.nan
        lno2_vis[:] = np.nan
        lno2_geo[:] = np.nan
        lno2[:] = np.nan

        return t_overpass, ds_mask, ds_amf, ds_lightning, ptropo, scd_no2_bkgd, amf_lno2, lno2_mask, lno2_vis, lno2_geo, lno2
    else:
        # calculate LNO2Vis and LNO2Geo
        #   lno2_mask is the segmentation of SCDTrop larger than background SCD
        lno2_mask, scd_no2_bkgd, lno2_vis, lno2_geo = calc_lno2vis(scd_no2, scd_no2_norm, ds_mask, ds_amf, threshold, alpha_bkgd, features, crf_min)

        # subset data by segmentation mask
        # update the area as the segmentation area, instead of the large lightning mask area
        #   ds_mask['area'] = ds_mask['area'].where(lno2_mask)
        pcld = ds_mask['cloud_pressure_crb'].where(lno2_mask)/100

        # set the cloud pressure as peak pressure disturbed by peak_offset
        peak_pressure = pcld.min() + peak_offset
        logging.debug(f'Peak pressure is set as {peak_pressure}')

        # create the a priori lightning NO2 profile
        # Gaussian distribution of LNO2: $a * e^\frac{{-{(x - b)}^2}}{2c^{2}}$
        #   a is neglected because the division of integration offsets it
        factor = -1.0 / (2 * np.power(peak_width, 2))
        pclr = xr.concat([ds_mask['bot_p'], ds_mask['top_p'][-1, ...]], dim='layer')
        pclr = pclr.rolling({pclr.dims[0]: 2}).mean()[1:, ...]
        lno2_priori = np.exp(np.power(pclr - peak_pressure, 2) * factor)

        # calculate AMF_LNO2 which depends on the shape instead of quantity,
        #   and get the VCD_LNO2 which is named "no2Trop" in the Dataset
        ds_amflno2 = cal_amf(ds_mask, s5p_origin,
                             xr.merge([lno2_priori.rename('no2'), ds_mask['temperature'].rename('tk')]),
                             bAmfClr, bAmfCld)

        lno2 = ds_amflno2['no2Trop']
        amf_lno2 = ds_amflno2['amfTrop']

        return t_overpass, ds_mask, ds_amf, ds_lightning, \
               ptropo.rename('tropopause_pressure'), scd_no2_bkgd.rename('scdBkgd'), \
               amf_lno2.rename('amflno2'), \
               lno2_mask.rename('lno2_mask'), \
               lno2_vis.rename('lno2vis').where(lno2_mask), \
               lno2_geo.rename('lno2geo').where(lno2_mask), \
               lno2.rename('lno2').where(lno2_mask)


def save_data(case_num, ds_group, ds_lightning, savedir):
    """Save masked data"""
    # get the saving path
    output_file = os.path.join(savedir, f'S5P_LNO2_{kind}_test.nc')

    # set compression
    comp = dict(zlib=True, complevel=7)

    # get the No. of swath
    swath_name = ds_group.attrs['s5p_filename'].split('_')[-4]
    logging.info(' '*4 + f'Saving swath: {swath_name}')

    logging.debug(' '*8 + 'Saving S5P products ...')
    enc = {var: comp for var in ds_group.data_vars}

    if os.path.isfile(output_file):
        mode = 'a'
    else:
        mode = 'w'

    ds_group.to_netcdf(path=output_file,
                       group=f'Case{case_num}/Swath{swath_name}/S5P',
                       engine='netcdf4',
                       encoding=enc,
                       mode=mode)

    logging.debug(' '*8 + 'Saving lightning ...')
    enc = {var: comp for var in ds_lightning.data_vars}

    ds_lightning.to_netcdf(path=output_file,
                            group=f'Case{case_num}/Swath{swath_name}/Lightning',
                            engine='netcdf4',
                            encoding=enc,
                            mode='a')

    return output_file


def main():
    if kind == 'production':
        csv_file = 'fresh_lightning_cases.csv'
    elif kind == 'lifetime':
        csv_file = 'no_lightning_cases.csv'

    savedir = cfg['output_data_dir']
    df = pd.read_csv(os.path.join(savedir, csv_file))

    for case_num, df_group in df.groupby('case'):
        logging.info(f'Case: {case_num}')
        for row_id, row in df_group.iterrows():
            t_overpass, ds_mask, ds_amf, ds_lightning, ptropo, scd_no2_bkgd, amflno2, lno2_mask, lno2vis, lno2geo, lno2 = read_file(row, lut)

            # merge calculated variables into one Dataset
            ds_merge = xr.merge([ptropo, scd_no2_bkgd, amflno2, lno2_mask, lno2vis, lno2geo, lno2])

            # assign time dim
            ds_mask = ds_mask.assign_coords(time=t_overpass).expand_dims('time')
            ds_amf = ds_amf.assign_coords(time=t_overpass).expand_dims('time')
            ds_merge = ds_merge.assign_coords(time=t_overpass).expand_dims('time')

            # merge all Datasets
            ds_group = xr.merge([ds_mask, ds_amf, ds_merge])
            logging.debug('S5P Dataset', ds_group)
            logging.debug('Lightning Dataset', ds_lightning)

            # save data
            output_file = save_data(case_num, ds_group, ds_lightning, savedir)

    logging.info(f'Saved to {output_file}')


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    logging.info(cfg)
    kind = 'production'  # 'production' or 'lifetime'

    # load the LUT data
    lut_pattern = glob(cfg['s5p_dir']+'/S5P_OPER_LUT_NO2AMF*')
    lut = xr.open_mfdataset(lut_pattern, combine='by_coords')

    # directory where clean lightning cases are saved
    clean_dir = cfg['output_data_dir'] + '/clean_lightning/'

    main()
