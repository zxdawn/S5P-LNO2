'''
Modified from Xin Zhang's another repository:
    https://github.com/zxdawn/S5P-WRFChem/blob/master/main/s5p_utils.py
'''

import logging
import scipy
import pandas as pd
import numpy as np
import xarray as xr


def cal_dphi(saa, vaa):
    '''Calculate the relative azimuth angle (dphi)
    The relative azimuth angle is defined as
        the absolute difference between the viewing azimuth angle and
        the solar azimuth angle.
    It ranges between 0 and 180 degrees

    Ref: http://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
    '''

    delta = saa - vaa
    delta = delta.where(delta>=0, delta+360)
    delta = delta.where(delta<360, delta-360)
    dphi = abs(delta-180).rename('dphi')

    return dphi.rename('dphi')


def cal_tropo(pclr, itropo):
    '''Calculate tropopause pressure based on index'''

    # get fill_values and overwrite with 0, because index array should be int.
    # Check submitted issue by Xin:
    #   https://github.com/pydata/xarray/issues/3955
    #itropo = itropo.fillna(itropo._FillValue)
    #tropo_bool = itropo == itropo._FillValue
    itropo = itropo.where((itropo>=0)&(itropo<34))
    itropo = itropo.fillna(-999)
    tropo_bool = itropo == -999
    itropo = xr.where(tropo_bool, 0, itropo)

    # isel with itropo
    ptropo = pclr.isel(layer=itropo.load().astype(int))

    # mask data and set fill_value pixels to nan
    ptropo = xr.where(tropo_bool, np.nan, ptropo).rename('tropopause_pressure')

    return ptropo


def concat_p(layers, ptm5):
    '''Concatenate surface pressures, cloud pressures and tm5 pressures'''
    s5p_pcld = xr.concat([ptm5, xr.concat(layers, 'layer')], dim='layer')

    logging.debug(' '*6 + 'Sorting pressures ...')
    # get the sorting index
    sort_index = (-1*s5p_pcld.load()).argsort(axis=0)  # (layer, y, x)
    # sort pressures
    s5p_pcld = xr.DataArray(np.take_along_axis(s5p_pcld.values, sort_index, axis=0),
                            dims=['plevel', 'y', 'x'])
    # assign plevel coordinates
    s5p_pcld = s5p_pcld.assign_coords(plevel=range(s5p_pcld.sizes['plevel']))

    return s5p_pcld


def interp_to_layer(profile, pclr, pcld):
    '''Interpolate data to pressure levels including additional layer (surface/cloud)'''

    # calculate the full level pressure which is used by no2
    pclr = pclr.rolling({pclr.dims[0]: 2}).mean()[1:, ...]
    pcld = pcld.rolling({pcld.dims[0]: 2}).mean()[1:, ...]

    return np.exp(xr_interp(np.log(profile),
                            np.log(pclr), pclr.dims[0],
                            np.log(pcld), pcld.dims[0]).transpose(
                            pcld.dims[0],
                            ...,
                            transpose_coords=False))


def interp1d_sp(data, x, xi):
    '''Linear interpolate function'''
    f = scipy.interpolate.interp1d(x, data, fill_value='extrapolate')

    return f(xi)


def xr_interp(input_array,
              input_p, input_p_dimname,
              interp_p, interp_p_dimname):
    '''Interpolate 3D array by another 3D array
    Args:
        input_array:
                the original array
                - dims: input_p_dimname, y, x
        input_p:
                pressure of the original array
                - dims: input_p_dimname, y, x
        interp_p:
                the pressure levels which input_array is interpolated to
                - dims: interp_p_dimname, y, x
        input_p_dimname:
                the name of the vertical dim for input_p
        interp_p_dimname:
                the name of the vertical dim for interp_p
    '''

    logging.debug(' '*8 + f'Interpolating from {input_p_dimname} to {interp_p_dimname}')

    return xr.apply_ufunc(
        interp1d_sp,
        input_array.chunk({input_p_dimname: input_array.sizes[input_p_dimname],
                        'y': input_array.sizes['y'],
                        'x': input_array.sizes['x'],
                        }),
        input_p,
        interp_p,
        input_core_dims=[[input_p_dimname], [input_p_dimname], [interp_p_dimname]],
        output_core_dims=[[interp_p_dimname]],
        exclude_dims=set((input_p_dimname,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[input_array.dtype],
    )


def integPr(no2, s5p_p, psfc, ptropo):
    '''
    Integrate the vertical mixing ratios to get vertical column densities
    Because the first TM5 pressure level is the surface pressure
        and we use the tropopause pressure (tp) from TM5,
    we just need to add up the layers from the surface (l = 1)
        up to and including the tropopause level (l = l_{tp}^{TM5}).
    Args:
        no2: no2 vertical mixing ratio (ppp)
        s5p_p: pressure levels (hPa)
        psfc: surface pressure (hPa)
        ptropo: tropopause pressure (hPa)
    Return:
        integrate from psfc to ptropo
    Diagram:
        (Values of "layer" start from 0)
        [tm5_pressure]                           [layer]
        P34(index=33) ++++++++++++++++++++++++++ <--- TM5 top
                      -------------------------- layer32
                      ..........................
                      ..........................
        P26(index=25) ++++++++++++++++++++++++++ <-- Ptropo/integration top
                      -------------------------- layer24 (tropopause)
        P25           ++++++++++++++++++++++++++
                      ..........................
                      ..........................
        P3 (P2 upper) ++++++++++++++++++++++++++
                      -------------------------- layer1 (no2[1])
        P2 (P1 upper) ++++++++++++++++++++++++++
                      -------------------------- layer0 (no2[0])
        P1(index=0)   ++++++++++++++++++++++++++
        For this case, subcolumn is summed from layer0 to layer 32 and
                       sub_layer is from P1 to P26.
        As a result,   vcd should be summed layer0 to layer24,
                       which means sub_layer should be cropped using [:-1, ...]
    '''

    logging.debug(' '*8 + 'Integrating from surface to tropopause ...')

    # constants
    R = 287.3
    T0 = 273.15
    g0 = 9.80665
    p0 = 1.01325e5

    subcolumn = 10 * R * T0 / (g0*p0) \
                   * no2*1e6 \
                   * abs(s5p_p.diff(s5p_p.dims[0])) * 2.6867e16  # DU to moleclues/cm2

    sub_layer = (s5p_p <= psfc) & (s5p_p > ptropo)

    # sum from surface (ground or cloud pressure) to tropopause
    layername = no2.dims[0]
    vcd = subcolumn.where(sub_layer[:-1, ...].values).sum(layername, skipna=True)
    vcd = vcd.where(vcd!=0)

    logging.debug(' '*12 + 'Finish integration')

    return vcd


def assign_attrs(da, df_attrs):
    '''assign attributes to DataArray'''
    attrs = df_attrs.loc[da.name]

    return da.assign_attrs(units=attrs.loc['units'],
                           description=attrs.loc['description']
                           )


def scene_mode(s5p):
    '''The scene mode is turned on for two situations:

    1. cloud fractions > 1.0
    2. snow_ice_flag > 5

    Scene mode: the cloud fraction is set to 1
                the cloud albedo is set to the scene albedo
                and the cloud pressure is set to the scene pressure.
    '''

    mask = (s5p['snow_ice_flag'] > 5) | (s5p['cloud_fraction_crb_nitrogendioxide_window'] > 1)

    s5p['cloud_fraction_crb_nitrogendioxide_window']  = xr.where(mask, 1, s5p['cloud_fraction_crb_nitrogendioxide_window'])
    s5p['cloud_radiance_fraction_nitrogendioxide_window'] = xr.where(mask, 1, s5p['cloud_radiance_fraction_nitrogendioxide_window'])
    s5p['cloud_pressure_crb'] = xr.where(mask, s5p['apparent_scene_pressure'], s5p['cloud_pressure_crb'])
    s5p['cloud_albedo_crb'] = xr.where(mask, s5p['scene_albedo'], s5p['cloud_albedo_crb'])


def cal_bamf(s5p, lut):
    '''Calculate the Box-AMFs based on the LUT file
    Args:
        - albedo: Surface Albedo
        - dphi: Relative azimuth angle
        - mu: Cosine of viewing zenith angle
        - mu0: Cosine of solar zenith angle
        - p: Pressure Levels
        - p_surface: surface_air_pressure
        - sza: Solar zenith angle
        - vza: Viewing zenith angle
        - amf: Box air mass factor
    '''

    logging.debug(' '*4 + 'Calculating box-AMFs using LUT ...')

    new_dim = ['y', 'x']

    # bak up the original s5p
    s5p_origin = s5p.copy()

    # check whether switch to scene mode for each pixels
    scene_mode(s5p)

    # get vars from s5p data
    albedo = xr.DataArray(s5p['surface_albedo_nitrogendioxide_window'],
                          dims=new_dim)
    cloud_albedo = xr.DataArray(s5p['cloud_albedo_crb'],
                                dims=new_dim)

    # use surface pressure from TROPOMI (input from ERA-Interim)
    p_surface = xr.DataArray(s5p['surface_pressure']/1e2,  # hPa
                             dims=new_dim)
    p_cloud = xr.DataArray(s5p['cloud_pressure_crb']/1e2,  # hPa
                           dims=new_dim)

    # calculate angles
    dphi = xr.DataArray(cal_dphi(s5p['solar_azimuth_angle'],
                                 s5p['viewing_azimuth_angle']),
                        dims=new_dim)
    mu0 = xr.DataArray(np.cos(np.deg2rad(s5p['solar_zenith_angle'])),
                       dims=new_dim)
    mu = xr.DataArray(np.cos(np.deg2rad(s5p['viewing_zenith_angle'])),
                      dims=new_dim)

    da = lut['amf'].assign_coords(p=np.log(lut['amf'].p), p_surface=np.log(lut['amf'].p_surface))
    # da = da.where(da>0)

    # interpolate data by 2d arrays
    '''
    if you meet "KeyError: nan", see the method below:
        although xarray >= 0.16.1 fix this issue,
        regrid_dataset broken with xarray=0.16.1
            (https://github.com/pangeo-data/xESMF/pull/47)
        so, check https://github.com/pydata/xarray/pull/3924
            and edit the xarray/core/missing.py file by yourself
    '''
    bAmfClr_p = da.interp(albedo=albedo.clip(0,1),
                          p_surface=np.log(p_surface),
                          dphi=dphi,
                          mu0=mu0,
                          mu=mu)

    bAmfCld_p = da.interp(albedo=cloud_albedo.clip(0,1),
                          p_surface=np.log(p_cloud),
                          dphi=dphi,
                          mu0=mu0,
                          mu=mu)

    # interpolate to TM5 pressure levels
    bAmfClr = xr_interp(bAmfClr_p,
                        bAmfClr_p.coords['p'], 'p',
                        np.log(xr.concat([s5p['bot_p'].rolling({'layer': 2}).mean()[1:, ...], s5p['top_p'][-2:, ...].rolling({'layer': 2}).mean()[1:, ...]], dim='layer')),
                        'layer',
                        )
                        # np.log(s5p['p'].rolling({'layer': 2}).mean()[1:, ...].load()), 'layer').transpose(
                        # 'layer',
                        # ...,
                        # transpose_coords=False)

    bAmfCld = xr_interp(bAmfCld_p,
                        bAmfCld_p.coords['p'], 'p',
                        np.log(xr.concat([s5p['bot_p'].rolling({'layer': 2}).mean()[1:, ...], s5p['top_p'][-2:, ...].rolling({'layer': 2}).mean()[1:, ...]], dim='layer')),
                        'layer',
                        )
                        # np.log(s5p['p'].rolling({'layer': 2}).mean()[1:, ...].load()), 'layer').transpose(
                        # 'layer',
                        # ...,
                        # transpose_coords=False)

    # because the bAMF is normalized by amf_geo in the LUT file, we need to multiply bAMFs by amf_geo
    bAmfClr *= s5p['amf_geo']
    bAmfCld *= s5p['amf_geo']

    # # convert s5p['p'] to dask array
    # s5p['p'] = s5p['p'].chunk({'layer': s5p['p'].shape[0],
    #                            'y': s5p['p'].shape[1],
    #                            'x': s5p['p'].shape[2]})

    logging.debug(' '*8 + 'Finish calculating box-AMFs')

    return s5p_origin, bAmfClr, bAmfCld, [albedo, p_surface, cloud_albedo, p_cloud, dphi, mu0, mu]


def cal_amf(s5p, s5p_origin, interp_ds, bAmfClr, bAmfCld):
    '''Calculate AMFs'''
    logging.debug(' '*4 + 'Calculating AMFs based on box-AMFs ...')

    # get simulated profiles
    no2 = interp_ds['no2']
    tk = interp_ds['tk']

    # for LNOx research
    if 'no' in interp_ds.keys():
        no = interp_ds['no']
    if 'o3' in interp_ds.keys():
        o3 = interp_ds['o3']

    # the temperature correction factor, see TROPOMI ATBD file
    ts = 220  # temperature of cross-section [K]
    factor = 1 - 0.00316*(tk-ts) + 3.39e-6*(tk-ts)**2

    # load variables
    psfc = s5p['surface_pressure'] / 1e2  # hPa
    pcld = s5p['cloud_pressure_crb'] / 1e2  # hPa
    cf = s5p['cloud_fraction_crb_nitrogendioxide_window']
    crf = s5p['cloud_radiance_fraction_nitrogendioxide_window']
    itropo = s5p['tm5_tropopause_layer_index']
    # s5p_pclr = s5p['p']
    s5p_pclr = xr.concat([s5p['bot_p'], s5p['top_p'][-1, ...]], dim='layer')
    ptropo = cal_tropo(s5p_pclr, itropo)

    # set units
    psfc.attrs['units'] = 'hPa'
    pcld.attrs['units'] = 'hPa'

    # concatenate surface pressures, cloud pressures and tm5 pressures
    s5p_pcld = concat_p([psfc, pcld], s5p_pclr)

    # get the scattering weights
    bAmfClr = bAmfClr * factor
    bAmfCld = bAmfCld * factor

    # interpolate profiles to pressure levels including cloud pressure
    no2 = interp_to_layer(no2, s5p_pclr, s5p_pcld)
    bAmfClr = interp_to_layer(bAmfClr, s5p_pclr, s5p_pcld)
    bAmfCld = interp_to_layer(bAmfCld, s5p_pclr, s5p_pcld)
    clearSW = no2 * bAmfClr
    cloudySW = no2 * bAmfCld

    # for LNOx research
    if 'no' in interp_ds.keys():
        no = interp_to_layer(no, s5p_pclr, s5p_pcld).rename('noapriori')
    if 'o3' in interp_ds.keys():
        o3 = interp_to_layer(o3, s5p_pclr, s5p_pcld).rename('o3apriori')

    # logging.info(' '*6 + 'Calculating ghost column ...')
    # ghost = integPr(no2, s5p_pcld, psfc, pcld.rename('tropopause_pressure')).rename('ghost_column')

    # integrate from surface pressure to tropopause
    logging.debug(' '*6 + 'Calculating vcdGnd ...')
    vcdGnd = integPr(no2, s5p_pcld, psfc, ptropo).rename('vcdGnd')

    # integrate from cloud pressure to tropopause
    logging.debug(' '*6 + 'Calculating vcdCld ...')
    vcdCld = integPr(no2, s5p_pcld, pcld, ptropo).rename('vcdCld')

    # for LNOx research
    if 'no' in interp_ds.keys():
        logging.debug(' '*6 + 'Calculating vcdGnd_no ...')
        vcdGnd_no = integPr(no, s5p_pcld, psfc, ptropo).rename('vcdGnd_no')

    logging.debug(' '*6 + 'Calculating scdClr ...')
    scdClr = integPr(clearSW, s5p_pcld, psfc, ptropo).rename('scdClr')

    logging.debug(' '*6 + 'Calculating scdCld ...')
    scdCld = integPr(cloudySW, s5p_pcld, pcld, ptropo).rename('scdCld')

    # #set Cld DataArays to nan again for "clear" pixels
    # cld_pixels = (cf > 0) & (crf > 0) & (pcld > ptropo)
    # vcdCld = vcdCld.where(cld_pixels, 0)
    # scdCld = scdCld.where(cld_pixels, 0)

    # calculate AMFs
    logging.debug(' '*6 + 'Calculating amfClr ...')
    amfClr = scdClr / vcdGnd
    # https://github.com/pydata/xarray/issues/2283
    # amfClr = amfClr.where((crf != 1) & (cf != 1), 0)

    logging.debug(' '*6 + 'Calculating amfCld ...')
    amfCld = scdCld / vcdGnd
    # amfCld = amfCld.where(cld_pixels, 0)

    logging.debug(' '*6 + 'Calculating amf and amfVis ...')
    amf = amfCld*crf + amfClr*(1-crf)
    amfVis = amf*vcdGnd / (vcdCld*crf+vcdGnd*(1-crf))

    # calculate averaging kernel
    logging.debug(' '*6 + 'Calculating averaging kernel ...')
    sc_weights = crf*bAmfCld.where((s5p_pcld.rolling({s5p_pcld.dims[0]: 2}).mean()[1:, ...] < ptropo), 0) \
        + (1-crf)*bAmfClr
    avgKernel = sc_weights / amf

    # calculate vertical column densities
    scdTrop = s5p['nitrogendioxide_tropospheric_column'] * s5p['air_mass_factor_troposphere']
    no2Trop = scdTrop / amf
    no2TropVis = scdTrop / amfVis

    # set the variables changed by scene mode to original values
    s5p['cloud_fraction_crb_nitrogendioxide_window'] = s5p_origin['cloud_fraction_crb_nitrogendioxide_window']
    s5p['cloud_radiance_fraction_nitrogendioxide_window'] = s5p_origin['cloud_radiance_fraction_nitrogendioxide_window']
    s5p['cloud_pressure_crb'] = s5p_origin['cloud_pressure_crb']
    s5p['cloud_albedo_crb'] = s5p_origin['cloud_albedo_crb']

    # rename DataArrays
    amf = amf.rename('amfTrop')
    amfVis = amfVis.rename('amfTropVis')
    bAmfClr = bAmfClr.rename('swClr')
    bAmfCld = bAmfCld.rename('swCld')
    avgKernel = avgKernel.rename('avgKernel')
    no2 = no2.rename('no2apriori')
    s5p_pcld = s5p_pcld.rename('plevels')
    no2Trop = no2Trop.rename('no2Trop')
    no2TropVis = no2TropVis.rename('no2TropVis')

    # read attributes table
    df_attrs = pd.read_csv('./attrs_table.csv',
                           sep=' *, *',  # delete spaces
                           engine="python"
                           ).set_index('name')

    # drop useless coordinates and assign attributes
    da_list = []
    saved_da = [amf, amfVis, bAmfClr, bAmfCld, avgKernel,
                no2, no2Trop, no2TropVis,
                scdClr, scdCld, vcdGnd,
                ptropo, s5p_pcld.isel(plevel=slice(None, -1))]

    if 'no' in interp_ds.keys():
        if 'o3' in interp_ds.keys():
            saved_da.extend([o3, no, vcdGnd_no])
        else:
            saved_da.extend([no, vcdGnd_no])

    for da in saved_da:
        da_list.append(assign_attrs(da.drop(list(da.coords)), df_attrs))

    # merge to one Dataset
    ds = xr.merge(da_list)

    return ds
