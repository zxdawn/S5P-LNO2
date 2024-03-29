'''
INPUT:
    - S5P (TROPOPMI) L2 product files
    - ENGLN lightning flash csv files or VAISALA lightning density/flash files
    - fresh_lightning_cases.csv or nolightning_cases.csv generated by `s5p_lno2_cases.py`

OUTPUT:
    JPG figures of selected cases

UPDATE:
    Xin Zhang:
        2022-07-13: basic version
'''

import gc
import logging
import os

import pandas as pd
import proplot as pplt
import xarray as xr
from pandarallel import pandarallel

from s5p_lno2_utils import Config

pplt.rc.reso = 'med'

# Disable warnings
from warnings import filterwarnings

filterwarnings("ignore")
# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


def read_data(filename):
    ds_tropomi = xr.open_dataset(filename, group='S5P').isel(time=0) \
                    [['nitrogendioxide_tropospheric_column', 'qa_value', 'processing_quality_flags',
                      'solar_zenith_angle', 'cloud_radiance_fraction_nitrogendioxide_window',
                      'apparent_scene_pressure', 'cloud_pressure_crb', 'nitrogendioxide_segmentation', 'lightning_mask']]

    ds_lightning = xr.open_dataset(filename, group='Lightning')

    return ds_tropomi, ds_lightning


def plot_scatter(ax, ds_lightning, level, cmap='Browns5_r', loc='r'):
    s = ax.scatter(ds_lightning['longitude_pred'].sel(level=level),
                   ds_lightning['latitude_pred'].sel(level=level),
                   marker="$\u25EF$", cmap=cmap,
                   # cmap_kw = {'left': 0.05, 'right': 1},
                   c=ds_lightning['delta'], s=6,
                   vmax=0,
                   # alpha=0.5
                   )
    ax.colorbar(s, loc=loc, label=f'Relative Flash Time ({level} hPa)')
    return s


def plot_func(ds_tropomi, ds_lightning, case_index, frame, mask_label):
    """Plot data

    NO2 VCD, PQF, SZA, Segmentation, CRF, CP, tracer+mask, and masked NO2 VCD

    """
    # get the data inside each labeled mask
    lightning_mask = ds_tropomi['lightning_mask'] == mask_label
    ds_lightning_mask = ds_tropomi.where(lightning_mask, drop=True)
    lon_mask = ds_lightning_mask.coords['longitude']
    lat_mask = ds_lightning_mask.coords['latitude']

    lon_min = lon_mask.min().values
    lon_max = lon_mask.max().values
    lat_min = lat_mask.min().values
    lat_max = lat_mask.max().values

    # create another mask for plotting NO2 in case of the bug of pcolormesh cross the pole
    ds_mask = ds_tropomi.where((ds_tropomi.coords['longitude'] >= lon_min) &
                               (ds_tropomi.coords['longitude'] <= lon_max) &
                               (ds_tropomi.coords['latitude'] >= lat_min) &
                               (ds_tropomi.coords['latitude'] <= lat_max),
                               drop=True)

    # general setting
    plot_set = {'x': 'longitude', 'y': 'latitude',
                'xlim': (lon_min, lon_max), 'ylim': (lat_min, lat_max),
                'discrete': False}

    fig, axs = pplt.subplots(nrows=3, ncols=3, span=False, sharey=3, sharex=3)

    ax = axs[0]
    # in case there's almost no valid data
    if ds_lightning_mask['nitrogendioxide_tropospheric_column'].sizes['x'] <= 1 or \
        ds_lightning_mask['nitrogendioxide_tropospheric_column'].sizes['y'] <= 1:
        print('no valid data ...')
        return
    ds_mask['nitrogendioxide_tropospheric_column'].plot(**plot_set,
                                                        vmin=0, vmax=4e-5,
                                                        cmap='Thermal',
                                                        cbar_kwargs={"label": "[mol m$^{-2}$]",
                                                                     # 'orientation':'horizontal',
                                                                     'loc': 'bottom'},
                                                        ax=ax)
    ax.format(title='NO$_2$ VCD')

    ax = axs[1]
    pqf_mask = 0b1111111

    pqf = ds_mask['processing_quality_flags'].to_masked_array().astype('uint32') & pqf_mask
    ax.pcolormesh(ds_mask['longitude'], ds_mask['latitude'], (pqf != 0), cmap='tab10', levels=2)
    ax.format(title='Processing Quality Flag \n (Blue is 0)')

    # ds_mask['qa_value'].plot(**plot_set, vmin=0, vmax=1,
    #                                            cmap='Paired',
    #                                            cbar_kwargs={"label": ""},
    #                                            ax=ax)
    # ax.format(title='qa_value')

    ax = axs[2]
    ds_mask['solar_zenith_angle'].plot(**plot_set,
                                       cmap='Thermal',
                                       cbar_kwargs={"label": "[degree]"},
                                       ax=ax)
    ax.format(title='Solar Zenith Angle')

    ax = axs[3]
    ds_mask['nitrogendioxide_segmentation'].where(ds_mask['nitrogendioxide_segmentation'] > 0)\
        .plot(**plot_set, cmap='Accent', add_colorbar=False, ax=ax)
    ax.format(title='Segmentation of NO$_2$ VCD')

    ax = axs[4]
    ds_mask['cloud_radiance_fraction_nitrogendioxide_window']\
        .plot(**plot_set,
              vmin=0.5, vmax=1,
              cmap='Spectral',
              # cmap_kw={'left':0.2, 'right':0.95},
              cbar_kwargs={"label": ""},
              extend='max',
              ax=ax
              )
    ax.format(title='Cloud Radiance Fraction')

    ax = axs[5]
    (ds_mask['cloud_pressure_crb']/1e2).plot(**plot_set,
    # (ds_mask['apparent_scene_pressure']/1e2).plot(**plot_set,
                                             vmin=150, vmax=700,
                                             cmap='Spectral_r',
                                             # cmap_kw={'left':0.05, 'right':0.8},
                                             cbar_kwargs={"label": "[hPa]"},
                                             ax=ax
                                             )
    ax.format(title='Cloud Pressure')
    # ax.format(title='Apparent Scene Pressure')

    ax = axs[6]
    # plot transported lightning tracer at three pressure levels
    # plot_scatter(ax, ds_lightning, 650, cmap='Blue2', loc='l')
    plot_scatter(ax, ds_lightning, 500, cmap='Blue2', loc='l')
    # plot_scatter(ax, ds_lightning, 450, cmap='Purples2')
    # plot_scatter(ax, ds_lightning, 350, cmap='Browns3', loc='r')
    plot_scatter(ax, ds_lightning, 300, cmap='Browns3', loc='r')

    # plot the contour of mask
    ax.contour(lightning_mask.longitude, lightning_mask.latitude, lightning_mask,
               levels=[1], colors=['red5'])

    recent_lightning = (ds_lightning['delta'].where(ds_lightning['lightning_label'] == mask_label) > -100).sum().values
    ax.format(title=f"Horizontal transport of lightning tracer \n Flash Count in 100 min : {recent_lightning} \n Mask label: {mask_label}")

    ax = axs[7]
    ds_lightning_mask['nitrogendioxide_tropospheric_column'].plot\
        (**plot_set,
         vmin=0, vmax=ds_lightning_mask['nitrogendioxide_tropospheric_column'].max(),
         cmap='Thermal',
         cbar_kwargs={"label": "[mol m$^{-2}$]"},
         ax=ax
         )
    ax.format(title='Masked NO$_2$ VCD')

    ax = axs[8]
    ds_lightning_mask['nitrogendioxide_tropospheric_column'].plot(**plot_set,
                                                                  cmap='Thermal',
                                                                  cbar_kwargs={"label": "[mol m$^{-2}$]"},
                                                                  ax=ax
                                                                  )
    ax.format(title='Masked NO$_2$ VCD')

    # in case the point cross 180E
    if (len(ds_lightning['longitude']) > 0) and (lon_max - lon_min > 180):
        lon_east = [n for n in ds_lightning['longitude'] if n >= 0]
        if len(lon_east) > 0:
            lon_min = min(lon_east).values
        else:
            # sometimes the mask is wrong
            lon_min = ds_lightning['longitude'].min().values
            lon_max = ds_lightning['longitude'].max().values

    axs.format(xlim=(lon_min, lon_max), ylim=(lat_min, lat_max),
               suptitle=ds_tropomi.s5p_filename,
               xformatter='deglat', yformatter='deglon',
               xlabel='', ylabel='',
               )

    savedir = os.path.join(cfg['output_fig_dir'], ds_tropomi.attrs['s5p_filename'][20:26])

    if not os.path.exists(savedir):
        try:
            os.makedirs(savedir)
        except:
            # sometimes it tries to create the dir at the same time
            pass

    savename = os.path.join(savedir, f"Case{case_index}_Frame{frame}_"+ds_tropomi.attrs['s5p_filename'][:-19]+f'_{mask_label}.jpg')
    logging.info(f'Saving to {savename}')
    fig.savefig(savename, dpi=300)

    del ds_tropomi, ds_lightning
    gc.collect()


def plot_data(grp):
    """Plot linked cases

    one grp is one case which has row data for each timestamp

    """
    frame = 0
    for _, row in grp.iterrows():
        ds_tropomi, ds_lightning = read_data(row['filename'])
        plot_func(ds_tropomi, ds_lightning, row['case'], frame, row['label'])
        frame += 1


def main():
    # read case file
    filename = output_data_dir + f'/{kind}_cases.csv'
    df = pd.read_csv(filename)

    # plot each case group
    pandarallel.initialize(nb_workers=5, progress_bar=False)
    df.groupby(['case']).parallel_apply(plot_data)
    # df.groupby(['case']).apply(plot_data)


if __name__ == '__main__':
    # read config file
    cfg = Config('settings.txt')
    overwrite = cfg.get('overwrite', 'True') == 'True'
    output_data_dir = cfg['output_data_dir']
    logging.info(cfg)

    kind = 'fresh_lightning'  # fresh_lightning or no_lightning

    main()
