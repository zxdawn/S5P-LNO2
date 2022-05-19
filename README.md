# S5P-LNO2

## Work flow

1. Select lightning-swaths with both lightning and high NO2, and save them to nc files. [*s5p_lnox_main.py*]
2. Find the consecutive lightning-swaths and save to csv files. [s5p_lnox_consecutive_swaths.py]
3. Extract TM5 no2\_vmr and temperature profiles for consecutive lightning-swaths. [*s5p_lnox_tm5_extract.py*]
4. Link the consecutive lightning-swaths and
   - calculate lightning variables (AMFs, SCD_Bkgd, tropopause_pressure, lno2vis, lno2_geo and lno2) and save them to one netcdf file called "S5P_LNO2.nc" [*s5p_lnox_link.py*]
   - plot linked variables and save them to images [*s5p_lnox_plot_tracks.py*]
5. Calculate lightning NO2 production efficiency and save all useful variables to one CSV file called "S5P_LNO2_PE.csv". [*s5p_lnox_pe.py*]

For details about the products mentioned above, please check **[User's Guide]()**.

## Input Data

The used data are listed below.

The input paths are shown in parentheses. Please feel free to modify them in `settings.txt`.

1. TROPOMI (`<s5p_dir>/<yyymm>/S5P_**__L2__NO2____`)

   There're three main methods of downloading the TROPOMI NO2 L2 data:

   - [Sentinel-5P Pre-Operations Data Hub](https://s5phub.copernicus.eu/dhus/#/home)
   - [GES DISC](https://disc.gsfc.nasa.gov/datasets/S5P_L2__NO2____HiR_1/summary)
   - [S5P-PAL](https://data-portal.s5p-pal.com/): this is the reprocessed NO2 data from April 2018 - September 2021. The two sources above will be updated soon.

2. [ERA5](https://doi.org/10.24381/cds.bd0915c6) (`<era5_dir>/era5_<yyyymm>.nc`)

   The pressure level (200 - 700 hPa) ERA5 data (u and v) are used to predict the transport of lightning air in the upper troposphere.

   Note that for the TROPOMI data on the first day of month, we need the ERA5 data on the last day of the previous month.

   e.g. `S5P...20220201...` needs `era5_202201.nc` which at least has the data on 2022-01-31.

3. Lightning Data (`<lightning_dir>/<yyymm>/<yyyymmdd>.csv`)

   The lightning data should be saved in CSV format and have at least three fields: timestamp, longitude, and latitude.
