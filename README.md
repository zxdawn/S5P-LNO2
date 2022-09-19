# S5P-LNO2

## Work flow

![workflow](workflow.jpg)

Users need to prepare the *Input Data* first and then run the scripts one by one.

Please feel free to modify the *settings.txt* file for your own research.



Detailed explanations of main scripts:

1. Select lightning-swaths with both lightning and high NO2, and save them to nc files. [*s5p_lno2_main.py*]
2. Link the consecutive lightning-swath cases and save variables (case No., filename, and mask label) to csv files. [*s5p_lno2_cases.py*]
2. Combine the generated csv file into fresh\_lightning\_cases.csv or nolightning\_cases.csv.  [*s5p_lno2_cases_sum.py*]
2. (Optional) Plot linked variables and save them as images to filter cases manually [*s5p_lno2_plot.py*]. We are still thinking how to convert the manual part into an auto one.
3. Extract TM5 no2\_vmr and temperature profiles for consecutive lightning-swaths. [*s5p_lno2_tm5_extract.py*]
5. Calculate lightning variables (AMFs, SCD_Bkgd, tropopause_pressure, lno2vis, lno2_geo and lno2) and save them to one netcdf file called "S5P_LNO2.nc" [*s5p_lno2_product.py*]
6. Calculate lightning NO2 production efficiency and save all useful variables to one CSV file called "S5P_LNO2_PE.csv". [*s5p_lno2_pe.py*]

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
