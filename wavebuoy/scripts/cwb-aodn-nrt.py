#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

from wavebuoy.wavebuoy import WaveBuoy
from wavebuoy.sofar.api import SofarAPI
from wavebuoy.netcdf.lookup import NetCDFFileHandler



load_dotenv()


if __name__ == "__main__":

    # Args handling
    """
    - Implement arguments:
        - output_path (probably a path in UWA server)
        - start/end dates for whole pipeline(extraction, processing)
        - start/end dates for qc
        - qc window from present to past (in hours, days, months, etc)

    """

    period_to_process = timedelta(days=90) # months
    period_to_qualify = 24 # hours



    # Loading metadata
    """ 
    - Load buoys_metadata.csv as Pandas DataFrame
    - Get site_ids (site names)
    - Get Sofar API Tokens
    - 
    """

    wb = WaveBuoy()
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata_token_sorted)    
    
     
    
    # General ETL idea
    """
    - Iterate over each site_id
    - Handle error for each site_name
    - Reuse aodn logging approach
    - 

    """

    for site_id in wb.site_ids:
        print(site_id)
        sofar_api.check_token_iteration(next_token=wb.buoys_metadata_token_sorted.loc[site_id, 'sofar_token'])
        
        spotter_id = sofar_api.get_spot_id(site_id=site_id, buoys_metadata=wb.buoys_metadata_token_sorted)
        spotter_obj = sofar_api.select_spotter_obj_from_spotter_grid(spot_id=spotter_id,
                                                                spotter_grid=sofar_api.spotter_grid,
                                                                devices=sofar_api.devices)
        
        latest_available_datetime = sofar_api.get_latest_available_datetime(spotter_obj=spotter_obj)

        print(latest_available_datetime)

        nc_file_paths_list = wb.lookup_netcdf_files(site_id=site_id, latest_available_datetime=latest_available_datetime)

        print(nc_file_paths_list)

        print("="*10)

        break

    # Extract
    """
    Algoritm 1
    - Get current date_time (focus on current month)
    - Get last available date_time for that month
    - Get last downloaded date_time from the NetCDF files
    -

    Algoritm 2
    - Get current date_time
    - Get last available date_time from SofarAPI
    - Get last downloaded date_time (latest NetCDF file)
    . using with open OR loading the whole file
    - Extract data SofarAPI using previous date_times
    -
    """


    # Transform
    """
    - Convert to Pandas DataFrame
    - Convert date_times, date_types, etc
    - Outputs Pandas DataFrame
    
    """

    # Qualify
    """
    - check if current extraction has 24h hours 
        . if not, load last NetCDF file as a pandas DataFrame
            - Check if last NetCDF file has XX hours from current date
                . if not, load second latest NetCDF file as a Pandas DataFrame
    - 
    . range test
    . spike test
    - check if 
    

    """


    # Final Transform (Generate NetCDF)
    """
    - Generate monthly NetCDFs 
    *note: use AODN naming convention:
    template -> IMOS_COASTAL-WAVE-BUOYS_YYYYMMDD_SITENAME_RT_WAVE-PARAMETERS_monthly.nc
    e.g. IMOS_COASTAL-WAVE-BUOYS_20241101_HILLARYS_RT_WAVE-PARAMETERS_monthly.nc
    -
    
    """


    # Load



