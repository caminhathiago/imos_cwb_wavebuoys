#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.utils import args


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

    # vargs = args()

    ### TEMPORARY SETUP (REMOVE WHEN DONE)
    window = 24 # hours
    window_unit = "hours"
    period_to_qualify = 24 # hours
    ### END OF TEMPORARY SETUP


    # Loading metadata
    """ 
    - Load buoys_metadata.csv as Pandas DataFrame
    - Get site_ids (site names)
    - Get Sofar API Tokens
    - 
    """


    wb = WaveBuoy(buoy_type="sofar")
    # sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata_token_sorted)    

    ### TEMPORARY SETUP TO AVOID UNECESSARY SOFAR API CALLS (REMOVE WHEN DONE)
    import pickle
    with open("tests\sofar_api_object.pkl", "rb") as pickle_file:
        sofar_api = pickle.load(pickle_file)
    ### END OF TEMPORARY SETUP 
    
    # General ETL idea
    """
    - Iterate over each site_id
    - Handle error for each site_name
    - Reuse aodn logging approach
    - 

    """

    for idx, site in wb.buoys_metadata_token_sorted.iterrows():
        print(site.name)

        # Relevant loads ---------------------------------------
        sofar_api.check_token_iteration(next_token=wb.buoys_metadata_token_sorted.loc[site.name, 'sofar_token'])
    
        spot_id = sofar_api.get_spot_id(site_id=site.name, buoys_metadata=wb.buoys_metadata_token_sorted)
        spotter_obj = sofar_api.select_spotter_obj_from_spotter_grid(spot_id=spot_id,
                                                                spotter_grid=sofar_api.spotter_grid,
                                                                devices=sofar_api.devices)
        
        latest_available_datetime = sofar_api.get_latest_available_datetime(spotter_obj=spotter_obj)

        nc_files_available = wb.get_available_nc_files(institution=site.region,
                                                             site_id=site.name)
        
        if nc_files_available:
            nc_files_needed = wb.lookup_netcdf_files_needed(institution=site.region,
                                                        site_id=site.name,
                                                        latest_available_datetime=latest_available_datetime,
                                                        window=window,
                                                        window_unit=window_unit)
            
            latest_nc_file_available = wb.get_latest_nc_file_available(institution=site.region,
                                                                site_id=site.name)
            latest_processed_datetime = wb.get_latest_processed_datetime(nc_file_path=latest_nc_file_available)
            print(latest_nc_file_available)
            print(latest_processed_datetime)
            
            availability_check = wb.check_nc_files_needed_available(nc_files_needed=nc_files_needed,
                                                            nc_files_available=nc_files_available)
            if availability_check:
                nc_to_load = nc_files_needed
            else:
                nc_to_load = latest_nc_file_available
            print("loading previous")
            previous_data_df = wb.load_datasets(nc_file_paths=nc_to_load)
            print(previous_data_df.columns)
        else:
            # change naming
            latest_processed_datetime = wb.generate_window_start_datetime(latest_available_datetime=latest_available_datetime,
                                                                           window=window,
                                                                           window_unit=window_unit)

        # Extraction ---------------------------------------
        new_data_raw = spotter_obj.grab_data(start_date=latest_processed_datetime,
                                        end_date=latest_available_datetime,
                                        include_track=False,
                                        include_barometer_data=True,
                                        include_waves=True,
                                        include_wind=True,
                                        include_surface_temp_data=True,
                                        include_frequency_data=False)
        
        # Processing ---------------------------------------
        waves_new_data_df = wb.convert_to_dataframe(raw_data=new_data_raw,
                                                    parameters_type="waves")
        waves_new_data_df = wb.convert_to_datetime(data=waves_new_data_df)
        
        
        sst_new_data_df = wb.convert_to_dataframe(raw_data=new_data_raw,
                                                parameters_type="surfaceTemp")
        sst_new_data_df = wb.convert_to_datetime(data=sst_new_data_df)

        all_new_data_df = wb.merge_parameter_types(waves=waves_new_data_df,
                                                    sst=sst_new_data_df)

        all_new_data_df = wb.conform_columns_names_aodn(data=all_new_data_df)
        all_new_data_df = wb.drop_unwanted_columns(data=all_new_data_df)
        all_new_data_df = wb.sort_datetimes(data=all_new_data_df)
        # TEMPORARY SETUP
        all_new_data_df["check"] = "new"

        if 'previous_data' in globals():
            all_data_df = wb.concat_previous_new(previous_data=previous_data_df,
                                                new_data=all_new_data_df)
            
        # TEMPORARY SETUP (REMOVE WHEN DONE)
        all_data_df.to_csv("tests/all_data_df_output.csv", index=False)



            



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



