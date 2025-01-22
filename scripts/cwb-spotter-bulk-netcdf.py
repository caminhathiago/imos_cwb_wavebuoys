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
from wavebuoy_nrt.utils import args, IMOSLogging


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
    vargs = args()

    # Start general logging
    runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
    general_log_file = os.path.join(vargs.output_path, "logs", f"general_process.log") # f"{runtime}_general_process.log"
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name="general_logger",
                                                logging_filepath=general_log_file)

    wb = WaveBuoy(buoy_type="sofar")
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata)    

    # ### TEMPORARY SETUP TO AVOID UNECESSARY SOFAR API CALLS (REMOVE WHEN DONE)
    # import pickle
    # with open("tests\sofar_api_object.pkl", "rb") as pickle_file:
    #     sofar_api = pickle.load(pickle_file)
    # ### END OF TEMPORARY SETUP 
    

    # ### TEMPORARY SETUP TO AVOID UNECESSARY SOFAR API CALLS (REMOVE WHEN DONE)
    site = wb.buoys_metadata.loc["Hillarys"]
    wb.buoys_metadata = wb.buoys_metadata.loc[["Hillarys","Hillarys_HSM"]].copy()
    for idx, site in wb.buoys_metadata.iterrows():
        
        GENERAL_LOGGER.info(f"=========== {site.name.upper()} processing ===========")

        site_log_file = os.path.join(vargs.output_path, "logs", f"[CURRENT_SITE]_process.log") # f"{runtime}_[CURRENT_SITE]_process.log
        SITE_LOGGER = IMOSLogging().logging_start(logger_name="site_logger",
                                                logging_filepath=site_log_file)
        
        try:       
            
            

            # Relevant loads ---------------------------------------
            
            latest_available_datetime = sofar_api.get_latest_available_time(spot_id=site.serial, token=site.sofar_token)
            window_start_time = wb.generate_window_start_time(latest_available_datetime=latest_available_datetime,
                                                                window=int(vargs.window),
                                                                window_unit=vargs.window_unit)

            nc_files_available = wb.get_available_nc_files(institution=site.region,
                                                                site_id=site.name)

            if nc_files_available:
                nc_files_needed = wb.lookup_netcdf_files_needed(institution=site.region,
                                                            site_id=site.name,
                                                            latest_available_datetime=latest_available_datetime,
                                                            window=int(vargs.window),
                                                            window_unit=vargs.window_unit)
                
                latest_nc_file_available = wb.get_latest_nc_file_available(institution=site.region,
                                                                        site_id=site.name)
                latest_processed_time = wb.get_latest_processed_time(nc_file_path=latest_nc_file_available)
                
                availability_check, nc_file_paths = wb.check_nc_files_needed_available(nc_files_needed=nc_files_needed,
                                                                nc_files_available=nc_files_available)
                
                if availability_check: # any or all needed ar available
                    earliest_nc_file_available = wb.get_earliest_nc_file_available(institution=site.region,
                                                                                site_id=site.name)
                    earliest_available_time = wb.get_earliest_processed_time(nc_file_path=latest_nc_file_available)
                    
                    if window_start_time < earliest_available_time:
                        nc_to_load = None
                    else:
                        nc_to_load = nc_file_paths
                
                else: # none is available
                    if vargs.backfill:
                        nc_to_load = latest_nc_file_available
                    else:
                        nc_to_load = None
                
                previous_data_df = wb.load_datasets(nc_file_paths=nc_to_load)
                window_start_time = latest_processed_time

            # Extraction ---------------------------------------

            new_data_raw = sofar_api.get_wave_data(spot_id=site.serial,
                                            token=site.sofar_token,
                                            start_date=window_start_time,
                                            end_date=latest_available_datetime + timedelta(hours=1))
        
            # Processing ---------------------------------------
            
            waves_new_data_df = wb.convert_wave_data_to_dataframe(raw_data=new_data_raw, parameters_type="waves")
            waves_new_data_df = wb.convert_to_datetime(data=waves_new_data_df)
            
            sst_new_data_df = wb.convert_wave_data_to_dataframe(raw_data=new_data_raw, parameters_type="surfaceTemp")
            sst_new_data_df = wb.convert_to_datetime(data=sst_new_data_df)

            if sst_new_data_df is None and site.version in ("smart_mooring", "half_smart_mooring"):
                new_sensor_data_raw = sofar_api.get_sensor_data(spot_id=site.serial,
                                                            token=site.sofar_token,
                                                            start_date=window_start_time,
                                                            end_date=latest_available_datetime + timedelta(hours=1))

                sensor_new_data_df = wb.convert_smart_mooring_to_dataframe(raw_data=new_sensor_data_raw)
                sensor_new_data_df = wb.convert_to_datetime(data=sensor_new_data_df)
                sst_new_data_df = wb.get_sst_from_smart_mooring(data=sensor_new_data_df,
                                                                sensor_type="temperature")
                
            all_new_data_df = wb.merge_parameter_types(waves=waves_new_data_df, sst=sst_new_data_df)

            all_new_data_df = wb.conform_columns_names_aodn(data=all_new_data_df)
            all_new_data_df = wb.drop_unwanted_columns(data=all_new_data_df)
            all_new_data_df = wb.sort_datetimes(data=all_new_data_df)
            
            # TEMPORARY SETUP
            all_new_data_df["check"] = "new"

            if 'previous_data_df' in locals():
                if not previous_data_df.empty: # check if empty
                    all_data_df = wb.concat_previous_new(previous_data=previous_data_df,
                                                    new_data=all_new_data_df)
            else:
                all_data_df = all_new_data_df

            # TEMPORARY SETUP (REMOVE WHEN DONE)
            csv_file_path = os.path.join(vargs.output_path, "test_files", "all_data_df_output.csv")
            all_data_df.to_csv(csv_file_path, index=False)
            
            GENERAL_LOGGER.info(f"=========== {site.name.upper()} successfully processed. ===========")
    
        except Exception as e:
            error_message = IMOSLogging().unexpected_error_message.format(site_name=site.name.upper())
            GENERAL_LOGGER.error(error_message)
            SITE_LOGGER.error(str(e), exc_info=True)
        
        # Closing current site logging
        imos_logging = IMOSLogging()
        site_logger_file_path = imos_logging.get_log_file_path(SITE_LOGGER)
        imos_logging.logging_stop(logger=SITE_LOGGER)
        imos_logging.rename_log_file(logger=SITE_LOGGER, site_name=site.name, file_path=site_logger_file_path)

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



