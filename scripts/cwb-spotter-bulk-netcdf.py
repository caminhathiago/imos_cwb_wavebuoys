#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
import pandas as pd

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.utils import args, IMOSLogging


load_dotenv()


if __name__ == "__main__":

    # Args handling
    vargs = args()

    # Start general logging
    
    general_log_file = os.path.join(vargs.output_path, "logs", f"general_process.log") # f"{runtime}_general_process.log"
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name="general_logger",
                                                logging_filepath=general_log_file)

    wb = WaveBuoy(buoy_type="sofar")
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata)    
    imos_logging = IMOSLogging() 
    
    # ### TEMPORARY SETUP TO AVOID UNECESSARY SOFAR API CALLS (REMOVE WHEN DONE)
    wb.buoys_metadata = wb.buoys_metadata.loc[["Hillarys","Hillarys_HSM"]].copy()
    # END OF TEMPORARY SETUP

    for idx, site in wb.buoys_metadata.iterrows():
        
        GENERAL_LOGGER.info(f"=========== {site.name.upper()} processing ===========")

        site_log_file = os.path.join(vargs.output_path, "logs", f"{site.name.upper()}_run.log") # f"{runtime}_[CURRENT_SITE]_process.log
        SITE_LOGGER = IMOSLogging().logging_start(logger_name="site_logger", logging_filepath=site_log_file)
        
        GENERAL_LOGGER.info(f"{site.name.upper()} log file created as {site_log_file}")
        SITE_LOGGER.info(f"{site.name.upper()} processing start")

        try:       
            # Relevant loads ---------------------------------------
            SITE_LOGGER.info("LOADING STEP ====================================")
            
            latest_available_time = sofar_api.get_latest_available_time(spot_id=site.serial, token=site.sofar_token)
            SITE_LOGGER.info(f"grabed latest_available_time: {latest_available_time}")

            window_start_time = wb.generate_window_start_time(latest_available_datetime=latest_available_time,
                                                                window=int(vargs.window),
                                                                window_unit=vargs.window_unit)
            SITE_LOGGER.info(f"window start generated as {latest_available_time} minus {vargs.window} {vargs.window_unit}: {window_start_time}")

            nc_files_available = wb.get_available_nc_files(institution=site.region, site_id=site.name)
            SITE_LOGGER.info(f"available nc files: {nc_files_available}")

            if nc_files_available:
                nc_files_needed = wb.lookup_netcdf_files_needed(institution=site.region,
                                                            site_id=site.name,
                                                            latest_available_datetime=latest_available_time,
                                                            window=int(vargs.window),
                                                            window_unit=vargs.window_unit)
                SITE_LOGGER.info(f"nc files needed based on defined window: {nc_files_needed}")


                latest_nc_file_available = wb.get_latest_nc_file_available(institution=site.region, site_id=site.name)

                latest_processed_time = wb.get_latest_processed_time(nc_file_path=latest_nc_file_available)
                SITE_LOGGER.info(f"latest processed time: {latest_processed_time}")


                availability_check, nc_file_paths = wb.check_nc_files_needed_available(nc_files_needed=nc_files_needed,
                                                                            nc_files_available=nc_files_available)
                
                if availability_check:
                    SITE_LOGGER.info("any or all needed nc files are available.")
                    earliest_nc_file_available = wb.get_earliest_nc_file_available(institution=site.region,
                                                                                site_id=site.name)
                    earliest_available_time = wb.get_earliest_processed_time(nc_file_path=latest_nc_file_available)
                    
                    if window_start_time < earliest_available_time:
                        SITE_LOGGER.info("desired window start time is older than earliest available time, extract new data and overwrite all available nc files.")
                        nc_to_load = None
                    else:
                        nc_to_load = nc_file_paths
                
                else:
                    SITE_LOGGER.info("no needed nc files are available.")
                    
                    if vargs.backfill:
                        SITE_LOGGER.info(f"backfilling using latest nc file available as backfill argument is set to {vargs.backfill}")
                        
                        nc_to_load = latest_nc_file_available
                    else:
                        SITE_LOGGER.info("no Backfilling as backfill argument is set to {vargs.backfill}. Only creating new nc files with newly extracted data.")
                        
                        nc_to_load = None
            
                previous_data_df = wb.load_datasets(nc_file_paths=nc_to_load)
                window_start_time = latest_processed_time
                SITE_LOGGER.info(f"considering window start time as lastest processed time ({window_start_time}) as previous nc files are being loaded.")
            
            else:
                SITE_LOGGER.info("no previous nc files available. Extract new data and create new nc files.")

            # Extraction ---------------------------------------
            SITE_LOGGER.info("EXTRACTION STEP ====================================")

            window_end_date = latest_available_time + timedelta(hours=1)
            new_raw_data = sofar_api.fetch_wave_data(spot_id=site.serial,
                                            token=site.sofar_token,
                                            start_date=window_start_time,
                                            end_date=window_end_date)
            wb.generate_pickle_file(data=new_raw_data, file_name="new_data_raw", site_name=site.name)
            SITE_LOGGER.info(f"raw spotter data extracted from Sofar API")

            if not new_raw_data["waves"]:
                SITE_LOGGER.info("No data for the desired period. Aborting processing for this site")
                break

            # Processing ---------------------------------------
            SITE_LOGGER.info("PROCESSING STEP ====================================")

            # split_raw_data = wb.split_processing_source(raw_data=new_raw_data)

            waves = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="waves")
            waves = wb.convert_to_datetime(data=waves)
            # change to priority_source to hdr at some point
            # waves_new_data_df = wb.select_processing_source(data=waves_new_data_df, priority_source="embedded")
            # waves = wb.drop_unwanted_columns(data=waves)
            SITE_LOGGER.info(f"waves data converted to DataFrame and pre-processed")

            

            if new_raw_data["surfaceTemp"]:
                sst = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="surfaceTemp")
                sst = wb.convert_to_datetime(data=sst)
                # change to priority_source to hdr at some point
                # sst = wb.select_processing_source(data=sst, priority_source="embedded") 
                # sst = wb.drop_unwanted_columns(data=sst)
                wb.generate_pickle_file(data=sst, file_name="surfaceTemp_new_data", site_name=site.name)

               

                SITE_LOGGER.info(f"sst data converted to DataFrame and pre-processed if exists")


            if not new_raw_data["surfaceTemp"] and site.version in ("smart_mooring", "half_smart_mooring"):
                SITE_LOGGER.info(f"no sst available from spotter, grab smart mooring data since it is available (i.e. buoy version: {site.version})")
                
                new_sensor_data_raw = sofar_api.get_sensor_data(spot_id=site.serial,
                                                            token=site.sofar_token,
                                                            start_date=window_start_time,
                                                            end_date=window_end_date)
                
                wb.generate_pickle_file(data=new_sensor_data_raw, file_name="smart_mooring_raw", site_name=site.name)

                SITE_LOGGER.info(f"raw smart mooring data extracted from Sofar API")
                
                sst_sm = wb.convert_smart_mooring_to_dataframe(raw_data=new_sensor_data_raw)
                sst_sm = wb.convert_to_datetime(data=sst_sm)
                sst_sm = wb.get_sst_from_smart_mooring(data=sst_sm, sensor_type="temperature")
                sst_sm = wb.process_smart_mooring_columns(data=sst_sm)
                sst = wb.round_parameter_values(data=sst_sm, parameter="SST")
                
                # TEMPORARY SETUP
                sst["processing_source"] = "embedded"
                sst2 = sst.copy()
                sst2["processing_source"] = "hdr"
                sst = pd.concat([sst, sst2], axis=0)
                # END OF TEMPORARY SETUP (REMOVE WHEN DONE)

                SITE_LOGGER.info("smart mooring data processed")

            all_new_data_df = wb.merge_parameter_types(waves=waves,
                                                       sst=sst,
                                                       consider_processing_source=True)


            wb.generate_pickle_file(data=sst, file_name="sst_new_data_df", site_name=site.name)
            wb.generate_pickle_file(data=waves, file_name="waves_new_data_df", site_name=site.name)
            wb.generate_pickle_file(data=all_new_data_df, file_name="all_new_data_df", site_name=site.name)

            SITE_LOGGER.info("waves and sst/upper smart mooring temperature sensor merged")

            # TEMPORARY SETUP
            test = wb.test_duplicated(data=all_new_data_df)
            SITE_LOGGER.warning(f"data has duplicated values? R: {test}")
            # END OF TEMPORARY SETUP (REMOVE WHEN DONE)

            all_new_data_df = wb.create_timeseries_aodn_column(data=all_new_data_df)
            all_new_data_df = wb.conform_columns_names_aodn(data=all_new_data_df)
            all_new_data_df = wb.sort_datetimes(data=all_new_data_df)

            # TEMPORARY SETUP
            all_new_data_df["check"] = "new"
            # END OF TEMPORARY SETUP (REMOVE WHEN DONE)
            print(nc_files_available)
            if nc_files_available:
                if not previous_data_df.empty:
                    # TEMPORARY SETUP
                    previous_data_df["processing_source"] = "embedded"
                    # END OF TEMPORARY SETUP (REMOVE WHEN DONE)
                    all_data_df = wb.concat_previous_new(previous_data=previous_data_df,
                                                    new_data=all_new_data_df)
                    SITE_LOGGER.info("concatenate new data with previous since available")
                else:
                    all_data_df = all_new_data_df
            else:
                all_data_df = all_new_data_df


            
            # TEMPORARY SETUP (REMOVE WHEN DONE)
            csv_file_path = os.path.join(vargs.output_path, "test_files", f"{site.name.lower()}_all_data_df_output.csv")
            all_data_df.reset_index().to_csv(csv_file_path, index=False)
            SITE_LOGGER.info(f"processed data saved as '{csv_file_path}'")
            
            # END OF TEMPORARY SETUP (REMOVE WHEN DONE)
            
            # Qualification ---------------------------------------
            # GENERAL_LOGGER.info("Starting qualification step")
            SITE_LOGGER.info("QUALIFICATION STEP ====================================")

            all_data_hdr = wb.select_processing_source(data=all_data_df, processing_source="hdr")
            all_data_embedded = wb.select_processing_source(data=all_data_df, processing_source="embedded")


            qc = WaveBuoyQC(config_id=1)
            qc.load_data(data=all_data_embedded)
            parameters_to_qc = qc.get_parameters_to_qc(data=all_data_embedded, qc_config=qc.qc_config)      
            qualified_data_embedded = qc.qualify(data=all_data_embedded,
                                        parameters=parameters_to_qc,
                                        gross_range_test=True,
                                        rate_of_change_test=True)

            if not all_data_hdr.empty:
                qc = WaveBuoyQC(config_id=1)
                qc.load_data(data=all_data_hdr)
                parameters_to_qc = qc.get_parameters_to_qc(data=all_data_hdr, qc_config=qc.qc_config)      
                qualified_data_hdr = qc.qualify(data=all_data_hdr,
                                            parameters=parameters_to_qc,
                                            gross_range_test=True,
                                            rate_of_change_test=True)

            SITE_LOGGER.info("Qualification successfull")

            # qualified_data_summarized = qc.summarize_flags(data=qualified_data, parameter_type="waves")

            # TEMPORARY SETUP (REMOVE WHEN DONE)
            csv_file_path_embedded = os.path.join(vargs.output_path, "test_files", f"{site.name.lower()}_qualified_embedded.csv")
            qualified_data_embedded.to_csv(csv_file_path_embedded, index=False)
            SITE_LOGGER.info(f"qualified data saved as '{csv_file_path_embedded}'")

            if not all_data_hdr.empty:
                csv_file_path_hdr = os.path.join(vargs.output_path, "test_files", f"{site.name.lower()}_qualified_hdr.csv")
                qualified_data_hdr.to_csv(csv_file_path_hdr, index=False)
                SITE_LOGGER.info(f"qualified data saved as '{csv_file_path_hdr}'")
            # END OF TEMPORARY SETUP (REMOVE WHEN DONE)

            GENERAL_LOGGER.info("Starting qualification step")
            # Processing Nc File --------------------------------------------
            SITE_LOGGER.info("NC FILE PROCESSING STEP ====================================")

            # all_data_df_qualified = wb.
            # dataset = wb.convert_dataframe_to_dataset(data=all_data_df_qualified)

            # Loading -------------------

            # dataset = all_data_df_qualified.to_xarray()
            # nc_file_path = os.path.join(vargs.output_path, "test_files", f"{site.name.lower()}_qualified.nc")
            # dataset.to_netcdf(nc_file_path, engine="netcdf4")
            # SITE_LOGGER.info(f"raw nc file (no attributes) saved as '{nc_file_path}'")

            GENERAL_LOGGER.info(f"Processing successful")
            imos_logging.logging_stop(logger=SITE_LOGGER)

        except Exception as e:
            error_message = IMOSLogging().unexpected_error_message.format(site_name=site.name.upper())
            GENERAL_LOGGER.error(error_message)
            SITE_LOGGER.error(str(e), exc_info=True)
        
            # Closing current site logging
            site_logger_file_path = imos_logging.get_log_file_path(SITE_LOGGER)
            imos_logging.logging_stop(logger=SITE_LOGGER)
            if e:
                imos_logging.rename_log_file_if_error(site_name=site.name, file_path=site_logger_file_path)
            
            continue

        GENERAL_LOGGER.info(f"=========== {site.name.upper()} successfully processed. ===========")





