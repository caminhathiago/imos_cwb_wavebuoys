import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
import pandas as pd

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.netcdf.writer import ncWriter, ncAttrsComposer, ncAttrsExtractor, ncProcessor, ncMetaDataLoader
from wavebuoy_nrt.utils import args_processing, IMOSLogging, generalTesting


load_dotenv()


if __name__ == "__main__":

    # Args handling
    vargs = args_processing()

    # Start general logging
    
    general_log_file = os.path.join(vargs.output_path, "logs", f"general_process.log") # f"{runtime}_general_process.log"
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name="general_logger",
                                                logging_filepath=general_log_file)

    wb = WaveBuoy(buoy_type="sofar")
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata)    
    imos_logging = IMOSLogging() 

    # ### TEMPORARY SETUP TO AVOID UNECESSARY SOFAR API CALLS (REMOVE WHEN DONE)
    # "MtEliza", "Hillarys", "Central"
    wb.buoys_metadata = wb.buoys_metadata.loc[["Central"]].copy()
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
            
            meta_data_loader = ncMetaDataLoader(buoys_metadata=wb.buoys_metadata)
            deployment_metadata = meta_data_loader.load_latest_deployment_metadata(site_name=site.name)

            latest_available_time = sofar_api.get_latest_available_time(spot_id=site.serial, token=site.sofar_token)
            SITE_LOGGER.info(f"grabed latest_available_time: {latest_available_time}")

            window_start_time = wb.generate_window_start_time(latest_available_datetime=latest_available_time,
                                                                window=int(vargs.window),
                                                                window_unit=vargs.window_unit)
            SITE_LOGGER.info(f"window start generated as {latest_available_time} minus {vargs.window} {vargs.window_unit}: {window_start_time}")

            nc_files_available = wb.get_available_nc_files(site_id=site.name,
                                                           files_path=vargs.incoming_path,
                                                           deployment_metadata=deployment_metadata,
                                                           parameters_type="spectral")
            SITE_LOGGER.info(f"available nc files: {nc_files_available}")

            if nc_files_available:
                nc_files_needed = wb.lookup_netcdf_files_needed(deployment_metadata=deployment_metadata,
                                                            site_id=site.name,
                                                            latest_available_datetime=latest_available_time,
                                                            window=int(vargs.window),
                                                            window_unit=vargs.window_unit,
                                                            data_type="spectral")
                SITE_LOGGER.info(f"nc files needed based on defined window: {nc_files_needed}")

                latest_nc_file_available = wb.get_latest_nc_file_available(deployment_metadata=deployment_metadata,
                                                                           site_id=site.name,
                                                                           files_path=vargs.incoming_path,
                                                                           parameters_type="spectral")
                SITE_LOGGER.info(f"latest_nc_file_available: {latest_nc_file_available}")

                latest_processed_time = wb.get_latest_processed_time(nc_file_path=latest_nc_file_available)
                SITE_LOGGER.info(f"latest processed time: {latest_processed_time}")
                
                if latest_processed_time == latest_available_time:
                    SITE_LOGGER.info("No new data available. Aborting processing for this site")
                    GENERAL_LOGGER.info(f"Processing successful")
                    imos_logging.logging_stop(logger=SITE_LOGGER)
                    continue

                availability_check, nc_file_paths = wb.check_nc_files_needed_available(nc_files_needed=nc_files_needed,
                                                                            nc_files_available=nc_files_available)

                if availability_check:
                    SITE_LOGGER.info("any or all needed nc files are available.")
                    earliest_nc_file_available = wb.get_earliest_nc_file_available(deployment_metadata=deployment_metadata,
                                                                                site_id=site.name,
                                                                                files_path=vargs.incoming_path)
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

                previous_data_df = wb.load_datasets(nc_file_paths=nc_to_load, flag_previous_new=vargs.flag_previous_new)
                window_start_time = latest_processed_time
                SITE_LOGGER.info(f"considering window start time as lastest processed time ({window_start_time}) as previous nc files are being loaded.")
            
            else:
                SITE_LOGGER.info("no previous nc files available. Extract new data and create new nc files.")



            # Extraction ---------------------------------------
            SITE_LOGGER.info("EXTRACTION STEP ====================================")

            window_end_date = latest_available_time
            new_raw_data = sofar_api.fetch_wave_data(spot_id=site.serial,
                                            token=site.sofar_token,
                                            start_date=window_start_time,
                                            end_date=window_end_date,
                                            include_frequency_data=True,
                                            processing_sources="embedded")

            if not sofar_api.check_new_data(raw_data=new_raw_data, dataset_type="frequencyData"):
                SITE_LOGGER.info("No data for the desired period. Aborting processing for this site")
                GENERAL_LOGGER.info(f"Processing successful")
                imos_logging.logging_stop(logger=SITE_LOGGER)
                continue


            # Processing ---------------------------------------
            SITE_LOGGER.info("PRE-PROCESSING STEP ====================================")

            spectra = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="frequencyData")
            spectra = wb.convert_to_datetime(data=spectra)
            SITE_LOGGER.info(f"waves data converted to DataFrame and pre-processed")

            spectra = wb.create_timeseries_aodn_column(data=spectra)
            spectra = wb.conform_columns_names_aodn(data=spectra, parameters_type="spectral")
            spectra = wb.sort_datetimes(data=spectra)

            if vargs.flag_previous_new:
                spectra["flag_previous_new"] = "new"

            if nc_files_available:
                if not previous_data_df.empty:
                    # TEMPORARY SETUP
                    previous_data_df["processing_source"] = "embedded"
                    # END OF TEMPORARY SETUP (REMOVE WHEN DONE)
                    spectra = wb.concat_previous_new(previous_data=previous_data_df,
                                                    new_data=spectra)
                    SITE_LOGGER.info("concatenate new data with previous since available")
            
            # TEMPORARY SETUP (REMOVE WHEN DONE)
            csv_file_path = os.path.join(vargs.output_path, "test_files", f"{site.name.lower()}_all_data_df_output.csv")
            spectra.reset_index().to_csv(csv_file_path, index=False)
            SITE_LOGGER.info(f"processed data saved as '{csv_file_path}'")

            

            # Qualification ---------------------------------------
            # GENERAL_LOGGER.info("Starting qualification step")
            SITE_LOGGER.info("QUALIFICATION STEP ====================================")


            # Processing Nc File --------------------------------------------
            SITE_LOGGER.info("NC FILE PROCESSING STEP ====================================")

            nc_writer = ncWriter(buoy_type="sofar")
            nc_attrs_composer = ncAttrsComposer(buoys_metadata=wb.buoys_metadata,
                                                deployment_metadata=deployment_metadata,
                                                parameters_type="spectral")
    
            ds_embedded = ncProcessor.compose_dataset(data=spectra, parameters_type="spectral")
            SITE_LOGGER.info("embedded dataset composed")

            ds_embedded = nc_attrs_composer.assign_general_attributes(dataset=ds_embedded, site_name=site.name)
            SITE_LOGGER.info("general attributes assigned to embedded dataset")
            
            ds_embedded = ncProcessor.create_timeseries_variable(dataset=ds_embedded)
            SITE_LOGGER.info("time series variable created in embedded dataset")

            periods_embedded = ncProcessor.extract_monthly_periods_dataset(dataset=ds_embedded)
            ds_objects_embedded = ncProcessor.split_dataset_monthly(dataset=ds_embedded, periods=periods_embedded)
            SITE_LOGGER.info(f"combined dataset split monthly for periods {periods_embedded}")

            ds_objects_embedded = ncProcessor.process_time_to_CF_convention(dataset_objects=ds_objects_embedded)
            SITE_LOGGER.info("dataset objects time dimension processed to conform to CF conventions")

            ds_objects_embedded = nc_attrs_composer.assign_variables_attributes_dataset_objects(dataset_objects=ds_objects_embedded)
            SITE_LOGGER.info("variables attributes assigned to datasets")

            # # ds_objects_embedded = ncProcessor.convert_dtypes(dataset_objects=ds_objects_embedded)
            # # SITE_LOGGER.info("variables dtypes converted and now conforming to template")

            nc_file_names_embedded = nc_writer.compose_file_names(
                                        site_id=site.name.upper(),
                                        periods=periods_embedded,
                                        deployment_metadata=deployment_metadata,
                                        parameters_type="spectral")


            nc_writer.save_nc_file(output_path=vargs.incoming_path,
                                   file_names=nc_file_names_embedded,
                                   dataset_objects=ds_objects_embedded,
                                   parameters_type="spectral")

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
                imos_logging.rename_log_file_if_error(site_name=site.name, file_path=site_logger_file_path,
                                                      add_runtime=False)
                
            continue