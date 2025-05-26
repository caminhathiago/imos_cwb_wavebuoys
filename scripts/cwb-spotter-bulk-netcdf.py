#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
import pandas as pd

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.netcdf.writer import ncWriter, ncAttrsComposer, ncAttrsExtractor, ncProcessor, ncMetaDataLoader
from wavebuoy_nrt.netcdf.validation import ncValidator
from wavebuoy_nrt.utils import args_processing, IMOSLogging, generalTesting, csvOutput
from wavebuoy_nrt.alerts.email import Email


load_dotenv()


def main():

    # Args handling
    vargs = args_processing()

    # Start general logging
    general_log_file = os.path.join(vargs.incoming_path, "logs", f"general_{os.path.basename(__file__).removesuffix(".py")}.log") # f"{runtime}_general_process.log"
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name="general_logger",
                                                logging_filepath=general_log_file)

    wb = WaveBuoy(buoy_type="sofar")
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata)    
    imos_logging = IMOSLogging() 
    
    if vargs.site_to_process:
        wb.buoys_metadata = wb.buoys_metadata.loc[wb.buoys_metadata.index.isin(vargs.site_to_process)].copy()
    
    sites_error_logs = []

    for idx, site in wb.buoys_metadata.iterrows():
        
        GENERAL_LOGGER.info(f"=========== {site.name.upper()} processing ===========")

        site_log_file = os.path.join(vargs.incoming_path,
                                     "sites",
                                     site.name.replace("_",""), 
                                     "logs", 
                                     f"{site.name.upper()}_{os.path.basename(__file__).removesuffix(".py")}.log") # f"{runtime}_[CURRENT_SITE]_process.log
        
        SITE_LOGGER = IMOSLogging().logging_start(logger_name="site_logger", logging_filepath=site_log_file)
        
        SITE_LOGGER.info(f"{site.name.upper()} processing start")

        try:       
            # Relevant loads ---------------------------------------
            SITE_LOGGER.info("LOADING STEP ====================================")
            
            meta_data_loader = ncMetaDataLoader(buoys_metadata=wb.buoys_metadata)
            deployment_metadata = meta_data_loader.load_latest_deployment_metadata(site_name=site.name)
            regional_metadata = meta_data_loader.load_regional_metadata()


            latest_available_time = sofar_api.get_latest_available_time(spot_id=site.serial, token=site.sofar_token)
            SITE_LOGGER.info(f"grabed latest_available_time: {latest_available_time}")

            window_start_time = wb.generate_window_start_time(latest_available_datetime=latest_available_time,
                                                            window=int(vargs.window),
                                                            window_unit=vargs.window_unit)
            SITE_LOGGER.info(f"window start generated as {latest_available_time} minus {vargs.window} {vargs.window_unit}: {window_start_time}")

            nc_files_available = wb.get_available_nc_files(site_id=site.name,
                                                           files_path=vargs.incoming_path,
                                                           deployment_metadata=deployment_metadata,
                                                           parameters_type="bulk")
            SITE_LOGGER.info(f"available nc files: {nc_files_available}")

            if nc_files_available:
                nc_files_needed = wb.lookup_netcdf_files_needed(deployment_metadata=deployment_metadata,
                                                                regional_metadata=regional_metadata,
                                                            site_id=site.name,
                                                            latest_available_datetime=latest_available_time,
                                                            window=int(vargs.window),
                                                            window_unit=vargs.window_unit,
                                                            incoming_path=vargs.incoming_path,
                                                            data_type="bulk")
                SITE_LOGGER.info(f"nc files needed based on defined window: {nc_files_needed}")


                latest_nc_file_available = wb.get_latest_nc_file_available(deployment_metadata=deployment_metadata,
                                                                           site_id=site.name,
                                                                           files_path=vargs.incoming_path,
                                                                           parameters_type="bulk")
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
                    earliest_available_time = wb.get_earliest_processed_time(nc_file_path=earliest_nc_file_available)
                    
                    if (window_start_time < earliest_available_time) and len(nc_file_paths) > 1:
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

                
                previous_data_df = wb.load_datasets(nc_file_paths=nc_to_load,
                                                    flag_previous_new=vargs.flag_previous_new,
                                                    parameters_type="bulk")
                if not previous_data_df.empty:
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
                                            processing_sources="embedded")

            # print("======================")
            if not sofar_api.check_new_data(raw_data=new_raw_data, dataset_type="waves"):
                SITE_LOGGER.info("No data for the desired period. Aborting processing for this site")
                GENERAL_LOGGER.info(f"Processing successful")
                imos_logging.logging_stop(logger=SITE_LOGGER)
                continue

            # Processing ---------------------------------------
            SITE_LOGGER.info("PRE-PROCESSING STEP ====================================")

            waves = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="waves")
            waves = wb.convert_to_datetime(data=waves)         
            
            all_new_data_df = waves.copy()
            all_new_data_df = wb.create_timeseries_aodn_column(data=all_new_data_df)
            all_new_data_df = wb.conform_columns_names_aodn(data=all_new_data_df)
            all_new_data_df = wb.sort_datetimes(data=all_new_data_df)

            if vargs.flag_previous_new:
                all_new_data_df["flag_previous_new"] = "new"

            if nc_files_available:
                if not previous_data_df.empty:
                    if vargs.flag_previous_new:
                        previous_data_df["processing_source"] = "prev"
                    all_data_df = wb.concat_previous_new(previous_data=previous_data_df,
                                                    new_data=all_new_data_df)
                    SITE_LOGGER.info("concatenate new data with previous since available")
                else:
                    all_data_df = all_new_data_df
            else:
                all_data_df = all_new_data_df
            
            csvOutput.save_csv(data=all_data_df, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves.csv")
            
            # Qualification ---------------------------------------
            # GENERAL_LOGGER.info("Starting qualification step")
            SITE_LOGGER.info("QUALIFICATION STEP ====================================")

            qc = WaveBuoyQC(config_id=1)

            all_data_df = qc.create_global_qc_columns(data=all_data_df)
            
            qc.load_data(data=all_data_df)
            parameters_to_qc = qc.get_parameters_to_qc(data=all_data_df, qc_config=qc.qc_config)      
            qualified_data_embedded = qc.qualify(data=all_data_df,
                                        parameters=parameters_to_qc,
                                        gross_range_test=True,
                                        rate_of_change_test=True)
            SITE_LOGGER.info("Qualification successfull")
            
            csvOutput.save_csv(data=all_data_df, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves_qc_subflags.csv")
            csvOutput.save_csv(data=qualified_data_embedded, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves_qc.csv")
            
            # Processing Nc File --------------------------------------------
            SITE_LOGGER.info("NC FILE PROCESSING STEP ====================================")

            nc_writer = ncWriter(buoy_type="sofar")
            nc_attrs_composer = ncAttrsComposer(buoys_metadata=wb.buoys_metadata,
                                                deployment_metadata=deployment_metadata,
                                                regional_metadata=regional_metadata)

            # embedded dataset
            ds_embedded = ncProcessor.compose_dataset(data=qualified_data_embedded)
            SITE_LOGGER.info("embedded dataset composed")
            
            ds_embedded = ncProcessor.convert_dtypes(dataset=ds_embedded, parameters_type="bulk")
            SITE_LOGGER.info("variables dtypes converted and now conforming to template")

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
            
            nc_file_names_embedded = nc_writer.compose_file_names(
                                        site_id=site.name.upper(),
                                        periods=periods_embedded,
                                        deployment_metadata=deployment_metadata,
                                        regional_metadata=regional_metadata,
                                        parameters_type="bulk")
           
           # Validation before saving
           
            reports_path = ncValidator().generate_reports_path(site.name, vargs.incoming_path)
            for nc_file_name, ds_object in zip(nc_file_names_embedded, ds_objects_embedded):
                validation_log = ncValidator().create_validation_log_contents(nc_file_name)
                ncValidator().validate_regional_metadata(nc_file_name, ds_object, regional_metadata, validation_log)
                ncValidator().validate_spot_id(site.name, ds_object, deployment_metadata, wb.buoys_metadata, validation_log)
                # ncValidator().validate_site_name(site.name, ds_object, deployment_metadata, validation_log)
                ncValidator().save_validation_log(reports_path, nc_file_name, validation_log)

            if ncValidator.check_any_fail(validation_log):
                # Saving netcdf file
                nc_writer.save_nc_file(site_id=site.name,
                                        output_path=vargs.incoming_path,
                                    file_names=nc_file_names_embedded,
                                    dataset_objects=ds_objects_embedded)
                SITE_LOGGER.info(f"embedded nc files saved to the output path as {nc_file_names_embedded}")
            else:
                raise ValueError(f"{validation_log["contents"]}")    


            GENERAL_LOGGER.info(f"Processing successful")
            imos_logging.logging_stop(logger=SITE_LOGGER)

        except Exception as e:
            error_message = IMOSLogging().unexpected_error_message.format(site_name=site.name.upper())
            GENERAL_LOGGER.error(str(e), exc_info=True)
            SITE_LOGGER.error(str(e), exc_info=True)
        
            # Closing current site logging
            site_logger_file_path = imos_logging.get_log_file_path(SITE_LOGGER)
            imos_logging.logging_stop(logger=SITE_LOGGER)
            error_logger_file_path = imos_logging.rename_log_file_if_error(site_name=site.name,
                                                                           file_path=site_logger_file_path,
                                                                            script_name=os.path.basename(__file__).removesuffix(".py"),
                                                                            add_runtime=False)
            sites_error_logs.append(error_logger_file_path)
            
            continue

    if sites_error_logs:
        if vargs.email_alert:
            e = Email(script_name=os.path.basename(__file__),
                    email=os.getenv("EMAIL_TO"),
                    log_file_path=sites_error_logs)
            e.send()

        GENERAL_LOGGER.info(f"=========== {site.name.upper()} successfully processed. ===========")

if __name__ == "__main__":
    main()

