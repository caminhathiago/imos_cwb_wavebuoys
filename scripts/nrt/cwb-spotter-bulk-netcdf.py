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

            latest_available_time = sofar_api.get_latest_available_time(spot_id=site.serial,
                                                                        token=site.sofar_token,
                                                                        data_type="bulk",
                                                                        processing_sources="embedded")
            if not latest_available_time:
                SITE_LOGGER.warning("Skipping to next site. If trying to fetch for Spectra, this site probably doesn't have Spectra in NRT enabled.")
                continue
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

                
                waves_previous, temp_previous = wb.load_datasets(nc_file_paths=nc_to_load,
                                                    flag_previous_new=vargs.flag_previous_new,
                                                    parameters_type="bulk")
                if not waves_previous.empty:
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

            # Waves
            waves = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="waves")
            waves = wb.convert_to_datetime(data=waves)         
            
            waves = wb.create_timeseries_aodn_column(data=waves)
            waves = wb.conform_columns_names_aodn(data=waves)
            waves = wb.sort_datetimes(data=waves)

            if vargs.flag_previous_new:
                waves["flag_previous_new"] = "new"

            if nc_files_available:
                if not waves_previous.empty:
                    if vargs.flag_previous_new:
                        waves_previous["processing_source"] = "prev"
                    waves = wb.concat_previous_new(previous_data=waves_previous,
                                                    new_data=waves,
                                                    drop_duplicates=True)
                    SITE_LOGGER.info("concatenate new data with previous since available")

            
            csvOutput.save_csv(data=waves, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves.csv")
            
            # Temp
            temp = None
            if new_raw_data["surfaceTemp"]:
                temp = wb.convert_wave_data_to_dataframe(raw_data=new_raw_data, parameters_type="surfaceTemp")
                temp = wb.convert_to_datetime(data=temp, parameter_type="temp")
                # generalTesting().generate_pickle_file(data=temp, file_name="surfaceTemp_new_data", site_name=site.name)
                SITE_LOGGER.info(f"temp data converted to DataFrame and pre-processed if exists")
            
            elif not new_raw_data["surfaceTemp"] and site.version in ("smart_mooring", "half_smart_mooring"):
                SITE_LOGGER.info(f"no sst available from spotter, grab smart mooring data since it is available (i.e. buoy version: {site.version})")
                
                new_sensor_data_raw = sofar_api.get_sensor_data(spot_id=site.serial,
                                                            token=site.sofar_token,
                                                            start_date=window_start_time,
                                                            end_date=window_end_date)
                SITE_LOGGER.info(f"raw smart mooring data extracted from Sofar API")
                
                temp = wb.convert_smart_mooring_to_dataframe(raw_data=new_sensor_data_raw)
                temp = wb.convert_to_datetime(data=temp, parameter_type="temp")
                temp = wb.get_temp_from_smart_mooring(data=temp, sensor_type="temperature")
                temp = wb.process_smart_mooring_columns(data=temp)

                SITE_LOGGER.info("smart mooring data processed")

            if temp is not None:

                temp = wb.create_timeseries_aodn_column(data=temp)
                temp = wb.conform_columns_names_aodn(data=temp)
                temp = wb.sort_datetimes(data=temp)
                temp = wb.drop_lat_lon(data=temp)
                temp = wb.filter_timerange(data=temp, waves=waves)

                if nc_files_available:
                    if not temp_previous.empty:
                        if vargs.flag_previous_new:
                            temp_previous["processing_source"] = "prev"
                        temp = wb.concat_previous_new(previous_data=temp_previous,
                                                        new_data=temp,
                                                        data_type="temp")
                        SITE_LOGGER.info("concatenate new data with previous since available")

                csvOutput.save_csv(data=temp, site_name=site.name.upper(), file_path=vargs.incoming_path, file_name_preffix="_temp_data.csv")

            # Qualification ---------------------------------------
            SITE_LOGGER.info("QUALIFICATION STEP ====================================")

            # Waves
            qc = WaveBuoyQC(config_id=1)

            waves = qc.create_global_qc_columns(data=waves, parameter_type="waves")
            
            qc.load_data(data=waves)
            parameters_to_qc = qc.get_parameters_to_qc(data=waves, qc_config=qc.qc_config)
            waves_qualified, waves_subflags = qc.qualify(data=waves,
                                                 parameter_type="waves",
                                                parameters=parameters_to_qc,
                                                window = int(vargs.window),
                                                gross_range_test=True,
                                                rate_of_change_test=True)
            
            SITE_LOGGER.info("Qualification successfull")
            
            csvOutput.save_csv(data=waves_subflags, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves_qc_subflags.csv")
            csvOutput.save_csv(data=waves_qualified, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_waves_qc.csv")
            
            # Temp
            temp_qualified = None
            if temp is not None:

                temp = qc.create_global_qc_columns(data=temp, parameter_type="temp")
                
                qc = WaveBuoyQC(config_id=1)

                qc.load_data(data=temp)
                parameters_to_qc = qc.get_parameters_to_qc(data=temp, qc_config=qc.qc_config)

                temp_qualified, temp_subflags = qc.qualify(data=temp,
                                                 parameter_type="temp",
                                                parameters=parameters_to_qc,
                                                window = int(vargs.window),
                                                gross_range_test=True,
                                                rate_of_change_test=True)
                
                SITE_LOGGER.info("Qualification successfull")
                
                csvOutput.save_csv(data=temp_subflags, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_temp_qc_subflags.csv")
                csvOutput.save_csv(data=temp_qualified, site_name=site.name, file_path=vargs.incoming_path, file_name_preffix="_temp_qc.csv")


            # Processing Nc File --------------------------------------------
            SITE_LOGGER.info("NC FILE PROCESSING STEP ====================================")

            nc_writer = ncWriter(buoy_type="sofar")
            nc_attrs_composer = ncAttrsComposer(buoys_metadata=wb.buoys_metadata,
                                                deployment_metadata=deployment_metadata,
                                                regional_metadata=regional_metadata)

            # embedded dataset
            ds = ncProcessor.compose_dataset(waves=waves_qualified, temp=temp_qualified)
            SITE_LOGGER.info("embedded dataset composed")
            
            ds = ncProcessor.convert_dtypes(dataset=ds, parameters_type="bulk")
            SITE_LOGGER.info("variables dtypes converted and now conforming to template")

            ### Drifters conversion name
            site_name, drifter = ncProcessor.convert_drifter_name(site_name=site.name)

            ds = nc_attrs_composer.assign_general_attributes(dataset=ds, site_name=site_name, drifter=drifter)
            SITE_LOGGER.info("general attributes assigned to embedded dataset")
            
            ds = ncProcessor.create_timeseries_variable(dataset=ds)
            SITE_LOGGER.info("time series variable created in embedded dataset")

            periods_embedded = ncProcessor.extract_monthly_periods_dataset(dataset=ds)
            ds_objects = ncProcessor.split_dataset_monthly(dataset=ds, periods=periods_embedded)
            SITE_LOGGER.info(f"combined dataset split monthly for periods {periods_embedded}")
            
            ds_objects = ncProcessor.process_time_to_CF_convention(dataset_objects=ds_objects)
            SITE_LOGGER.info("dataset objects time dimension processed to conform to CF conventions")
            
            ds_objects = nc_attrs_composer.assign_variables_attributes_dataset_objects(dataset_objects=ds_objects)
            SITE_LOGGER.info("variables attributes assigned to datasets")
            
            nc_file_names_embedded = nc_writer.compose_file_names(
                                        site_id=site_name,
                                        periods=periods_embedded,
                                        deployment_metadata=deployment_metadata,
                                        regional_metadata=regional_metadata,
                                        parameters_type="bulk")
           
           # Validation before saving
           
            reports_path = ncValidator().generate_reports_path(site.name, vargs.incoming_path)
            for nc_file_name, ds_object in zip(nc_file_names_embedded, ds_objects):
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
                                    dataset_objects=ds_objects)
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

