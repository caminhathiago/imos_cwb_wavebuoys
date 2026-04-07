# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
import pandas as pd

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.processor.auswaves import ProcAusWaves as pa
# from wavebuoy_nrt.netcdf.writer import ncWriter, ncAttrsComposer, ncAttrsExtractor, ncProcessor, ncMetaDataLoader
# from wavebuoy_nrt.netcdf.validation import ncValidator
from wavebuoy_nrt.utils import args_auswaves_processing, IMOSLogging
from wavebuoy_nrt.alerts.email import Email

load_dotenv()

def generate_site_logger(vargs, site):
     
    site_log_file = os.path.join(vargs.incoming_path,
                                    "sites",
                                    site.name.replace("_",""), 
                                    "logs", 
                                    f"{site.name.upper()}_{os.path.basename(__file__).removesuffix(".py")}.log") # f"{runtime}_[CURRENT_SITE]_process.log
    
    return IMOSLogging().logging_start(logger_name="site_logger", logging_filepath=site_log_file)

def extract(vargs, wb, site):
    
    SITE_LOGGER.info("EXTRACTION STEP ====================================")

    latest_available_time_emb = sofar_api.get_latest_available_time(spot_id=site.serial,
                                                                    token=site.sofar_token,
                                                                    data_type="bulk",
                                                                    processing_sources="embedded")
    
    latest_available_time_hdr = sofar_api.get_latest_available_time(spot_id=site.serial,
                                                                    token=site.sofar_token,
                                                                    data_type="bulk",
                                                                    processing_sources="hdr")

    latest_times = [latest_available_time_emb, latest_available_time_hdr]
    latest_times = [v for v in latest_times if v is not None]

    latest_available_time = max(latest_times) if latest_times else None

    if latest_available_time is None:
        SITE_LOGGER.info(f"No latest time could be retrieved.")
        return

    SITE_LOGGER.info(f"grabed latest_available_time: {latest_available_time}")

    window_start_time = wb.generate_window_start_time(latest_available_datetime=latest_available_time,
                                                        window=int(vargs.window),
                                                        window_unit=vargs.window_unit)
    
    SITE_LOGGER.info(f"window start generated as {latest_available_time} minus {vargs.window} {vargs.window_unit}: {window_start_time}")
    
    window_end_date = datetime.now() + timedelta(hours=1)

    new_raw_data_embedded = sofar_api.fetch_wave_data(spot_id=site.serial,
                                        token=site.sofar_token,
                                        start_date=window_start_time,
                                        end_date=window_end_date,
                                        processing_sources="embedded")

    new_raw_data_hdr = sofar_api.fetch_wave_data(spot_id=site.serial,
                                        token=site.sofar_token,
                                        start_date=window_start_time,
                                        end_date=window_end_date,
                                        processing_sources="hdr")

    
    
    # if site.version in ("smart_mooring", "half_smart_mooring"):
    new_sensor_data_raw = sofar_api.get_sensor_data(spot_id=site.serial,
                                                    token=site.sofar_token,
                                                    start_date=window_start_time,
                                                    end_date=window_end_date)

    if not sofar_api.check_new_data(raw_data=new_raw_data_embedded, dataset_type="waves"):
        return

    previous_data = extract_previous(window_start_time, window_end_date, site)

    return {
        "previous_data":previous_data,
        "new_raw_data_embedded": new_raw_data_embedded,
        "new_raw_data_hdr": new_raw_data_hdr,
        "new_sensor_data_raw": new_sensor_data_raw,
        "latest_available_time": latest_available_time,
        "window_start_time": window_start_time
    }

def extract_previous(window_start_time, window_end_date, site):

    from wavebuoy_nrt.aws.aws import CWBAWSS3

    cwb_s3 = CWBAWSS3(
        aws_access_key_id=os.getenv('AUSWAVES_AWS_S3_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AUSWAVES_AWS_S3_ACCESS_KEY_SECRET'),
        region_name=os.getenv('AUSWAVES_AWS_S3_REGION'),
        bucket=os.getenv('AUSWAVES_AWS_S3_BUCKET'),                        
        prefix=os.getenv('AUSWAVES_AWS_S3_PREFIX'),                        
    )

    needed_csvs = cwb_s3.generate_needed_files_s3keys(site, window_start_time.date(), window_end_date.date())

    # needed_csvs = ['auswaves/vicwaves/Bob/text_archive/2026/04/Bob_20260410.csv', 'auswaves/vicwaves/Bob/text_archive/2026/04/Bob_20260411.csv']

    missing_data_errors, previous_data = cwb_s3.get_csvs(needed_csvs)

    if missing_data_errors:
        for error in missing_data_errors:
            SITE_LOGGER.warning(f"No csvs found for: {error["s3Key"]}. Error raised: {error["error"]}")

    return previous_data

def transform(new):

    SITE_LOGGER.info("PRE-PROCESSING STEP ====================================")

    parameter_types = ['waves', 'partitionData', 'wind','surfaceTemp', 'barometerData']
    
    # Process new data -----------------------------------
    data = []
    for parameter_type in parameter_types:

        if parameter_type not in new['new_raw_data_embedded']:
            continue

        param_data = wb.convert_wave_data_to_dataframe(new['new_raw_data_embedded'], parameter_type)
        if param_data is None:
            SITE_LOGGER.info(f"No data for {parameter_type}, skipping processing")
            continue

        param_data = pa.convert_to_datetime(param_data)         
        param_data = pa.sort_datetimes(param_data)
        
        if parameter_type == 'partitionData':
            param_data = pa.process_partition_data(param_data)

        if parameter_type != "waves":
            param_data = pa.drop_redundant_columns(param_data)

        data.append(param_data)
    
    data = pa.merge_on_time(data, method='merge')

    if new['new_sensor_data_raw']:
        
        sm_data = pa.convert_sm_to_dataframe(new['new_sensor_data_raw'])
        sm_data = pa.convert_to_datetime(sm_data)
        sm_data = pa.pivot_sm_data(sm_data)
        sm_data = pa.conform_sm_cols_to_auswaves(sm_data)

        data = pa.combine_spotter_sm(data, sm_data, method="interpolation")

    
    data['site'] = site.name
    data['spot_id'] = site.serial
    
    data = pa.process_time_unix_column(data, method='create')

    data = pa.conform_cols_to_auswaves(data)
    data = pa.create_missing_columns(data)
    data = pa.drop_unwanted_columns(data)
    
    data = pa.fill_nan_9999(data)
    
    data = pa.reorder_cols(data)

    # Process previous data ----------------------

    previous_data = new['previous_data']

    if previous_data is None:
        return data

    previous_data = pa.strip_columns(previous_data)

    previous_data = pa.convert_to_datetime(previous_data)

    data = pa.concat_previous_new(previous_data, data)

    data = pa.drop_timestamp_duplicates(data)

    return data

def qualify(data) -> pd.DataFrame:
    
    # Pre-QC process --------------------
    data = pa.process_time_unix_column(data, method='drop')

    data = pa.replace_9999_nan(data)

    data = pa.split_parameters_types(data)

    # Qualification --------------------
    from wavebuoy_nrt.qc.qcTests import WaveBuoyQC

    config_id = 1
    wbqc = WaveBuoyQC(config_id=config_id)
     
    for param_type, metadata in data.items():
        
        if not metadata['to_qc']:
            continue

        metadata['data'] = pa.conform_AODN_cols(metadata['data'])
        
        parameters_to_qc = wbqc.get_parameters_to_qc(data=metadata['data'], qc_config=wbqc.qc_config)
        metadata['data_qced'], metadata['data'] = wbqc.qualify(data=metadata['data'],
                                    parameter_type=param_type,
                                    parameters=parameters_to_qc,
                                    window = int(vargs.window),
                                    gross_range_test=True,
                                    rate_of_change_test=True,
                                    missing_values_test=True)

    # Post-QC process ------------------------
    data = pa.combine_parameters_types(data)

    data = pa.process_time_unix_column(data, method="create")
 
    data = pa.fill_nan_9999(data)
    
    data = pa.conform_AODN_cols(data, direction='backwards')
    data = pa.conform_cols_to_auswaves(data)
    data = pa.reorder_cols(data)

    return data

def load(data):

    # Split dataframes to daily csvs ---------------------------
    
    data = pa.conform_cols_to_auswaves(data, strip=False)

    data_daily = pa.split_data_daily(data)

    data_daily = pa.conform_timestamp_format_auswaves(data_daily)

    from wavebuoy_nrt.aws.aws import CWBAWSS3

    cwb_s3 = CWBAWSS3(
        aws_access_key_id=os.getenv('AUSWAVES_AWS_S3_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AUSWAVES_AWS_S3_ACCESS_KEY_SECRET'),
        region_name=os.getenv('AUSWAVES_AWS_S3_REGION'),
        bucket=os.getenv('AUSWAVES_AWS_S3_BUCKET'),                        
        prefix=os.getenv('AUSWAVES_AWS_S3_PREFIX'),                        
    )

    data_daily = cwb_s3.generate_daily_csv_keys(data_daily, site)

    try:
        for day in data_daily:
            cwb_s3.put_daily_csvs(day)
            SITE_LOGGER.info(f"successfully put file: {day['s3Key']}")
    
    except Exception as e:
        raise e



if __name__ == "__main__":

    # Args handling
    vargs = args_auswaves_processing()

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
        SITE_LOGGER = generate_site_logger(vargs, site)
        SITE_LOGGER.info(f"{site.name.upper()} processing start")

        try:

            data = extract(vargs, wb, site)
            
            if data is None:
                SITE_LOGGER.warning("No data for the desired period. Aborting processing for this site")
                GENERAL_LOGGER.info(f"Processing successful")
                imos_logging.logging_stop(logger=SITE_LOGGER)
                continue
            
            data = transform(data)
            
            if data.empty:
                SITE_LOGGER.warning("No new data transmitted since last pipeline execution. Aborting processing for this site")
                imos_logging.logging_stop(logger=SITE_LOGGER)
                continue

            if vargs.enable_qualification:
                data = qualify(data)

            load(data)


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

    if sites_error_logs:
        if vargs.email_alert:
            e = Email(script_name=os.path.basename(__file__),
                    email=os.getenv("EMAIL_TO"),
                    log_file_path=sites_error_logs)
            e.send()