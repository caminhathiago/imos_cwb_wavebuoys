import os
import json
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset
import pandas as pd
from dotenv import load_dotenv
import glob

from wavebuoy_nrt.ftp.ftp import ncPusher
from wavebuoy_nrt.utils import args_pushing, IMOSLogging
# from wavebuoy_nrt.netcdf.validation import ncValidator
# from wavebuoy_nrt.alerts.email import Email
# from wavebuoy_nrt.wavebuoy import WaveBuoy

load_dotenv()

def check_time_monotoniciy(dataset, logger) -> None:

    import numpy as np

    errors = []

    time_variables = [var for var in list(dataset.variables) if "TIME" in var]

    test_vars_map = {
    "ZDIS": ("LONGITUDE", "LATITUDE"),
    "WSSH": ("WSSH", "WPPE"),
    "ENERGY": ("ENERGY", "A1"),
    }

    # Find which key is present in the dataset
    test_vars = next(
        vars_ for key, vars_ in test_vars_map.items()
        if key in dataset.variables
    )

    if not test_vars:
        raise KeyError(f"None of {tuple(test_vars_map)} found in dataset variables")

    for time_var in time_variables:
        
        time_data = dataset[time_var][:].data
        
        test_data = []
        for test_var in test_vars:
            test_data.append((test_var, dataset[test_var][:].data))

        diff = np.diff(time_data)
        if np.all(diff > 0):
            logger.info(f"{time_var} monotonic increasing (whole) - PASS")
        
        else:
            logger.info(f"{time_var} monotonic increasing (whole) - FAIL")
            
            for i in range(len(time_data)):
                
                if time_data[i] - time_data[i-1] == 0:
                    
                    info_text = f"{i-1} - {time_data[i-1]}"
                    for data in test_data: 
                        info_text += f" - {data[0]}: {data[1][i-1]}" 
                    
                    logger.info(info_text)

                    info_text = f"{i} - {time_data[i]}"
                    for data in test_data: 
                        info_text += f" - {data[0]}: {data[1][i]}" 
                    
                    logger.info(info_text)
                    # logger.info(f"{i-1} - {time_data[i-1]} - {test_vars[0]}: {test_data1[i-1]} {test_vars[1]}: {test_data2[i-1]}")
                    # logger.info(f"{i} - {time_data[i]} - {test_vars[0]}: {test_data1[i]} {test_vars[1]}: {test_data2[i]}")
                    logger.info("---")

            errors.append(time_var)

    if errors:    
        raise ValueError(f"{errors} variables are not monotonically increasing")

def main():

    vargs = args_pushing()
    
    imos_logging = IMOSLogging() 
    log_file = os.path.join(vargs.incoming_path, "logs", f"aodn_ftp_push.log")
    LOGGER = imos_logging.logging_start(logger_name=f"{datetime.now().strftime("%Y%m%dT%H%M%S")}_aodn_ftp_push_logger", logging_filepath=log_file)
    
    LOGGER.info(f"Uploader script started".upper())
    
    # wb = WaveBuoy(buoy_type="sofar")

    buoys_to_proc = pd.read_csv(os.path.join(vargs.incoming_path, "delayed_mode_buoys_to_process.csv"))

    # buoys_to_proc = buoys_to_proc.loc[buoys_to_proc['dep_id'] == 1]

    BASE_PATH = r"C:\Users\00116827\cwb\dm_data"

    for row, dep in buoys_to_proc.iterrows():

        if dep['aodn_ready'] != 1:
            continue
        
        LOGGER.info(f"="*250)
        LOGGER.info(f"DEP_ID: {dep['dep_id']} - {dep['datapath']}")

        try:
            
            # GRABBING FILE PATHS --------------------------
            patterns = [
                "*DM_WAVE-PARAMETERS*",
                "*DM_WAVE-SPECTRA*",
                "*DM_WAVE-RAW*",
            ]

            files_to_push = []

            files_path = os.path.join(BASE_PATH, f"{dep["region"]}waves", dep["datapath"], "processed_py")

            for p in patterns:
                files_to_push.extend(
                    glob.glob(os.path.join(files_path, p))
                )

            files_to_push = list(set(files_to_push))

            # STABLISHING CONNECTION ------------------------

            ncp = ncPusher(host=os.getenv("FTP_HOST"),
                        user=os.getenv("FTP_USER"),
                        password=os.getenv("FTP_PASSWORD"))
            LOGGER.info("successfully connected with FTP server")

            working_dir = "wave"
            ncp.change_dir(working_dir)
            LOGGER.info(f"FTP working dir changed to '/{working_dir}'")

            ncp._secure_data_connection()

            for file in files_to_push:
                LOGGER.info(f"-"*15)
                LOGGER.info(f"DEP_ID: {dep['dep_id']} - {os.path.basename(file)}")

                try: 
                    
                    with Dataset(file) as ds:
                        LOGGER.info("valid netcdf format")
                        
                        # Check WATCH_CIRCLE_flag dtypes -------
                        if not "RAW-DISPLACEMENTS" in file:

                            flag_values = np.asarray(ds["WATCH_CIRCLE_flag"].flag_values)
                            
                            if flag_values.dtype != np.int8:
                                raise TypeError(f"WATCH_CIRCLE_flag flag_values attribute is of type {type(ds["WATCH_CIRCLE_flag"].flag_values[0])}, expected numpy.int8")

                            if ds["WATCH_CIRCLE_flag"].dtype != np.int8:
                                raise TypeError(f"WATCH_CIRCLE_flag data is of type {ds["WATCH_CIRCLE_flag"].dtype}, expected numpy.int8")

                            LOGGER.info("WATCH_CIRCLE_flag CF compliant (flag_values and data of dtype np.int8)")
                        
                        # Check TIME/TIME_LOCATION monotonicity -------
                        check_time_monotoniciy(ds, LOGGER)

                    ncp.push_file_to_ftp_dm(file=file)
                    LOGGER.info(f"file pushed")
                
                except Exception as e:
                    error_message = f"Error pushing file: {file}"
                    LOGGER.error(error_message)
                    LOGGER.error(str(e), exc_info=True)
                    continue

            # ncp.quit()             

        except Exception as e:
            LOGGER.error(str(e), exc_info=True)
            logger_file_path = imos_logging.get_log_file_path(LOGGER)
            print(logger_file_path)
            imos_logging.logging_stop(logger=LOGGER)
            error_logger_file_path = imos_logging.rename_push_log_if_error(file_path=logger_file_path, add_runtime=True)
            # if vargs.email_alert:
            #     e = Email(script_name=os.path.basename(__file__),
            #             email=os.getenv("EMAIL_TO"),
            #             log_file_path=error_logger_file_path)
            #     e.send()

        #     if files_to_push:
        #         LOGGER.info(f"files to push:")
        #         LOGGER.info(json.dumps(files_to_push, indent=6, default=str))
                
        #         working_dir = "wave"
        #         ncp.change_dir(working_dir)
        #         LOGGER.info(f"FTP working dir changed to '/{working_dir}'")

        #         ncp._secure_data_connection()
                
        #         files_report = ncp.create_files_report()
        #         for file in files_to_push:
        #             LOGGER.info(f"="*60)
        #             LOGGER.info(f"pushing {file['file_name']}")
                    
        #             validation_results = []
        #             try:
                        
        #                 validation_results.append(ncValidator().validade_nc_integrity(file["file_path"]))
        #                 LOGGER.info("file integrity validation passed")

        #                 if not all("passed" in result for result in validation_results):
        #                     raise Exception("One or more validation checks failed.")   
                            
        #                 ncp.push_file_to_ftp(file=file)
        #                 LOGGER.info(f"file pushed: {file['file_name']}")

        #                 ncp.update_files_report(files_report=files_report,
        #                                         file=file,
        #                                         error=False)
        #                 LOGGER.info(f"="*60)

        #             except Exception as e:
        #                 error_message = f"Error pushing file: {file['file_name']}"
        #                 LOGGER.error(error_message)
        #                 LOGGER.error(str(e), exc_info=True)
                        
        #                 ncp.update_files_report(files_report=files_report,
        #                                         file=file,
        #                                         error=True,
        #                                         validation_results=validation_results,
        #                                         exception=e)
                        
        #                 continue
                    
        #         ncp.quit()

        # LOGGER.info(f"Files pushed: {json.dumps(files_report['files_pushed'], indent=6)}")
        # LOGGER.info(f"Pusher script finished".upper())

        # if files_report["files_error"]:
        #     raise Exception(f"Error pushing one or more files: {json.dumps(files_report['files_error'], indent=6)}")
       
        # # LOGGER.info("pushing successful")
        # except Exception as e:
        #     LOGGER.error(str(e), exc_info=True)
        #     logger_file_path = imos_logging.get_log_file_path(LOGGER)
        #     print(logger_file_path)
        #     imos_logging.logging_stop(logger=LOGGER)
        #     error_logger_file_path = imos_logging.rename_push_log_if_error(file_path=logger_file_path, add_runtime=True)
        #     if vargs.email_alert:
        #         e = Email(script_name=os.path.basename(__file__),
        #                 email=os.getenv("EMAIL_TO"),
        #                 log_file_path=error_logger_file_path)
        #         e.send()

if __name__ == "__main__":
    main()