import os
import json

from wavebuoy_nrt.ftp.ftp import ncPusher
from wavebuoy_nrt.utils import args_pushing, IMOSLogging, generalTesting
from wavebuoy_nrt.netcdf.validation import ncValidator


def main():

    vargs = args_pushing()
    
    imos_logging = IMOSLogging() 
    log_file = os.path.join(vargs.incoming_path, "logs", f"push.log")
    LOGGER = imos_logging.logging_start(logger_name="push_logger", logging_filepath=log_file)
    
    
    ncp = ncPusher(host="127.0.0.1",
                user="testuser",
                password="testuser")
    LOGGER.info("successfully connected with FTP server")

    files_to_push = ncp.grab_nc_files_to_push(incoming_path=vargs.incoming_path, lookback_hours=vargs.lookback_hours)

    if files_to_push:
        LOGGER.info(f"files to push:")
        LOGGER.info(json.dumps(files_to_push, indent=6, default=str))
        
        ncp._secure_data_connection()
        for file in files_to_push:
            try:
                # check files are okay to be pushed (naming, contents, etc)
                ncValidator().validate_file_name(file["file_name"])
                ncValidator().validade_nc_integrity(file["file_path"])

                # nv.validate_nc_integrity()
                # nv.validate_compliance() # compare file name, data and attributes 

                ncp.push_file_to_ftp(file=file)
                LOGGER.info(f"file pushed: {file["file_name"]}")

            except Exception as e:
                error_message = f"Error pushing file: {file["file_name"]}"
                LOGGER.error(error_message)
                LOGGER.error(str(e), exc_info=True)
                # Closing current site logging
                # logger_file_path = imos_logging.get_log_file_path(LOGGER)
                imos_logging.logging_stop(logger=LOGGER)
                # imos_logging.rename_log_file_if_error(site_name=site.name, file_path=logger_file_path,
                #                                             add_runtime=False)
                continue

    else:
        LOGGER.info("no files to push. Aborting.")

    LOGGER.info("pushing successful")




if __name__ == "__main__":
    main()