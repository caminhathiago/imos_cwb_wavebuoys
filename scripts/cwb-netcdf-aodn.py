import os
import json
import sys

from dotenv import load_dotenv

from wavebuoy_nrt.ftp.ftp import ncPusher
from wavebuoy_nrt.utils import args_pushing, IMOSLogging, generalTesting
from wavebuoy_nrt.netcdf.validation import ncValidator
from wavebuoy_nrt.alerts.email import Email

load_dotenv()

def main():
    

    vargs = args_pushing()
    
    imos_logging = IMOSLogging() 
    log_file = os.path.join(vargs.incoming_path, "logs", f"aodn_ftp_push.log")
    LOGGER = imos_logging.logging_start(logger_name="aodn_ftp_push_logger", logging_filepath=log_file)
    
    LOGGER.info(f"Uploader script started".upper())
    LOGGER.info(f"pushing NC files created/modified within the last {vargs.lookback_hours} hours")
    
    try:
        ncp = ncPusher(host=os.getenv("FTP_HOST_TEST"),
                    user=os.getenv("FTP_USER_TEST"),
                    password=os.getenv("FTP_PASSWORD_TEST"))
        LOGGER.info("successfully connected with FTP server")

        files_to_push = ncp.grab_nc_files_to_push(incoming_path=vargs.incoming_path, lookback_hours=vargs.lookback_hours)

        if files_to_push:
            LOGGER.info(f"files to push:")
            LOGGER.info(json.dumps(files_to_push, indent=6, default=str))
            
            working_dir = "wave"
            ncp.change_dir(working_dir)
            LOGGER.info(f"FTP working dir changed to '/{working_dir}'")

            ncp._secure_data_connection()
            
            files_report = ncp.create_files_report()
            for file in files_to_push:
                LOGGER.info(f"="*60)
                LOGGER.info(f"pushing {file["file_name"]}")
                
                try:
                    # ncValidator().validate_file_name(file["file_name"])
                    LOGGER.info("file name validation passed")
                    
                    ncValidator().validade_nc_integrity(file["file_path"])
                    LOGGER.info("file integrity validation passed")

                    ncp.push_file_to_ftp(file=file)
                    LOGGER.info(f"file pushed: {file["file_name"]}")

                    ncp.update_files_report(files_report=files_report,
                                            file=file,
                                            error=False)
                    LOGGER.info(f"="*60)

                except Exception as e:
                    error_message = f"Error pushing file: {file["file_name"]}"
                    LOGGER.error(error_message)
                    LOGGER.error(str(e), exc_info=True)
                    ncp.update_files_report(files_report=files_report,
                                            file=file,
                                            error=True,
                                            exception=e)
                    continue
                
            ncp.quit()

        else:
            LOGGER.info("no files to push. Aborting.")
            sys.exit(1)

        LOGGER.info(f"Files pushed: {json.dumps(files_report["files_pushed"], indent=6)}")
        LOGGER.info(f"Uploader script finished".upper())

        if files_report["files_error"]:
            raise Exception(f"Error pushing one or more files: {json.dumps(files_report["files_error"], indent=6)}")
       
        # LOGGER.info("pushing successful")
    except Exception as e:
        LOGGER.error(str(e), exc_info=True)
        logger_file_path = imos_logging.get_log_file_path(LOGGER)
        print(logger_file_path)
        imos_logging.logging_stop(logger=LOGGER)
        error_logger_file_path = imos_logging.rename_push_log_if_error(file_path=logger_file_path, add_runtime=True)
        e = Email(script_name=os.path.basename(__file__),
                  email=os.getenv("EMAIL_TO"),
                  log_file_path=error_logger_file_path)
        e.send()

if __name__ == "__main__":
    main()