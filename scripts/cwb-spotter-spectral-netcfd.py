import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
import pandas as pd

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.netcdf.writer import ncWriter, ncAttrsComposer, ncAttrsExtractor, ncProcessor, ncMetaDataLoader
from wavebuoy_nrt.utils import args, IMOSLogging, generalTesting


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
    # "MtEliza", "Hillarys", "Central"
    wb.buoys_metadata = wb.buoys_metadata.loc[["Hillarys"]].copy()
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