import os
import logging
from datetime import datetime, timedelta
import re
import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

def args_processing():
 
    parser = argparse.ArgumentParser(description='Creates NetCDF files.\n '
                                     'Prints out the path of the new locally generated NetCDF file.')
    
    parser.add_argument('-o', '--output-path', dest='output_path', type=str, default=None,
                        help="output directory of netcdf file",
                        required=False)
    
    parser.add_argument('-l', '--log-path', dest='log_path', type=str, default=None,
                        help="directory where SD card files are stored",
                        required=False)

    parser.add_argument('-pp', '--deploy-dates', dest='deploy_dates', type=str, default=None, nargs=2,
                        help="deployment dates period to be processed. Please pass start and end dates as YYYYmmdd YYYYmmdd, separated by a blank space.",
                        required=False)

    parser.add_argument('-r', '--region', dest='region', type=str, default=None,
                        help="directory where SD card files are stored",
                        required=True)

    vargs = parser.parse_args()
    
    # if not os.path.exists(vargs.output_path):
    #     try:
    #         os.makedirs(vargs.output_path)
    #     except Exception:
    #         raise ValueError('{path} not a valid path'.format(path=vargs.output_path))
    #         sys.exit(1)

    # if not os.path.exists(vargs.log_path):
    #     raise ValueError('{path} not a valid path'.format(path=vargs.log_path))

    if vargs.deploy_dates:
        vargs.deploy_dates_start = datetime.strptime(vargs.deploy_dates[0],"%Y%m%d")
        vargs.deploy_dates_end = datetime.strptime(vargs.deploy_dates[1],"%Y%m%d")


    return vargs



def args_processing_dm():

    parser = argparse.ArgumentParser()
 
    parser.add_argument('-l', '--log-path', dest='log_path', type=str, default=None,
                        help="Directory where SD card files are stored",
                        required=True)
    
    parser.add_argument('-d', '--deploy-dates', dest='deploy_dates', type=str, default=None, nargs=2,
                        help="Deployment datetimes period to be processed. Please pass start and end dates as YYYYmmddTHHMMSS YYYYmmdd, separated by a blank space",
                        required=True)

    parser.add_argument('-ed', '--enable-dask', dest='enable_dask', action='store_true',
                    help="Whether to enable spectra calculation with dask threading")

    parser.add_argument('-ot', '--output-type', dest='output_type', type=str, default="netcdf",
                    help="Whether to save outputs as csv or netcdf, when applicable.")

    vargs = parser.parse_args()

    if not os.path.exists(vargs.log_path):
        raise ValueError('{path} not a valid path'.format(path=vargs.log_path))
    else:
        vargs.output_path = os.path.join(vargs.log_path, "processed")
        if not os.path.exists(vargs.output_path):
            os.makedirs(vargs.output_path)

    if vargs.deploy_dates:
        vargs.deploy_dates_start = datetime.strptime(vargs.deploy_dates[0],"%Y%m%dT%H%M%S")
        vargs.deploy_dates_end = datetime.strptime(vargs.deploy_dates[1],"%Y%m%dT%H%M%S")

    return vargs


class IMOSLogging:
    unexpected_error_message = "An unexpected error occurred when processing {site_name}\n Please check the site log for details"

    def __init__(self):
        pass

    def logging_start(self, logging_filepath, logger_name="general_logger", level=logging.INFO):
        """
        Start logging using the Python logging library.
        Parameters:
            logger_name (str): Name of the logger to create or retrieve.
            level (int): Logging level (default: logging.INFO).
        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        self.logging_filepath = os.path.join(logging_filepath, "logs", logger_name)

        if not os.path.exists(os.path.dirname(self.logging_filepath)):
            os.makedirs(os.path.dirname(self.logging_filepath))

        self.logger = logging.getLogger(logger_name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(level)
        handler = logging.FileHandler(self.logging_filepath, mode="w")
        handler.setLevel(level)
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.addHandler(stream_handler)

        return self.logger

    def logging_stop(self, logger):
        """Close logging handlers for the current logger."""
        handlers = list(logger.handlers)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def get_log_file_path(self, logger):
        return logger.handlers[0].baseFilename
    
    def rename_log_file_if_error(self, site_name: str, file_path, script_name: str, add_runtime: bool = True):
        site_name = site_name.upper()
        runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        pattern = f"{site_name}_{script_name}"
        new_name = "ERROR_" + f"{site_name}_{script_name}"
        if add_runtime:
            new_name += f"_{runtime}"

        new_file_name = re.sub(pattern, new_name, file_path)
        if os.path.exists(new_file_name):
            os.replace(file_path, new_file_name)
        else:
            os.rename(file_path, new_file_name)
        # GENERAL_LOGGER.info(f"{site_name} log file renamed as {new_file_name}")

        return os.path.join(file_path, new_file_name)

    def rename_push_log_if_error(self, file_path: str, add_runtime: bool = True):
        runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        pattern = "aodn_ftp_push"
        new_name = f"ERROR_aodn_ftp_push"
        if add_runtime:
            new_name += f"_{runtime}"

        new_file_name = re.sub(pattern, new_name, file_path)
        if os.path.exists(new_file_name):
            os.replace(file_path, new_file_name)
        else:
            os.rename(file_path, new_file_name)

        return os.path.join(file_path, new_file_name)