import argparse
import logging
import tempfile
import os
import sys
import re
from datetime import datetime, timedelta
import pickle


GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")


def args():
    """
    Returns the script arguments

        Parameters:

        Returns:
            vargs (obj): input arguments
    """
    parser = argparse.ArgumentParser(description='Creates NetCDF files.\n '
                                     'Prints out the path of the new locally generated NetCDF file.')
    
    parser.add_argument('-o', '--output-path', dest='output_path', type=str, default=None,
                        help="output directory of netcdf file",
                        required=True)
    
    parser.add_argument('-w', '--window', dest='window', type=str, default=24,
                        help="desired window from present backwards to be processed and qualified. Default to 24, please check argument --window-unit for the right desired unit.",
                        required=False)

    parser.add_argument('-wu', '--window-unit', dest='window_unit', type=str, default="hours",
                        help="desired window unit (hours:Default, months).",
                        required=False)

    parser.add_argument('-pp', '--period-to-process', dest='period_to_process', type=str, default=None, nargs=2,
                        help="desired period to be extracted, processed and qualified. Please pass start and end dates as YYYY-mm-ddTHH:MM separated by a blank space.",
                        required=False)
    
    parser.add_argument('-pqc', '--period-to-qualify', dest='period_to_qualify', type=str, default=None, nargs=2,
                        help="desired period to be qualified. Please pass start and end dates as YYYY-mm-ddTHH:MM separated by a blank space.",
                        required=False)

    parser.add_argument('-bf', '--backfill', dest='backfill', action="store_true",
                        help="wether the user wants to backfill the data from the latest available time back to the last processed time.",
                        required=False)
    # parser.add_argument('-p', '--push-to-incoming', dest='incoming_path', type=str, default=None,
    #                     help="incoming directory for files to be ingested by AODN pipeline (Optional)",
    #                     required=False)
    
    vargs = parser.parse_args()

    # if vargs.output_path is None:
    #     vargs.output_path = tempfile.mkdtemp()
    
    if not os.path.exists(vargs.output_path):
        try:
            os.makedirs(vargs.output_path)
        except Exception:
            raise ValueError('{path} not a valid path'.format(path=vargs.output_path))
            sys.exit(1)

    # if vargs.incoming_path:
    #     if not os.path.exists(vargs.incoming_path):
    #         raise ValueError('{path} not a valid path'.format(path=vargs.incoming_path))
    else:
        vargs.incoming_path = None

    if vargs.period_to_process:
        vargs.period_to_process = vargs.period_to_process.split()
        vargs.period_to_process_start_date = datetime.strptime(vargs.period_to_process[0],"%Y-%m-%dT%H:%M")
        vargs.period_to_process_end_date = datetime.strptime(vargs.period_to_process[1],"%Y-%m-%dT%H:%M")

    if vargs.period_to_qualify:
        vargs.period_to_qualify = vargs.period_to_qualify.split()
        vargs.period_to_qualify_start_date = datetime.strptime(vargs.period_to_qualify[0],"%Y-%m-%dT%H:%M")
        vargs.period_to_qualify_end_date = datetime.strptime(vargs.period_to_qualify[1],"%Y-%m-%dT%H:%M")

    if vargs.backfill:
        backfill = True
    else:
        backfill = False

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
        self.logging_filepath = logging_filepath

        if not os.path.exists(os.path.dirname(self.logging_filepath)):
            os.makedirs(os.path.dirname(self.logging_filepath))

        self.logger = logging.getLogger(logger_name)

        if not self.logger.hasHandlers():
            self.logger.setLevel(level)

            handler = logging.FileHandler(self.logging_filepath, mode="w")
            handler.setLevel(level)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

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
    
    def rename_log_file_if_error(self, site_name: str, file_path):
        site_name = site_name.upper()
        runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        pattern = f"{site_name}"
        new_name = f"{runtime}_{site_name}_error"

        new_file_name = re.sub(pattern, new_name, file_path)
        if os.path.exists(new_file_name):
            os.replace(file_path, new_file_name)
        else:
            os.rename(file_path, new_file_name)
        GENERAL_LOGGER.info(f"{site_name} log file renamed as {new_file_name}")

class generalTesting:
    def generate_pickle_file(self, data, file_name: str, site_name: str):
        file_path = f"tests/pickle_files/{site_name}_{file_name}.pkl"
        with open(file_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
            print(f"saved pkl as output_path/test_files/{site_name}_{file_name}.pkl")
        
    def open_pickle_file(self, file_name: str, site_name: str):
        file_path = f"tests/pickle_files/{site_name}_{file_name}.pkl"
        with open(file_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            print(f"openned pkl as output_path/test_files/{site_name}_{file_name}.pkl")
        return data