import argparse
import logging
import tempfile
import os
import sys
from datetime import datetime, timedelta

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

    
    return vargs


class IMOSLogging:

    def __init__(self, logging_filepath):
        self.logging_filepath = logging_filepath
        self.logger = []

    def logging_start(self):
        """ start logging using logging python library
        output:
           logger - similar to a file handler
        """
        if not os.path.exists(os.path.dirname(self.logging_filepath)):
            os.makedirs(os.path.dirname(self.logging_filepath))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # create a file handler
        handler = logging.FileHandler(self.logging_filepath)
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)
        return self.logger

    def logging_stop(self):
        """ close logging """
        # closes the handlers of the specified logger only
        x = list(self.logger.handlers)
        for i in x:
            self.logger.removeHandler(i)
            i.flush()
            i.close()
