
import os
import logging

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import xarray as xr
import glob
import re
from typing import Union, List


from wavebuoy_nrt.config.config import FILES_OUTPUT_PATH, NC_FILE_NAME_TEMPLATE, REGION_TO_INSTITUTION

GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")


class NetCDFFileHandler():
    """
    Class used to handle NetCDF files.
    """
    
    def __init__(self):
        pass

    def generate_filename(self, site_id: str, institution: str,date_time_month: datetime) -> str:
        """
        Generates an IMOS-compliant wave buoy data file name given a site ID and a datetime.

        PARAMETERS:
        ----------
        site_id : str
            The identifier for the wave buoy site.
        
        date_time_month : datetime
            A datetime object representing the month for which the data is being generated. Must always be composed of the 1st of the month.

        RETURNS:
        -------
        str
            A string representing the generated IMOS-compliant filename in the format:
            '{IMOS/UNIVERSITY}_{YYYYMMDD}_{site_id}_RT_WAVE-PARAMETERS_monthly.nc'
        """
        if date_time_month.day != 1:
            raise ValueError("Provide a datetime of the 1st of the desired month.")

        site_id = site_id.upper()

        date_time_month_str = date_time_month.strftime("%Y%m%d")
        return NC_FILE_NAME_TEMPLATE.format(date_time_month=date_time_month_str,
                                            site_id=site_id+2,
                                            institution=institution)

    def lookup_netcdf_files_needed(self, 
                            institution: str,
                            site_id: str, 
                            latest_available_datetime: datetime,
                            #minimum_datetime_recursion:datetime=datetime(2020,1,1), # This should be less than the minimum Datetime of the first spotter we ever deployed
                            window: int=24,
                            window_unit: str="hours") -> str:
            
            window_start_date = self.generate_window_start_time(latest_available_datetime=latest_available_datetime,
                                                                    window=window,
                                                                    window_unit=window_unit)
            monthly_daterange = self._generate_monthly_daterange(start_date=window_start_date,
                                                                end_date=latest_available_datetime)
            nc_file_paths_needed = []
            for month in monthly_daterange:
                nc_file_name = NC_FILE_NAME_TEMPLATE.format(institution=REGION_TO_INSTITUTION[institution],# Temporary data
                                                    monthly_datetime=month.strftime("%Y%m%d"),
                                                    site_id=site_id.upper())
                print(nc_file_name)
                nc_file_path = os.path.join(FILES_OUTPUT_PATH, nc_file_name)
                nc_file_paths_needed.append(nc_file_path)

            return nc_file_paths_needed
    
    def generate_window_start_time(self,
                                       latest_available_datetime: datetime, 
                                       window: int, 
                                       window_unit: str="hours"):
        kwargs = {window_unit:window}
        return latest_available_datetime - relativedelta(**kwargs)
    
    def _generate_monthly_daterange(self, start_date: datetime, end_date: datetime):
        
        monthly_daterange = pd.date_range(start_date, end_date, freq="MS")
        
        window_start_date_str = (start_date
                                .replace(day=1)
                                .date()
                                .strftime("%Y-%m-%d"))

        if window_start_date_str not in monthly_daterange:
                monthly_daterange = monthly_daterange.insert(item=window_start_date_str, loc=0)

        return monthly_daterange    

    def get_available_nc_files(self, institution: str, site_id: str) -> list:
        
        nc_file_filter = NC_FILE_NAME_TEMPLATE.format(institution=REGION_TO_INSTITUTION[institution],# Temporary data
                                                    monthly_datetime="*",
                                                    site_id=site_id.upper())
        nc_file_filter = f"{REGION_TO_INSTITUTION[institution]}*{site_id.upper()}*.nc"
        nc_file_path = os.path.join(FILES_OUTPUT_PATH, nc_file_filter)

        return glob.glob(nc_file_path)
    
    def get_latest_nc_file_available(self, institution:str, site_id:str) -> str:
        
        available_nc_files = self.get_available_nc_files(institution=institution,
                                                          site_id=site_id)
        date_pattern = re.compile(r"_(\d{8})_")
        most_recent_file_path = max(available_nc_files, key=lambda x: int(date_pattern.search(x).group(1)))

        return most_recent_file_path

    def get_latest_processed_time(self, nc_file_path: str) -> datetime:
        
        return pd.to_datetime((xr.open_dataset(nc_file_path)
                       ["TIME"]
                       .max()
                       .values)
                    ).to_pydatetime()

    def get_earliest_nc_file_available(self, institution:str, site_id:str) -> str:
        
        available_nc_files = self.get_available_nc_files(institution=institution,
                                                          site_id=site_id)
        date_pattern = re.compile(r"_(\d{8})_")
        most_recent_file_path = min(available_nc_files, key=lambda x: int(date_pattern.search(x).group(1)))

        return most_recent_file_path

    def get_earliest_processed_time(self, nc_file_path: str) -> datetime:
        
        return pd.to_datetime((xr.open_dataset(nc_file_path)
                       ["TIME"]
                       .min()
                       .values)
                    ).to_pydatetime()

    def check_nc_files_needed_available(self, nc_files_needed: list, nc_files_available: list):
        
        check = [file in nc_files_available for file in nc_files_needed]
        nc_file_paths = [file for file in nc_files_needed if file in nc_files_available]

        if all(check) or any(check):
            availability_check = True
        elif not all(check):
            availability_check = False
            
        return availability_check, nc_file_paths 
        
    def load_datasets(self, nc_file_paths: Union[List[str], str]) -> pd.DataFrame:

        if not nc_file_paths:
            return pd.DataFrame()

        if isinstance(nc_file_paths, list):
            global_dataframe = pd.DataFrame([])
            for nc_file in nc_file_paths:
                dataframe = (xr.open_dataset(nc_file, engine="netcdf4")
                             .to_dataframe()
                             .reset_index())
                global_dataframe = pd.concat([global_dataframe,dataframe])
        elif isinstance(nc_file_paths, str):
            global_dataframe = (xr.open_dataset(nc_file_paths, engine="netcdf4")
                                .to_dataframe()
                                .reset_index())

        # TEMPORARY SETUP
        global_dataframe["check"] = "prev"

        return global_dataframe

    def get_site_ids(self, buoys_metadata: pd.DataFrame) -> list:
        return buoys_metadata.name.list()

    def _validade_site_id(self, site_id: str, site_ids: list) -> bool:
        if site_id in site_ids:
            return True
        else:
            return False


    # def check_period(self, 
    #                 window: timedelta,
    #                 nc_dataset: xr.Dataset,
    #                 latest_available_datetime: datetime,
    #                 latest_processed_datetime: datetime):
        
    #     min_datetime = datetime.fromtimestamp(nc_dataset["TIME"].min().values.astype('datetime64[s]').astype(int))
        
    #     if (latest_processed_datetime - min_datetime) > window:
    #         print("Enough data points to be processed.")
    #         return True 
    #     else:
    #         print("NC File does not have enough data points to be processed.")
    #         return False


    # def get_latest_processed_datetime(self, nc_dataset:xr.Dataset):
    #     return datetime.fromtimestamp(
    #                                 (nc_dataset["TIME"]
    #                                  .max()
    #                                  .values
    #                                  .astype('datetime64[s]')
    #                                  .astype(int))
    #                                 )

    # def lookup_netcdf_files1(self, 
    #                         institution:str,
    #                         site_id:str, 
    #                         latest_available_datetime:datetime,
    #                         minimum_datetime_recursion:datetime=datetime(2020,1,1), # This should be less than the minimum Datetime of the first spotter we ever deployed
    #                         window:timedelta=timedelta(hours=24)) -> str:

        
    #     if latest_available_datetime < minimum_datetime_recursion:
    #         print(f"""No stored data from {minimum_datetime_recursion}.
    #               Considering {site_id} as a new site (i.e. New NetCDF created for {latest_available_datetime.year}-{latest_available_datetime.month}).
    #               """)
    #         return

    #     # nc_file_datetime = latest_available_datetime.replace(day=1).strftime("%Y%m%d") 
    #     year = latest_available_datetime.year  
    #     month = latest_available_datetime.month
    #     nc_file_name = NC_FILE_NAME_TEMPLATE.format(institution=REGION_TO_INSTITUTION[institution],# Temporary data
    #                                                 year=year,
    #                                                 month=month,
    #                                                 site_id=site_id.upper())
    #     nc_file_path = os.path.join(FILES_OUTPUT_PATH, nc_file_name)

    #     if os.path.isfile(nc_file_path):
    #         self.nc_file_paths_to_process.append(nc_file_path)
    #         nc_dataset = xr.open_dataset(nc_file_path)
            
    #         if self.latest_processed_datetime is None:
    #             latest_processed_datetime = self.get_latest_processed_date_time(nc_dataset)
        
    #         period_enough = self.check_period(window=window,
    #                                           nc_dataset=nc_dataset,
    #                                           latest_available_datetime=latest_available_datetime,
    #                                           latest_processed_datetime=latest_processed_datetime)
            
    #         if not period_enough:
    #             self.lookup_netcdf_files(institution=institution,
    #                                     site_id=site_id,
    #                                     latest_available_datetime=latest_available_datetime-timedelta(days=30)
    #                                         )
            
    #         return self.nc_file_paths_list 
        
    #     else:
    #         print(f"{nc_file_path} does not exist. Probably first data point of the month.")
    #         return self.lookup_netcdf_files(institution=institution,
    #                                 site_id=site_id,
    #                                 latest_available_datetime=latest_available_datetime-timedelta(days=30)
    #                                     )       

    # def netcdf_loader(self, nc_file_path:str):
    #     return xr.open_dataset(nc_file_path)

    # def _check_nc_files_exist(self, nc_file_paths_to_process: list) -> list:
        
    #     # Check for all files (best/expected scenario)
    #     if all([os.path.exists(f) for f in nc_file_paths_to_process]):
    #         return True
        
    #     # Check what files are missing
    #     else:
    #         missing_files = []
    #         for file in nc_file_paths_to_process:
    #             if not os.path.exists(file):
    #                 print("File exists")
    #                 missing_files.append(file)

    #         return missing_files