
import os

from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

from wavebuoy.config.config import FILES_OUTPUT_PATH, NC_FILE_NAME_TEMPLATE, REGION_TO_INSTITUTION

class NetCDFFileHandler():
    """
    Class used to handle NetCDF files.
    """
    nc_file_paths_to_process = []
    latest_processed_datetime = None
    
    def __init__(self):
        pass

    def generate_filename(self, site_id:str, institution:str,date_time_month:datetime) -> str:
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

    def lookup_netcdf_files(self, 
                            institution:str,
                            site_id:str, 
                            latest_available_datetime:datetime,
                            minimum_datetime_recursion:datetime=datetime(2020,1,1), # This should be less than the minimum Datetime of the first spotter we ever deployed
                            threshold:timedelta=timedelta(hours=24)) -> str:

        
        if latest_available_datetime < minimum_datetime_recursion:
            print(f"""No stored data from {minimum_datetime_recursion}.
                  Considering {site_id} as a new site (i.e. New NetCDF created for {latest_available_datetime.year}-{latest_available_datetime.month}).
                  """)
            return

        # nc_file_datetime = latest_available_datetime.replace(day=1).strftime("%Y%m%d") 
        year = latest_available_datetime.year  
        month = latest_available_datetime.month
        nc_file_name = NC_FILE_NAME_TEMPLATE.format(institution=REGION_TO_INSTITUTION[institution],# Temporary data
                                                    year=year,
                                                    month=month,
                                                    site_id=site_id.upper())
        nc_file_path = os.path.join(FILES_OUTPUT_PATH, nc_file_name)

        if os.path.isfile(nc_file_path):
            self.nc_file_paths_to_process.append(nc_file_path)
            nc_dataset = xr.open_dataset(nc_file_path)
            
            if self.latest_processed_datetime is None:
                latest_processed_datetime = self.get_latest_processed_date_time(nc_dataset)
        
            period_enough = self.check_period(threshold=threshold,
                                              nc_dataset=nc_dataset,
                                              latest_available_datetime=latest_available_datetime,
                                              latest_processed_datetime=latest_processed_datetime)
            
            if not period_enough:
                self.lookup_netcdf_files(institution=institution,
                                        site_id=site_id,
                                        latest_available_datetime=latest_available_datetime-timedelta(days=30)
                                            )
            
            return self.nc_file_paths_list 
        
        else:
            print(f"{nc_file_path} does not exist. Probably first data point of the month.")
            return self.lookup_netcdf_files(institution=institution,
                                    site_id=site_id,
                                    latest_available_datetime=latest_available_datetime-timedelta(days=30)
                                        )       

    # def netcdf_loader(self, nc_file_path:str):
    #     return xr.open_dataset(nc_file_path)

    def check_period(self, 
                    threshold:timedelta,
                    nc_dataset:xr.Dataset,
                    latest_available_datetime:datetime,
                    latest_processed_datetime:datetime):
        
        min_datetime = datetime.fromtimestamp(nc_dataset["TIME"].min().values.astype('datetime64[s]').astype(int))
        
        if (latest_processed_datetime - min_datetime) > threshold:
            print("Enough data points to be processed.")
            return True 
        else:
            print("NC File does not have enough data points to be processed.")
            return False

    def get_latest_processed_date_time(self, nc_dataset:xr.Dataset):
        return datetime.fromtimestamp(
                                    (nc_dataset["TIME"]
                                     .max()
                                     .values
                                     .astype('datetime64[s]')
                                     .astype(int))
                                    )

    def get_site_ids(self, buoys_metadata:pd.DataFrame) -> list:
        return buoys_metadata.name.list()


    def _validade_site_id(self, site_id:str, site_ids:list) -> bool:
        if site_id in site_ids:
            return True
        else:
            return False
        
    def _generate_daterange(self, latest_available_datetime:datetime,
                    threshold:timedelta):
        start_date = latest_available_datetime - threshold
