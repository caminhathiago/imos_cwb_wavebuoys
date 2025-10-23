import os
from typing import List
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from wavebuoy_nrt.processor.spotter import SpotterWaveBuoy
from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.utils import FilesHandler

load_dotenv()

GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")


class WaveBuoy(FilesHandler, NetCDFFileHandler, SpotterWaveBuoy):
    
    def __init__(self, buoy_type:str, buoys_metadata_file_name:str="buoys_metadata.csv", exclude_drifters:bool = False):
        
        self._exclude_drifters = exclude_drifters
        self.buoys_metadata = self._get_buoys_metadata(buoy_type=buoy_type, buoys_metadata_file_name=buoys_metadata_file_name)
        
        # self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)
        # self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata)
        # self.sites_per_region = self._get_sites_per_region(buoys_metadata=self.buoys_metadata)

    def _get_buoys_metadata(self, buoy_type:str, buoys_metadata_file_name:str):

        # try:
        file_path = os.path.join(os.getenv('IRDS_PATH'), "Data", "website", "auswaves")
        if not os.path.exists(file_path):
            raise FileNotFoundError("No such directory for buoys metadata: {}")

        buoys_metadata_path = self._get_file_path(file_name=buoys_metadata_file_name, file_path=file_path)
        # buoys_metadata_path = r"C:\Users\00116827\cwb\imos_cwb_wavebuoys\tests\nrt\20251008_drifters\buoys_metadata.csv"
        buoys_metadata = pd.read_csv(buoys_metadata_path)

        buoys_metadata = self._select_buoy_type(buoy_type=buoy_type, buoys_metadata=buoys_metadata)

        buoys_metadata["region"] = self._get_regions(buoys_metadata=buoys_metadata)

        buoys_metadata = buoys_metadata.set_index('name')

        if self._exclude_drifters:
            buoys_metadata = self._exclude_drifters_from_buoys_metadata(buoys_metadata=buoys_metadata)

        buoys_metadata = self._filter_process_aodn(buoys_metadata=buoys_metadata)
        GENERAL_LOGGER.info("Buoys metadata grabbed successfully")
        
        return buoys_metadata

        # except:
        #     error_message = "Loading and processing buoys_metadata.csv unsuccessful. Check if the file is corrupted or if its structure has been changed"
        #     GENERAL_LOGGER.error(error_message, exc_info=True)
        
    def _select_buoy_type(self, buoy_type:str, buoys_metadata:pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.loc[buoys_metadata["type"] == buoy_type]

    def _filter_process_aodn(self, buoys_metadata:pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.loc[buoys_metadata["process_aodn"] == 1]

    def _exclude_drifters_from_buoys_metadata(self, buoys_metadata: pd.DataFrame) -> pd.DataFrame:
        name_constraint = "drift".upper()
        indexes = [index for index in buoys_metadata.index if name_constraint not in index.upper()]
        return buoys_metadata.loc[indexes]

    def _get_site_ids(self, buoys_metadata=pd.DataFrame) -> list:
        return buoys_metadata.index.values
    
    def _get_sites(self, buoys_metadata=pd.DataFrame) -> list:
        return (buoys_metadata
                .reset_index()
                [["name","region"]]
                .to_dict(orient="records")
        )

    def _sort_sites_by_sofar_token(self, buoys_metadata=pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.sort_values("sofar_token")

    def get_latest_available_date_time(self):
        pass

    def generate_period_to_extract(self):
        pass

    def _get_regions(self, buoys_metadata:pd.DataFrame):
        return (buoys_metadata['archive_path']
                                    .str.extract(r'\\auswaves\\([a-z]+)waves')[0]
                                    .str.upper()
                    )
    
    def _get_sites_per_region(self, buoys_metadata:pd.DataFrame):
        pass 

    def convert_to_datetime(self, data: pd.DataFrame, timestamp_col_name: str="timestamp", parameter_type="waves") -> pd.DataFrame:
        
        if parameter_type == "waves":
            time_col = "TIME"
        elif parameter_type == "temp":
            time_col = "TIME_TEMP"
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")

        if data is not None:
            data[time_col] = (pd.to_datetime(data[timestamp_col_name], errors="coerce", utc=True)
                            .dt.tz_localize(None)) # making sure to generate tz naive times following AODN previous data and templates
            data = data.drop(columns=[timestamp_col_name])
            return data
        else:
            return None
        
    def sort_datetimes(self, data: pd.DataFrame) -> pd.DataFrame:
        time_col = [col for col in data.columns if "TIME" in col][0]
        return data.sort_values(time_col)

    def concat_previous_new(self,
                            previous_data: pd.DataFrame,
                            new_data: pd.DataFrame,
                            data_type:str = "waves",
                            drop_duplicates:bool = True) -> pd.DataFrame:
        
        if data_type == "waves":
            cols_to_drop = [col for col in previous_data.columns if "TEMP" in col and col not in ("LONGITUDE", "LATITUDE", "timeSeries")]
        elif data_type == "temp":
            cols_to_drop = [col for col in previous_data.columns if "TEMP" not in col]
        
        if cols_to_drop:
            previous_data = previous_data.drop(columns=cols_to_drop)

        
        if previous_data.empty:
            return new_data

        concat_data = pd.concat([previous_data, new_data], axis=0)

        time_col = [col for col in concat_data.columns if "TIME" in col][0]

        if drop_duplicates:
            return concat_data.drop_duplicates(subset=time_col)
        else:
            return concat_data     
        
    def create_timeseries_aodn_column(self, data: pd.DataFrame) -> pd.DataFrame:
        data["timeSeries"] = float(1)
        return data

    def drop_lat_lon(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=["LATITUDE", "LONGITUDE"])

    def filter_timerange(self, data:pd.DataFrame, waves:pd.DataFrame) -> pd.DataFrame:
        
        time_col_data = [col for col in data.columns if "TIME" in col][0]
        time_col_waves = [col for col in waves.columns if "TIME" in col][0]

        return data.loc[(data[time_col_data] >= waves[time_col_waves].min()) 
                 & (data[time_col_data] <= waves[time_col_waves].max())]
    
    def drop_duplicates(self, data:pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates(subset=["TEMP", "TIME_TEMP"]) 

