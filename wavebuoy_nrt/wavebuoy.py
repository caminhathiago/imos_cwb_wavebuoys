import os
from typing import List
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.config.config import FILES_PATH, AODN_COLUMNS_TEMPLATE
from wavebuoy_nrt.utils import args, IMOSLogging

GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")


class FilesHandler():
    def __init__(self):
        pass

    def _get_file_path(self, file_name):
        if os.path.exists(os.path.join(FILES_PATH, file_name)):
            return os.path.join(FILES_PATH, file_name)
        else:
            error_message = f"""Path for {file_name} does not exist.\nCheck if the correct path was provided, or if {file_name} was moved."""
            print(error_message)
            GENERAL_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)

class SpotterWaveBuoy():

    def convert_wave_data_to_dataframe(self, raw_data: dict, parameters_type: str) -> pd.DataFrame:
        """
        ['spotterId', 'limit', 'frequencyData', 'wind', 'waves', 'surfaceTemp', 'barometerData']
        
        """
        if raw_data[parameters_type]:
            return pd.DataFrame(raw_data[parameters_type])
        else:
            print(f"No data for {parameters_type}.")
            return None

    def split_processing_source(self, raw_data: dict, processing_sources: list = ["hdr","embedded"]) -> dict:
        
        split_raw_data = {}
        for source in processing_sources:
            selection = [data_point for data_point in raw_data["waves"] if data_point["processing_source"] == source]
            split_raw_data.update({source:selection})
        
        return split_raw_data

    def convert_smart_mooring_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        return pd.DataFrame(raw_data)

    def merge_parameter_types(self,
                              waves: pd.DataFrame, 
                              sst: pd.DataFrame = None,
                              consider_processing_source: bool = True) -> pd.DataFrame: # wind: pd.DataFrame
        # # IN PROGRESS
        # if wind:
        #     wind = wind.drop(columns=["latitude","longitude", "processing_source"])
        #     all = waves.merge(wind, on="timestamp")
        merge_condition = ["TIME"]
        if consider_processing_source:
            merge_condition.append("processing_source")
        
        if sst is not None:
            if not sst.empty:
                sst = sst.drop(columns=["latitude","longitude"])
                waves = waves.merge(sst, on=merge_condition, how='outer')

        waves = waves.sort_values(merge_condition)

        return waves    

    def conform_columns_names_aodn(self, data: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {k: v for k, v in AODN_COLUMNS_TEMPLATE.items() if v is not None}
        return data.rename(columns=rename_dict)

    def drop_unwanted_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            return data.drop(columns=["processing_source"])
        except:
            return data

    def check_parameters_type_available(self, raw_data: dict) -> list:
        parameters_types = raw_data.keys()
        for parameter_type in parameters_types:
            pass

    def get_sst_from_smart_mooring(self, 
                                   data : pd.DataFrame,
                                   sensor_type : str = "temperature") -> pd.DataFrame:
        
        for data_type in data["data_type_name"].unique():
            if sensor_type in data_type:
                position = data[data["data_type_name"] == data_type]["sensorPosition"].unique()
            else:
                print(f"No {sensor_type} present in this smart_mooring.")
                return None

        if len(position) > 1: # select surface sensor based on surface position (i.e. the lowest position)
            position = position.min()
        else:
            position = position[0]
                
        return data[data["sensorPosition"] == position]

    def process_smart_mooring_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ["sensorPosition","units","unit_type","data_type_name"]
        data = data.drop(columns=cols_to_drop)
        data = data.rename(columns={"value":"SST"})
        return data
    
    def round_parameter_values(self, data: pd.DataFrame, parameter: str, decimals: int = 2) -> pd.DataFrame:
        data[parameter] = data[parameter].round(decimals)
        return data
    
    def test_duplicated(self, data: pd.DataFrame) -> bool:
        return any(data.duplicated().values)
    
    def generate_pickle_file(self, data, file_name, site_name):
        import pickle
        with open(f"output_path/test_files/{site_name}_{file_name}.pkl", "wb") as pickle_file:
            pickle.dump(data, pickle_file)
            print(f"saved pkl as output_path/test_files/{site_name}_{file_name}.pkl")

    def select_processing_source(self, data: pd.DataFrame, processing_source: str="hdr") -> pd.DataFrame:
        return data[data["processing_source"] == processing_source]

    def select_priority_processing_source(self, data: pd.DataFrame, priority_source: str="hdr") -> pd.DataFrame:
        available_sources = data["processing_source"].unique()
        if priority_source in available_sources:
            return data[data["processing_source"] == priority_source]
        else:
            index = np.argwhere(available_sources!=priority_source)
            return data[data["processing_source"] == str(available_sources[index].squeeze())]


class WaveBuoy(FilesHandler, NetCDFFileHandler, SpotterWaveBuoy): #(CWBAWSs3):
    def __init__(self, buoy_type:str, buoys_metadata_file_name:str="buoys_metadata.csv"):
        self.buoys_metadata = self._get_buoys_metadata(buoy_type=buoy_type, buoys_metadata_file_name=buoys_metadata_file_name)
        # self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)
        # self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata)
        # self.sites_per_region = self._get_sites_per_region(buoys_metadata=self.buoys_metadata)

    def _get_buoys_metadata(self, buoy_type:str,buoys_metadata_file_name:str):
        try:
            buoys_metadata_path = self._get_file_path(file_name=buoys_metadata_file_name)
            buoys_metadata = pd.read_csv(buoys_metadata_path)
            buoys_metadata = self._select_buoy_type(buoy_type=buoy_type, buoys_metadata=buoys_metadata)
            buoys_metadata["region"] = self._get_regions(buoys_metadata=buoys_metadata)
            buoys_metadata = buoys_metadata.set_index('name')
            GENERAL_LOGGER.info("Buoys metadata grabbed successfully")
            return buoys_metadata

        except:
            error_message = "Loading and processing buoys_metadata.csv unsuccessful. Check if the file is corrupted or if its structure has been changed"
            GENERAL_LOGGER.error(error_message, exc_info=True)
        
    def _select_buoy_type(self, buoy_type:str, buoys_metadata:pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.loc[buoys_metadata["type"] == buoy_type]

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

    def convert_to_datetime(self, data: pd.DataFrame, timestamp_col_name: str="timestamp") -> pd.DataFrame:
        if data is not None:
            data["TIME"] = (pd.to_datetime(data[timestamp_col_name], errors="coerce", utc=True)
                            .dt.tz_localize(None)) # making sure to generate tz naive times following AODN previous data and templates
            data = data.drop(columns=[timestamp_col_name])
            return data
        else:
            return None
        
    def sort_datetimes(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sort_values("TIME")

    def concat_previous_new(self, previous_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([previous_data, new_data], axis=0)
    
    def create_timeseries_aodn_column(self, data: pd.DataFrame) -> pd.DataFrame:
        data["timeSeries"] = float(1)
        return data

