import os
from typing import List
from datetime import datetime, timedelta

import pandas as pd

from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.config.config import FILES_PATH, AODN_COLUMNS_TEMPLATE


class FilesHandler():
    def __init__(self):
        pass

    def _get_file_path(self, file_name):
        return os.path.join(FILES_PATH, file_name)
    
class SpotterWaveBuoy():

    def convert_to_dataframe(self, raw_data: dict, parameters_type: str) -> pd.DataFrame:
        """
        ['spotterId', 'limit', 'frequencyData', 'wind', 'waves', 'surfaceTemp', 'barometerData']
        
        """
        if raw_data[parameters_type]:
            return pd.DataFrame(raw_data[parameters_type])
        else:
            print(f"No data for {parameters_type}.")
            return None

    def merge_parameter_types(self, waves: pd.DataFrame, sst: pd.DataFrame) -> pd.DataFrame: # wind: pd.DataFrame
        # # IN PROGRESS
        # if wind:
        #     wind = wind.drop(columns=["latitude","longitude", "processing_source"])
        #     all = waves.merge(wind, on="timestamp")
        if sst:
            sst = sst.drop(columns=["latitude","longitude", "processing_source"])
            waves = waves.merge(sst, on="timestamp")
        
        return waves     

    def conform_columns_names_aodn(self, data: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {k: v for k, v in AODN_COLUMNS_TEMPLATE.items() if v is not None}
        return data.rename(columns=rename_dict)

    def drop_unwanted_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=["processing_source"])

    def check_parameters_type_available(self, raw_data: dict) -> list:
        parameters_types = raw_data.keys()
        for parameter_type in parameters_types:
            pass

class WaveBuoy(FilesHandler, NetCDFFileHandler, SpotterWaveBuoy): #(CWBAWSs3):
    def __init__(self, buoy_type:str,buoys_metadata_file_name:str="buoys_metadata.csv"):
        self.buoys_metadata = self._get_buoys_metadata(buoy_type=buoy_type, buoys_metadata_file_name=buoys_metadata_file_name)
        self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)
        self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata_token_sorted)
        self.sites_per_region = self._get_sites_per_region(buoys_metadata=self.buoys_metadata_token_sorted)

    def _get_buoys_metadata(self, buoy_type:str,buoys_metadata_file_name:str):
        buoys_metadata_path = self._get_file_path(file_name=buoys_metadata_file_name)
        buoys_metadata = pd.read_csv(buoys_metadata_path)
        buoys_metadata = self._select_buoy_type(buoy_type=buoy_type, buoys_metadata=buoys_metadata)
        buoys_metadata["region"] = self._get_regions(buoys_metadata=buoys_metadata)
        buoys_metadata = buoys_metadata.set_index('name')
        
        return buoys_metadata

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
            data["TIME"] = pd.to_datetime(data[timestamp_col_name], errors="coerce")
            data = data.drop(columns=[timestamp_col_name])
            return data
        else:
            return None
        
        
    
    def sort_datetimes(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sort_values("TIME")

    def concat_previous_new(self, previous_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([previous_data, new_data], axis=0)

