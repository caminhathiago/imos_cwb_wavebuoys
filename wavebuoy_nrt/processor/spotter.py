import os
from typing import List
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

from wavebuoy_nrt.config.config import  AODN_COLUMNS_TEMPLATE, AODN_SPECTRAL_COLUMNS_TEMPLATE

GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")

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
                              temp: pd.DataFrame = None,
                              consider_processing_source: bool = True,
                              how: str = "left") -> pd.DataFrame: # wind: pd.DataFrame
        # # IN PROGRESS
        # if wind:
        #     wind = wind.drop(columns=["latitude","longitude", "processing_source"])
        #     all = waves.merge(wind, on="timestamp")
        merge_condition = ["TIME"]
        if consider_processing_source:
            merge_condition.append("processing_source")
        
        if temp is not None:
            if not temp.empty:
                temp = temp.drop(columns=["latitude","longitude"])
                waves = waves.merge(temp, on=merge_condition, how=how)

        waves = waves.sort_values(merge_condition)

        return waves    

    def conform_columns_names_aodn(self, data: pd.DataFrame, parameters_type: str = "bulk") -> pd.DataFrame:
        if parameters_type == "bulk":
            columns_template = AODN_COLUMNS_TEMPLATE
        elif parameters_type == "spectral":
            columns_template = AODN_SPECTRAL_COLUMNS_TEMPLATE

        rename_dict = {k: v for k, v in columns_template.items() if v is not None}
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

    def get_temp_from_smart_mooring(self, 
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
        data = data.rename(columns={"value":"TEMP"})
        return data
    
    def round_parameter_values(self, data: pd.DataFrame, parameter: str, decimals: int = 2) -> pd.DataFrame:
        data[parameter] = data[parameter].round(decimals)
        return data
    
    def test_duplicated(self, data: pd.DataFrame) -> bool:
        return any(data.duplicated().values)

    def select_processing_source(self, data: pd.DataFrame, processing_source: str="hdr") -> pd.DataFrame:
        return data[data["processing_source"] == processing_source].copy()

    def select_priority_processing_source(self, data: pd.DataFrame, priority_source: str="hdr") -> pd.DataFrame:
        available_sources = data["processing_source"].unique()
        if priority_source in available_sources:
            return data[data["processing_source"] == priority_source]
        else:
            index = np.argwhere(available_sources!=priority_source)
            return data[data["processing_source"] == str(available_sources[index].squeeze())]
