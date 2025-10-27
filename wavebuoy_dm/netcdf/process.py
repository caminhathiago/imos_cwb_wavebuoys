import os
from typing import List, Tuple
from datetime import timedelta

from netCDF4 import date2num
import xarray as xr
import polars as pl
import pandas as pd
from pandas.core.indexes.period import PeriodIndex
import numpy as np

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*DatetimeProperties.to_pydatetime.*",
    category=FutureWarning
)

class Process:

    DTYPES_BULK = {'WSSH':{"dtype":np.float64},
                        'WPPE':{"dtype":np.float32},
                        'WPFM':{"dtype":np.float32},
                        'WPDI':{"dtype":np.float32},
                        'WPDS':{"dtype":np.float32},
                        'SSWMD':{"dtype":np.float32},
                        'WMDS':{"dtype":np.float32},
                        'TEMP':{"dtype":np.float32},
                        'WAVE_quality_control':{"dtype":np.int8},
                        'TEMP_quality_control':{"dtype":np.int8},
                        'WATCH_CIRCLE_flag':{"dtype":np.int8},
                        # 'TIME':{"dtype":np.float64},
                        'LATITUDE':{"dtype":np.float64},
                        'LONGITUDE':{"dtype":np.float64},
                        'timeSeries':{"dtype":np.int16}}
                        
    
    DTYPES_SPECTRAL = {
                "FREQUENCY": {"dtype": np.float32},
                "LATITUDE": {"dtype": np.float64},
                "LONGITUDE": {"dtype": np.float64},
                "A1": {"dtype": np.float32},
                "B1": {"dtype": np.float32},
                "A2": {"dtype": np.float32},
                "B2": {"dtype": np.float32},
                "ENERGY": {"dtype": np.float32},
                'timeSeries':{"dtype":np.int16},
                'WATCH_CIRCLE_flag':{"dtype":np.int8},
            }

    def __init__(self):
        pass

    def convert_time_to_CF_convention(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        unix_epoch_19500101 = -631152000 # unix epoch equivalent of 1950-01-01T00:00:00
        secs_in_day = 24*60*60

        col_time = self.extract_time_column_name(dataframe=dataframe)

        return dataframe.with_columns(
                    ((pl.col(col_time).dt.epoch(time_unit="s") - unix_epoch_19500101) / secs_in_day)
                    .alias("TIME_ds1950")
                    )

    def extract_time_column_name(self, dataframe: pl.DataFrame):
        col_time = [col_time for col_time in dataframe.columns if "time" in col_time.lower()]
        if col_time:
            return col_time[0]
        else:
            raise Exception("time column not found.")
        
    def convert_time_to_CF_convention(self, dataset: tuple) -> xr.Dataset:
        time = np.array(dataset["TIME"]
                            .to_dataframe()["TIME"]
                            .dt.to_pydatetime()
            )
        dataset["TIME"] = date2num(np.array(time),
                                "days since 1950-01-01 00:00:00 UTC",
                                "gregorian")
        
        time_second_dim = [col for col in list(dataset.variables) if col in ("TIME_TEMP","TIME_LOCATION")]
        if time_second_dim:
            time = np.array(dataset[time_second_dim[0]]
                            .to_dataframe()[time_second_dim[0]]
                            .dt.to_pydatetime()
                    )
            dataset[time_second_dim[0]] = date2num(np.array(time),
                                "days since 1950-01-01 00:00:00 UTC",
                                "gregorian")
            
        return dataset

    def convert_time_to_CF_convention_ds_list(self, dataset_objects: tuple) -> List[xr.Dataset]:
        for dataset in dataset_objects:
            dataset = self.convert_time_to_CF_convention(dataset=dataset)
        return dataset_objects
    
    @staticmethod
    def create_timeseries_variable(dataset: xr.Dataset) -> xr.Dataset:
        dataset["timeSeries"] = [np.int16(1)]
        return dataset

    
    @staticmethod
    def convert_dtypes(dataset:tuple, parameters_type:str = "bulk") -> xr.Dataset:
        
        if parameters_type == 'bulk':
            dtypes_dict = Process.DTYPES_BULK
        elif parameters_type == 'spectral':
            dtypes_dict = Process.DTYPES_SPECTRAL
        else:
            raise ValueError("Invalid parameters_type. Choose 'bulk' or 'spectral'.")

        dataset = dataset.copy()

        for var_name, target_dtype in dtypes_dict.items():
            if var_name in dataset:
                dataset[var_name] = dataset[var_name].astype(target_dtype["dtype"])

        return dataset
    
    @staticmethod
    def convert_dtypes_dataset_objects(dataset_objects: tuple, parameters_type:str = "bulk") -> List[xr.Dataset]:
        
        if parameters_type == 'bulk':
            dtypes_dict = Process.DTYPES_BULK
        elif parameters_type == 'spectral':
            dtypes_dict = Process.DTYPES_SPECTRAL

        for dataset in dataset_objects:
            dataset = dataset.map(lambda x: x.astype(dtypes_dict[x.name]["dtype"]) if x.name in dtypes_dict else x )

        return dataset_objects

class ncSpectra(Process):
    def __init__(self):
        super().__init__()

    def compose_dataset(self, global_result: pl.DataFrame) -> xr.Dataset:
        return xr.Dataset(
            {
                "ENERGY": (["TIME", "FREQUENCY"], global_result['ENERGY'].to_numpy()),
                "A1": (["TIME", "FREQUENCY"], global_result["A1"].to_numpy()),
                "B1": (["TIME", "FREQUENCY"], global_result["B1"].to_numpy()),
                "A2": (["TIME", "FREQUENCY"], global_result["A2"].to_numpy()),
                "B2": (["TIME", "FREQUENCY"], global_result["B2"].to_numpy()),
                "WATCH_CIRCLE_flag": ("TIME", global_result["WATCH_CIRCLE_flag"].to_numpy())
            },
            coords={
                "TIME": global_result["TIME"].to_numpy(),
                "FREQUENCY": global_result["FREQUENCY"].to_numpy()[0],
                "LATITUDE": ("TIME", global_result["LATITUDE"].to_numpy()),
                "LONGITUDE": ("TIME", global_result["LONGITUDE"].to_numpy())
            }
        )
    
class ncDisp(Process):
    def __init__(self):
        super().__init__()


    def compose_dataset(self, data: pl.DataFrame, data_gps: pl.DataFrame) -> xr.Dataset:
        return xr.Dataset(
            {
                "LATITUDE": ("TIME_LOCATION", data_gps["latitude"].to_numpy()),
                "LONGITUDE": ("TIME_LOCATION", data_gps["longitude"].to_numpy()),
                "XDIS": ("TIME", data["x"].to_numpy()),
                "YDIS": ("TIME", data["y"].to_numpy()),
                "ZDIS": ("TIME", data["z"].to_numpy())
            },
            coords={
                "TIME": data["datetime"].to_numpy(),
                "TIME_LOCATION": data_gps["datetime"].to_numpy(),
            }
        )

    def generate_time_ranges(self, dataframe: pl.DataFrame, chunk_period: str = "14d") -> pl.DataFrame:
        col_time = self.extract_time_column_name(dataframe)
        return (dataframe.group_by_dynamic(col_time, every=chunk_period, start_by="datapoint")
                .agg([pl.col(col_time).first().alias("start_datetime"),
                     pl.col(col_time).last().alias("end_datetime")])
        )

    def compose_datasets(self, data: pl.DataFrame, data_gps: pl.DataFrame, chunk_period: str = "14d"):
        time_ranges = self.generate_time_ranges(dataframe=data, chunk_period=chunk_period)
        col_time_gps = self.extract_time_column_name(data_gps)
        
        datasets = []
        for row in time_ranges.iter_rows(named=True):
            data_chunk = data.filter(
                        (data[col_time_gps] >= row['start_datetime']) & 
                        (data[col_time_gps] < row['end_datetime']))
            
            gps_chunk = data_gps.filter(
                        (data_gps[col_time_gps] >= row['start_datetime']) & 
                        (data_gps[col_time_gps] < row['end_datetime']))

            dataset = self.compose_dataset(data=data_chunk, data_gps=gps_chunk)
            
            datasets.append(dataset)

        return tuple(datasets)

    
    def extract_fortnightly_periods_dataset(self, dataset: xr.Dataset) -> pd.PeriodIndex:

        times = pd.to_datetime(dataset["TIME"].data)  
        naive_times = times.tz_localize(None) 

        start_date = naive_times.min().floor('D')  
        end_date = (naive_times.max() + pd.Timedelta(days=1)).normalize()
        periods = pd.date_range(start=start_date, end=end_date, freq='14D')

        unique = periods.to_period('D').unique()

        if str(unique[-1]) != str(naive_times.max().date()):
            unique = unique.append(pd.PeriodIndex([str(naive_times.max().date())], freq="D"))

        fortnight_periods = []
        for i, period in enumerate(unique):
            if period != unique[-1]:
                start = period
                end = unique[i + 1] - pd.Timedelta(days=1) if period != unique[-2] else unique[i + 1]
                fortnight_periods.append((start, end))
                
        for pair in fortnight_periods:
            print((str(pair[0]), str(pair[1])))
        
        return fortnight_periods

    def split_dataset_fortnightly(self, dataset: xr.Dataset, periods: PeriodIndex) -> Tuple[xr.Dataset, ...]:
        dataset_objects = []
        for period in periods:
            print(period)
            start = str(period[0])
            end = str(period[1]) 
            fortnightly_dataset = dataset.sel(TIME=slice(start, end), TIME_LOCATION=slice(start, end))
            
            if fortnightly_dataset.TIME.size == 0:
               periods.remove(period)
               continue

            dataset_objects.append(fortnightly_dataset)
        
        return tuple(dataset_objects), periods
    
class ncBulk(Process):
    def __init__(self):
        super().__init__()

    def _compose_coords_dimensions(self, waves: pd.DataFrame, temp: pd.DataFrame = None, parameters_type: str = "bulk") -> dict:

        if parameters_type == "bulk":
            coords = {
                    "TIME":("TIME", waves["TIME"]),
                    "LATITUDE":("TIME", waves["LATITUDE"]),
                    "LONGITUDE":("TIME", waves["LONGITUDE"])
                }
            if temp is not None:
                coords.update({"TIME_TEMP": ("TIME_TEMP", temp["TIME_TEMP"])})

        if parameters_type == "spectral":
            coords.update({"FREQUENCY": ("FREQUENCY", waves["FREQUENCY"].iloc[0])})

        return coords

    def _compose_data_vars(self, waves: pd.DataFrame, temp: pd.DataFrame = None, parameters_type: str = "bulk") -> dict:
      
        data_vars = {}
        cols_to_drop = ['TIME', 'LATITUDE', 'LONGITUDE']
        dimensions = ["TIME"]

        if parameters_type == "spectral":
            cols_to_drop.extend(["FREQUENCY", "DIFFREQUENCY", "DIRECTION", "DIRSPREAD"])
            dimensions.append("FREQUENCY")

        vars_to_include = waves.drop(columns=cols_to_drop).columns.to_list()
        if temp is not None:
            vars_to_include.extend(["TEMP","TEMP_quality_control"])


        for var in vars_to_include:
            
            if parameters_type == "bulk":
                if var in ("TEMP","TEMP_quality_control"):
                    if var in temp:
                        data_vars.update({var:(("TIME_TEMP"), temp[var])})
                
                else:
                    data_vars.update({var:(tuple(dimensions), waves[var])})

            elif parameters_type == "spectral":
                data_vars.update({var:(tuple(dimensions), np.vstack(waves[var].values))})
        
        return data_vars

    def compose_dataset(self, waves:pd.DataFrame, temp:pd.DataFrame = None, parameters_type:str = "bulk") -> xr.Dataset:
        
        coords = self._compose_coords_dimensions(waves=waves, temp=temp, parameters_type=parameters_type)
        data_vars = self._compose_data_vars(waves=waves, temp=temp, parameters_type=parameters_type)
        
        return xr.Dataset(coords=coords, data_vars=data_vars)


