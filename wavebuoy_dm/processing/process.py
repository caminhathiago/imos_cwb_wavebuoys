from typing import List, Dict
import logging
from datetime import datetime, timedelta
import os 

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import polars as pl
import pandas as pd
import numpy as np
from pyproj import Transformer
from geopy.distance import geodesic

from wavebuoy_dm.config.config import AODN_COLUMNS_TEMPLATE

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(message)s')


class FLTProcess:
    def __init__(self):
        pass

class HDRProcess:
    def __init__(self):
        pass

class LOCProcess:
    def __init__(self):
        pass

    def process_lat_lon(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        df_lazy = df_lazy.with_columns([
                    (pl.col("lat") + (pl.col("lat_min") / 10**5) / 60).alias("latitude"),
                    (pl.col("lon") + (pl.col("lon_min") / 10**5) / 60).alias("longitude")
                ])
        
        return df_lazy.drop(['lat','lon','lat_min','lon_min'])
    
    def filter_bad_lat_lon(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        idx = (
                (pl.col("latitude") > 90) | 
                (pl.col("latitude") < -90) | 
                (pl.col("longitude") > 180) |
                (pl.col("longitude") < -180)
        )
        return df_lazy.filter(~idx)

class SMDProcess:
    def __init__(self):
        pass

    

class csvProcess:
    def __init__(self):
        self.flt_process = FLTProcess()
        self.hdr_process = HDRProcess()
        self.loc_process = LOCProcess()

        self.column_map = {'FLT': {'outx(mm)': 'x', 'outy(mm)': 'y', 'outz(mm)': 'z'},
                            'LOC': {'lat(deg)':'lat', 'lat(min*1e5)':'lat_min', 'long(deg)':'lon', 'long(min*1e5)':'lon_min'},
                            'SST': {'temperature (C)':'temperature'},
                            'HDR': {'dmx(mm)':'x', 'dmy(mm)':'y', 'dmz(mm)':'z', 'dmn(mm)':'n'},
                            'BARO': {'pressure (mbar)':'baro_pressure'},
                            'SPC': {},
                            'SYS': {},
                        }

        self.suffix_name_map = {"FLT":"displacements",
                            "HDR":"displacements_hdr",
                            "BARO": "barometer",
                            "SST": "surface_temp",
                            "LOC": "gps"}

    def convert_to_datetime(self, 
                            df_lazy: pl.LazyFrame,
                            suffix: str,
                            drop_original_column: bool = True) -> pl.LazyFrame:
        """
        Add a 'datetime' column by converting 'GPS_Epoch_Time(s)' from seconds to nanoseconds.

        Args:
            df_lazy (pl.LazyFrame): The input lazy dataframe.

        Returns:
            pl.LazyFrame: The dataframe with an additional 'datetime' column.
        """

        time_col = self.get_time_column(df_lazy, suffix=suffix)

        # # Pre filter garbage data e.g. 1.7392e20, instead of around 1.7e9
        
        # valid_range = (0, 9.22e9)
        # df_lazy = df_lazy.filter(
        #     (pl.col(time_col) >= valid_range[0]) &
        #     (pl.col(time_col) <= valid_range[1])
        # )

        # df_lazy = df_lazy.with_columns(
        #         pl.from_epoch((pl.col(time_col) * 1e9)
        #         .cast(pl.Int64), time_unit="ns")
        #         .alias("datetime")
        #     )
        
        valid_range = (0, 4e9) # realistic GPS seconds (1980–2107)
        df_lazy = df_lazy.filter(
            (pl.col(time_col) >= valid_range[0]) &
            (pl.col(time_col) <= valid_range[1])   
        )

        df_lazy = df_lazy.with_columns(
            pl.from_epoch(pl.col(time_col), time_unit="s").alias("datetime")
        )

        # df_lazy = df_lazy.filter(
        #     (pl.col('datetime') > datetime(2010,1,1,0,0,0)) &
        #      (pl.col('datetime') < datetime.now())
        #      )

        if drop_original_column:
            return self.drop_column(df_lazy, column_name=time_col)
        else:
            return df_lazy 
        
    def sort_by_datetime(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        return df_lazy.sort("datetime")
    
    def rename_columns(self, df_lazy: pl.LazyFrame, column_map: Dict[str, str]) -> pl.LazyFrame:
        """
        Rename columns in the lazy dataframe based on a provided mapping.

        Args:
            df_lazy (pl.LazyFrame): The input lazy dataframe.
            column_map (Dict[str, str]): A dictionary mapping old column names to new column names.

        Returns:
            pl.LazyFrame: The dataframe with renamed columns.
        """
        return df_lazy.rename(column_map)
    
    def convert_displacement_to_meters(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        operations = [(pl.col('x') / 1000),
            (pl.col('y') / 1000),
            (pl.col('z') / 1000)]
        
        if 'n' in df_lazy.collect_schema():
            operations.append((pl.col('n') / 1000))
        
        return df_lazy.with_columns(operations)

    def drop_nat(sef, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        return df_lazy.filter(pl.col('datetime').is_not_null())

    def drop_column(self, df_lazy: pl.LazyFrame, column_name: str) -> pl.LazyFrame:
        """
        Drop a specified column from the lazy dataframe.

        Args:
            df_lazy (pl.LazyFrame): The input lazy dataframe.
            column_name (str): The name of the column to drop.

        Returns:
            pl.LazyFrame: The dataframe with the specified column removed.
        """
        return df_lazy.drop(column_name)

    def get_time_column(self, df_lazy: pl.LazyFrame, suffix: str) -> str:
        time_cols = [time_col for time_col in df_lazy.collect_schema().keys() if "time" in time_col.lower()]
        
        if len(time_cols) > 2 and suffix == "SPC":
            return time_cols[0]

        return time_cols[0]
    
    def process_concat_results(self, lazy_concat_results):
        results = lazy_concat_results.copy()
        for suffix in results:
            if isinstance(lazy_concat_results[suffix], pl.LazyFrame):
                df_lazy = lazy_concat_results[suffix] 
                df_lazy = self.convert_to_datetime(df_lazy, suffix)
                df_lazy = self.sort_by_datetime(df_lazy)
                df_lazy = self.rename_columns(df_lazy, column_map=self.column_map[suffix])
                
                if suffix == "FLT":
                    df_lazy = self.convert_displacement_to_meters(df_lazy)
                elif suffix == "HDR":
                    df_lazy = self.convert_displacement_to_meters(df_lazy)
                elif suffix == "LOC":
                    df_lazy = self.loc_process.process_lat_lon(df_lazy)
                    df_lazy = self.loc_process.filter_bad_lat_lon(df_lazy)
            
                if not suffix in ('SMD', 'SENS_AGG'):
                    df_lazy = self.drop_nat(df_lazy)

                results.update({suffix:df_lazy})
        return results

    def collect_results(self, results: dict) -> dict:
        collected_results = results.copy()
        for suffix in results:
            if isinstance(results[suffix], pl.LazyFrame):
                collected_result = results[suffix].collect()
                collected_results.update({suffix: collected_result})
        
        renamed_results = {
            self.suffix_name_map.get(suffix, suffix): value
            for suffix, value in collected_results.items()
        }

        return renamed_results

    def filter_absurd_datetimes(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
                (pl.col("datetime") >= datetime(2000,1,1,0,0)) & 
                (pl.col("datetime") >= datetime(2024,7,1,2,29,59,999999))
            )

    def interpolate_lat_lon(self, dataframe: pl.DataFrame, locations_dataframe: pl.DataFrame) -> pl.Series:
        
        data_time_numeric = dataframe["TIME"].to_numpy().astype("datetime64[s]").view(np.int64)
        gps_time_numeric = locations_dataframe["datetime"].to_numpy().astype("datetime64[s]").view(np.int64)

        interpolated_lon = np.interp(data_time_numeric, gps_time_numeric, locations_dataframe["longitude"].to_numpy())
        interpolated_lat = np.interp(data_time_numeric, gps_time_numeric, locations_dataframe["latitude"].to_numpy())
        
        return dataframe.with_columns(
                        pl.Series("LONGITUDE", interpolated_lon),
                        pl.Series("LATITUDE", interpolated_lat)
                        )
    
    

    def filter_deployment_dates(self, 
                                dataframe:pl.DataFrame,
                                timezone:str,
                                deploy_start:datetime, 
                                deploy_end:datetime,
                                time_crop_start:int = 18,
                                time_crop_end:int = 6) -> pl.DataFrame:
        
    #     time_col = [col for col in dataframe.columns if "TIME" in col.upper()]
    #     time_col_local = time_col.copy()
    #     time_col_local[0] += "_local"

    #     dataframe = dataframe.with_columns(
    #     (pl.col(time_col) + timedelta(hours=utc_offset)).alias(time_col_local[0])
    #     )

    #     filter_start_time = deploy_start + timedelta(hours=time_crop_start)
    #     filter_end_time = deploy_end + timedelta(hours=time_crop_end)

    #     dataframe = dataframe.filter(
    #             (pl.col(time_col_local) >= filter_start_time) & 
    #             (pl.col(time_col_local) <= filter_end_time)
    #         )

    #     return dataframe.drop(time_col_local)
    # find TIME column (first match)
        time_cols = [c for c in dataframe.columns if "TIME" in c.upper()]
        if not time_cols:
            return dataframe  # nothing to do

        time_col = time_cols[0]
        time_col_local = time_col + "_local"

        dataframe = dataframe.with_columns(
            (
                pl.col(time_col)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(timezone)
            ).alias(time_col_local)
        )

        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone)

        def _localize(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=tz)
            else:
                return dt.astimezone(tz)

        filter_start_time = _localize(deploy_start + timedelta(hours=time_crop_start))
        filter_end_time = _localize(deploy_end + timedelta(hours=time_crop_end))

        dataframe = dataframe.filter(
            (pl.col(time_col_local) >= filter_start_time) & (pl.col(time_col_local) <= filter_end_time)
        )

        return dataframe.drop(time_col_local)
    
    def buffer_gps_times(self, disp: pl.DataFrame, gps: pl.DataFrame, time_minutes:int=2) -> pl.DataFrame:
        
        time_col = [col for col in disp.columns if "TIME" in col.upper()]
        
        if not time_col:
            raise ValueError("No TIME column found in disp DataFrame.")
        
        time_col = time_col[0]

        start_time = disp[time_col].min()
        end_time = disp[time_col].max()

        buffered_start_time = start_time - timedelta(minutes=time_minutes)

        filtered_gps = gps.filter(
            (pl.col(time_col) >= buffered_start_time) & (pl.col(time_col) <= end_time)
        )

        return filtered_gps


    def get_deployment_latlon(self, deployment_metadata: pd.DataFrame) -> tuple[float]:
        return (deployment_metadata.loc["Latitude_nominal", "metadata_wave_buoy"],
                deployment_metadata.loc["Longitude_nominal", "metadata_wave_buoy"]
        )

    def filter_watch_circle_utm(self, 
                            dataframe:pl.DataFrame,
                            deploy_lat: float,
                            deploy_lon: float,
                            max_distance: float = 100) -> pl.DataFrame:
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)
        
        deploy_easting, deploy_northing = transformer.transform(deploy_lon, deploy_lat)

        def to_utm(lat, lon):
            easting, northing = transformer.transform(lon, lat)
            return [easting, northing]

        dataframe = dataframe.with_columns([
            pl.struct(["LONGITUDE", "LATITUDE"]).map_elements(
                lambda row: transformer.transform(row["LONGITUDE"], row["LATITUDE"])[0],
                return_dtype=pl.Float64
            ).alias("EASTING"),
            pl.struct(["LONGITUDE", "LATITUDE"]).map_elements(
                lambda row: transformer.transform(row["LONGITUDE"], row["LATITUDE"])[1],
                return_dtype=pl.Float64
            ).alias("NORTHING"),
        ])

        dataframe = dataframe.with_columns([
                                (
                                    ((pl.col("EASTING") - deploy_easting) ** 2 + (pl.col("NORTHING") - deploy_northing) ** 2)
                                    .sqrt()
                                    .alias("DIST_TO_DEPLOY")
                                )
                            ])
        
        return (dataframe
                    .filter(pl.col("DIST_TO_DEPLOY") <= max_distance)
                    .drop(["EASTING", "NORTHING", "DIST_TO_DEPLOY"])
        )

    def calculate_watch_circle(self, site_buoys_to_process:pd.DataFrame, reprocess:bool = False):
        
        s = site_buoys_to_process

        mainline = s.mainline_length + (s.mainline_length*s.mooring_stretch_factor)
        catenary = s.catenary_length + (s.catenary_length*s.mooring_stretch_factor)
        
        if reprocess:
            mainline = mainline + s.mainline_length_error
            catenary = catenary + s.catenary_length_error

            mainline *= (1 + s.mooring_stretch_factor)
            catenary *= (1 + s.mooring_stretch_factor)

        watch_circle = np.sqrt(mainline**2 - s.DeployDepth**2) + catenary + s.watch_circle_gps_error
        
        if np.isnan(watch_circle):
            raise ValueError(f"Calculated watchcircle lead to a NaN. Please check if mainline, catenary, deploy_depth and respective error configurations are provided in buoys_to_process.csv")

        return mainline, catenary, watch_circle

    def qc_watch_circle(self,
                        dataframe:pl.DataFrame,
                        deploy_lat:float,
                        deploy_lon:float,
                        watch_circle:float,
                        watch_circle_fail:float):
        
        raw_records = dataframe.shape[0]

        def calc_distance(lat, lon):
            return geodesic((deploy_lat, deploy_lon), (lat, lon)).meters
        
        dataframe = dataframe.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda row: calc_distance(row["LATITUDE"], row["LONGITUDE"]),
                return_dtype=pl.Float64
            ).alias("distance")
        )
    
        qc_flag_watch = "WATCH_quality_control_primary"
        if qc_flag_watch in dataframe.columns:
            qc_flag_watch = "WATCH_quality_control_secondary"

        dataframe = dataframe.with_columns(
            pl.when(pl.col("distance") >= watch_circle * watch_circle_fail).then(4)
            .when(pl.col("distance") >= watch_circle).then(3)
            .otherwise(1) 
            .alias(qc_flag_watch)
        )

        flag_counts = dataframe.select(pl.col(qc_flag_watch)).to_series().value_counts()

        pct_flag_3 = (
            flag_counts.filter(pl.col(qc_flag_watch) == 3)["count"][0] / raw_records * 100
            if (flag_counts[qc_flag_watch] == 3).any()
            else 0
        )

        pct_flag_4 = (
            flag_counts.filter(pl.col(qc_flag_watch) == 4)["count"][0] / raw_records * 100
            if (flag_counts[qc_flag_watch] == 4).any()
            else 0
        )

        return dataframe, round(pct_flag_3 + pct_flag_4,2)

    def filter_watch_circle_geodesic(self, 
                            dataframe:pl.DataFrame,
                            deploy_lat:float,
                            deploy_lon:float,
                            max_distance:float = 100,
                            percentage_threshold:float = .97) -> pl.DataFrame:
        
        raw_records = dataframe.shape[0]

        def calc_distance(lat, lon):
            return geodesic((deploy_lat, deploy_lon), (lat, lon)).meters
        
        dataframe = dataframe.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda row: calc_distance(row["LATITUDE"], row["LONGITUDE"]),
                return_dtype=pl.Float64
            ).alias("distance")
        )
        
        filtered_df = dataframe.filter(pl.col("distance") <= max_distance)
        
        filtered_df = filtered_df.drop("distance")
        percentage_cropped = 1 - (filtered_df.shape[0]/raw_records)
        
        if filtered_df.shape[0]/raw_records < percentage_threshold:
            
            mean_lat = dataframe.select(pl.mean("LATITUDE")).item()
            mean_lon = dataframe.select(pl.mean("LONGITUDE")).item()
            
            def calc_distance(lat, lon):
                return geodesic((mean_lat, mean_lon), (lat, lon)).meters

            dataframe = dataframe.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda row: calc_distance(row["LATITUDE"], row["LONGITUDE"]),
                    return_dtype=pl.Float64
                ).alias("distance")
            )
        
            filtered_df = dataframe.filter(pl.col("distance") <= max_distance)
            
            filtered_df = filtered_df.drop("distance")
            percentage_cropped = 1 - (filtered_df.shape[0]/raw_records)
            
            if filtered_df.shape[0]/raw_records < percentage_threshold:
                raise ValueError(f"Filtering watch circle removed more than 30% of records. Please revise DeployLat and DeployLon on buoys_to_process.csv")
        
        return filtered_df, percentage_cropped

    def save_qc_watch_circle_csv(self, data:pl.DataFrame, output_path:str) -> None:
        
        columns = ['TIME', 'LATITUDE', 'LONGITUDE', 'distance']
        watch_qc_cols = [col for col in data.columns if "WATCH_quality_control" in col]
        columns.extend(watch_qc_cols)

        data_file_name = os.path.join(output_path, "spectra_bulk_df_qc_watch.csv")
        data[columns].to_pandas().to_csv(data_file_name, index=False)

        return data_file_name

    def extract_drifting_periods(self, data:pl.DataFrame, output_path:str, reprocess:bool = False)-> None:
        
        if reprocess:
            col_flag = "WATCH_quality_control_secondary"
            flag = 4
        
        else:
            col_flag = "WATCH_quality_control_primary"
            flag = 3

        data = data.with_columns(
            (pl.col(col_flag) == flag).alias("WATCH_fail")
        )

        data = data.with_columns(
            (pl.col("WATCH_fail") != pl.col("WATCH_fail").shift(1)).cum_sum().alias("drifting_periods_id")
        )

        drifting_periods = (
            data.filter(pl.col("WATCH_fail"))
            .group_by("drifting_periods_id")
            .agg([
                pl.col("TIME").min().alias("drifting_start"),
                pl.col("TIME").max().alias("drifting_end")
            ])
            .sort("drifting_start")
        )

        drifting_periods_file_name = os.path.join(output_path, "drifting_periods_qc_watch.csv")
        drifting_periods.to_pandas().to_csv(drifting_periods_file_name, index=False)

    def convert_datatypes(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        dtype_mapping = {
                    "TIME_ds1950": pl.Float64,
                    "FREQUENCY": pl.Float32,
                    "LATITUDE": pl.Float64,
                    "LONGITUDE": pl.Float64,
                    "A1": pl.Float32,
                    "B1": pl.Float32,
                    "A2": pl.Float32,
                    "B2": pl.Float32,
                    "ENERGY": pl.Float32,
                            }
        
        return dataframe.with_columns([
                    dataframe[col].arr.eval(pl.element().cast(dtype.inner)).alias(col) if isinstance(dtype, pl.Array)  
                    else dataframe[col].cast(dtype)  
                    for col, dtype in dtype_mapping.items()
            ])
        
