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

        df_lazy = df_lazy.with_columns(
                pl.from_epoch((pl.col(time_col) * 1e9)
                .cast(pl.Int64), time_unit="ns")
                .alias("datetime")
            )
        
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
            self.suffix_name_map.get(suffix, suffix): value  # Replace if key exists, else keep original
            for suffix, value in collected_results.items()
        }

        return renamed_results

    # def collect_lazyframe_thread(self, suffix: str, lazy_frame: pl.LazyFrame, collected_results: dict):
    #     # Function to collect LazyFrames in a thread
    #     collected_result = lazy_frame.collect()
    #     collected_results.update({suffix: collected_result})
    
    # def collect_results_threading(self, results: dict) -> dict:
    #     collected_results = results.copy()
        
    #     # List to hold all threads
    #     threads = []
        
    #     # Iterate over the results and create threads
    #     for suffix in results:
    #         if isinstance(results[suffix], pl.LazyFrame):
    #             thread = threading.Thread(target=self.collect_lazyframe, args=(suffix, results[suffix], collected_results))
    #             threads.append(thread)
    #             thread.start()  # Start the thread
        
    #     # Wait for all threads to finish
    #     for thread in threads:
    #         thread.join()
        
    #     return collected_results


    # def collect_results_threadpool(self, results: dict) -> dict:
    #     """Uses ThreadPoolExecutor to run LazyFrame collection in parallel."""
    #     collected_results = results.copy()
        
    #     def collect_lazyframe(suffix, lazy_frame):
    #         if isinstance(lazy_frame, pl.LazyFrame):
    #             collected_results[suffix] = lazy_frame.collect()

    #     with ThreadPoolExecutor() as executor:
    #         executor.map(collect_lazyframe, results.keys(), results.values())

    #     return collected_results
    
    # def collect_results_threadpool2(self, results: dict) -> dict:
    #     """Uses ThreadPoolExecutor to run LazyFrame collection in parallel."""
    #     collected_results = {}  # Initialize an empty dictionary to avoid modifying the input

    #     def collect_lazyframe(suffix, lazy_frame):
    #         """Collects a LazyFrame and updates results."""
    #         if isinstance(lazy_frame, pl.LazyFrame):
    #             collected_results[suffix] = lazy_frame.collect()

    #     with ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(collect_lazyframe, suffix, lazy_frame): suffix for suffix, lazy_frame in results.items()}

    #         for future in futures:
    #             future.result()  # Ensures all tasks complete before returning results

    #     return collected_results
    
    # def collect_results_threadpool3(self, results: dict) -> dict:
    #     """Uses ThreadPoolExecutor to run LazyFrame collection in parallel."""
    #     collected_results = {}

    #     def collect_lazyframe(suffix, lazy_frame):
    #         """Collects a LazyFrame and updates results."""
    #         try:
    #             if isinstance(lazy_frame, pl.LazyFrame):
    #                 logging.debug(f"Starting collection for {suffix}")
    #                 collected_results[suffix] = lazy_frame.collect()
    #                 logging.debug(f"Successfully collected LazyFrame for {suffix}")
    #         except Exception as e:
    #             logging.error(f"Error collecting LazyFrame for {suffix}: {e}")
    #             raise  # Reraise the error so that the thread pool can handle it

    #     with ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(collect_lazyframe, suffix, lazy_frame): suffix for suffix, lazy_frame in results.items()}

    #         for future in futures:
    #             try:
    #                 future.result()  # Ensures all tasks complete before returning results
    #             except Exception as e:
    #                 logging.error(f"Error in thread {futures[future]}: {e}")

    #     return collected_results
    
    def collect_results_threadpool4(self, results: dict) -> dict:
        """Uses ThreadPoolExecutor to run LazyFrame collection in parallel."""
        collected_results = {}

        def collect_lazyframe(suffix, lazy_frame):
            """Collects a LazyFrame and updates results."""
            try:
                if isinstance(lazy_frame, pl.LazyFrame):
                    logging.debug(f"Starting collection for {suffix}")
                    collected_results[suffix] = lazy_frame.collect()
                    logging.debug(f"Successfully collected LazyFrame for {suffix}")
            except Exception as e:
                logging.error(f"Error collecting LazyFrame for {suffix}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")  # Log the full traceback for debugging
                raise  # Optionally, re-raise the exception to propagate if needed

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(collect_lazyframe, suffix, lazy_frame): suffix for suffix, lazy_frame in results.items()}

            # Iterate through the futures as they complete
            for future in as_completed(futures):
                suffix = futures[future]  # Get the corresponding suffix for the future
                try:
                    future.result()  # Ensure the task completes and handle any exceptions that occur
                except Exception as e:
                    logging.error(f"Thread for {suffix} encountered an error: {e}")
                    logging.error(f"Traceback for {suffix}: {traceback.format_exc()}")

        renamed_results = {
            self.suffix_name_map.get(suffix, suffix): value  # Replace if key exists, else keep original
            for suffix, value in collected_results.items()
        }

        return renamed_results
    
    # def collect_lazyframe_chunks(lazy_frame, chunk_size):
    #     """Process a LazyFrame in parallel using Dask."""
    #     # Determine the number of chunks
    #     total_rows = lazy_frame.fetch_row_count()  # Fetch row count without collecting
    #     num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    #     # Create Dask tasks for slicing and collecting chunks
    #     tasks = [
    #         dask.delayed(lazy_frame.slice(i * chunk_size, chunk_size).collect)()
    #         for i in range(num_chunks)
    #     ]

    #     # Trigger parallel collection
    #     collected_chunks = dask.compute(*tasks)

    #     # Combine the collected chunks into a single DataFrame
    #     return pl.concat(collected_chunks)

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
                                utc_offset:int,
                                deploy_start:datetime, 
                                deploy_end:datetime,
                                time_crop_start:int = 18,
                                time_crop_end:int = 6) -> pl.DataFrame:
        
        time_col = [col for col in dataframe.columns if "TIME" in col.upper()]
        time_col_local = time_col.copy()
        time_col_local[0] += "_local"

        dataframe = dataframe.with_columns(
        (pl.col(time_col) + timedelta(hours=utc_offset)).alias(time_col_local[0])
        )

        filter_start_time = deploy_start + timedelta(hours=time_crop_start)
        filter_end_time = deploy_end + timedelta(hours=time_crop_end)

        # filter deployment and recovery datetimes
        dataframe = dataframe.filter(
                (pl.col(time_col_local) >= filter_start_time) & 
                (pl.col(time_col_local) <= filter_end_time)
            )

        # filter hours_crop hours from first/last timepoints
        # filter_start_time = dataframe[time_col][0].item() + timedelta(hours=hours_crop_start)
        # filter_end_time = dataframe[time_col][-1].item() - timedelta(hours=hours_crop_end)

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

    # def watch_circle_from_mooring_metadata(self, )

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

        # Add UTM coordinates for LATITUDE and LONGITUDE
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
            .otherwise(1)  # or 0 if you prefer unflagged points to be 0
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

        # filtered_df = dataframe.filter(pl.col("distance") >= watch_circle) # flag as 3
        # filtered_df = dataframe.filter(pl.col("distance") >= watch_circle*watch_circle_fail) # flag as 4
        
        # percentage_cropped = (1 - (filtered_df.shape[0]/raw_records)) * 100

        return dataframe, (pct_flag_3 + pct_flag_4)



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
    # If it's an Array, convert each element to the correct type
    dataframe[col].arr.eval(pl.element().cast(dtype.inner)).alias(col) if isinstance(dtype, pl.Array)  
    # Otherwise, just cast normally
    else dataframe[col].cast(dtype)  
    for col, dtype in dtype_mapping.items()
])
        
