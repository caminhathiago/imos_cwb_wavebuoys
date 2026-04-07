from datetime import datetime

import pandas as pd
import numpy as np

class ProcAusWaves:


    """
    
       
       'startFrequency_sea', 'endFrequency_sea', ,
       , , 
    
    """

    # order is important
    # key : (auswaves col name, is stripped from space preffix)
    COLS_MAPPING = {
        'time_unix': ('Time (UNIX/UTC)', True),
        'timestamp': ('Timestamp (UTC)', False),
        'site': ('Site', False),
        'spot_id': ('BuoyID', False),

        # Waves
        'significantWaveHeight': ('Hsig (m)', False),
        'significantWaveHeight_swell': ('Hsig_swell (m)', False),
        'significantWaveHeight_sea': ('Hsig_sea (m)', False),
        'peakPeriod': ('Tp (s)', False),
        'meanPeriod': ('Tm (s)', True),
        'meanPeriod_swell': ('Tm_swell (s)', False),
        'meanPeriod_sea': ('Tm_sea (s)', False),
        'peakDirection': ('Dp (deg)', False),
        'peakDirectionalSpread': ('DpSpr (deg)', False),
        'meanDirection': ('Dm (deg)', False),
        'meanDirection_swell': ('Dm_swell (deg)', False),
        'meanDirection_sea': ('Dm_sea (deg)', False),
        'meanDirectionalSpread': ('DmSpr (deg)', False),
        'meanDirectionalSpread_swell': ('DmSpr_swell (deg)', False),
        'meanDirectionalSpread_sea': ('DmSpr_sea (deg)', False),
        'WAVE_quality_control': ('QF_waves', True),

        # SST
        'degrees': ('SST (degC)', False),
        'TEMP_quality_control_sst': ('QF_sst', False),

        'bottom_temp': ('Bottom Temp (degC)', False),
        'TEMP_quality_control_bottom': ('QF_bott_temp', False),

        # Winds
        'speed': ('WindSpeed (m/s)', False),
        'direction': ('WindDirec (deg)', False),

        # Coords
        'latitude': ('Latitude (deg)', False),
        'longitude': ('Longitude (deg)', False),
    }

    SM_COLS_MAPPING = {
        # RBR Temperature
        ("bm_rbr_coda_temperature_mean_20bits", 1): "SST (degC)",
        ("bm_rbr_coda_temperature_mean_20bits", 2): "Middle Temp (degC)",
        ("bm_rbr_coda_temperature_mean_20bits", 3): "Bottom Temp (degC)",
        # RBR Pressure
        ("bm_rbr_coda_pressure_mean_21bits", 1): "Surface Pressure (hPa)",
        ("bm_rbr_coda_pressure_mean_21bits", 2): "Middle Pressure (hPa)",
        ("bm_rbr_coda_pressure_mean_21bits", 3): "Bottom Pressure (hPa)",
        # Current meter
        ("aanderaa_abs_speed_mean_15bits", 1): "Surface CurrmentMag (m/s)",
        ("aanderaa_abs_speed_mean_15bits", 2): "CurrmentMag (m/s)",
        ("aanderaa_abs_speed_mean_15bits", 3): "Bottom CurrmentMag (m/s)",
        ("aanderaa_direction_circ_mean_13bits", 1): "Surface CurrentDir (m/s)",
        ("aanderaa_direction_circ_mean_13bits", 2): "CurrentDir (m/s)",
        ("aanderaa_direction_circ_mean_13bits", 3): "Bottom CurrentDir (m/s)",
        ("aanderaa_abs_tilt_mean_8bits", 1): "Surface CurrentMeterTilt (m/s)",
        ("aanderaa_abs_tilt_mean_8bits", 2): "CurrentMeterTilt (m/s)",
        ("aanderaa_abs_tilt_mean_8bits", 3): "Bottom CurrentMeterTilt (m/s)"
    }

    COLS_ORDER = [
        'Time (UNIX/UTC)', 'Timestamp (UTC)', 'Site', 'BuoyID',
        'Hsig (m)', 'Hsig_swell (m)', 'Hsig_sea (m)',
        'Tp (s)', 'Tm (s)', 'Tm_swell (s)', 'Tm_sea (s)',
        'Dp (deg)', 'DpSpr (deg)', 'Dm (deg)',
        'Dm_swell (deg)', 'Dm_sea (deg)',
        'DmSpr (deg)', 'DmSpr_swell (deg)', 'DmSpr_sea (deg)',
        'QF_waves',
        'SST (degC)', 'QF_sst',
        'Bottom Temp (degC)', 'QF_bott_temp',
        'WindSpeed (m/s)', 'WindDirec (deg)',
        'CurrmentMag (m/s)','CurrentDir (deg)',
        'Latitude (deg)', 'Longitude (deg)'
    ]   

    QC_MAP = {
        "QF_waves":"WAVE_quality_control",
        "QF_sst":"TEMP_quality_control",
        "QF_bott_temp": "TEMP_bottom_quality_control",
        'Hsig (m)':"WSSH",
        'Tp (s)':'WPPE',
        'Dp (deg)':"WPDI",
        'SST (degC)':'TEMP',
        'Bottom Temp (degC)':'TEMP_bottom',        
    }

    WAVES_PARAMS_QC_SPLIT = [
        'Timestamp (UTC)', 'Site', 'BuoyID',
        'Hsig (m)', 'Hsig_swell (m)', 'Hsig_sea (m)',
        'Tp (s)', 'Tm (s)', 'Tm_swell (s)', 'Tm_sea (s)',
        'Dp (deg)', 'DpSpr (deg)', 'Dm (deg)',
        'Dm_swell (deg)', 'Dm_sea (deg)',
        'DmSpr (deg)', 'DmSpr_swell (deg)', 'DmSpr_sea (deg)',
        'QF_waves','Latitude (deg)', 'Longitude (deg)'
    ]

    TEMP_PARAMS_QC_SPLIT = [
        'Timestamp (UTC)', 'Site', 'BuoyID',
        'SST (degC)', 'QF_sst','Latitude (deg)', 'Longitude (deg)'
    ]

    TEMP_BOTTOM_PARAMS_QC_SPLIT = [
        'Timestamp (UTC)', 'Site', 'BuoyID',
        'Bottom Temp (degC)', 'QF_bott_temp','Latitude (deg)', 'Longitude (deg)'
    ]

    WINDS_PARAMS_QC_SPLIT = [
        'Timestamp (UTC)', 'Site', 'BuoyID',
        'WindSpeed (m/s)', 'WindDirec (deg)','Latitude (deg)', 'Longitude (deg)'
    ]

    CURRENTS_PARAMS_QC_SPLIT = [
        'Timestamp (UTC)', 'Site', 'BuoyID',
        'CurrmentMag (m/s)','CurrentDir (deg)','Latitude (deg)', 'Longitude (deg)'
    ]

    PARAMETER_TYPES = ['waves', 'temp', 'temp_bottom']


    def get_time_col_name(data:pd.DataFrame) -> str:
        return [col for col in data.columns if "timestamp" in col.lower()][0]

    def convert_to_datetime(data:pd.DataFrame, timestamp_col_name: str=None):
        
        if not timestamp_col_name:
            timestamp_col_name = ProcAusWaves.get_time_col_name(data)

        if data is not None:
            data[timestamp_col_name] = (pd.to_datetime(data[timestamp_col_name], errors="coerce", utc=True)
                            .dt.tz_localize(None)) # making sure to generate tz naive times following AODN previous data and templates
            return data
        else:
            return None

    def sort_datetimes(data: pd.DataFrame) -> pd.DataFrame:
        time_col = ProcAusWaves.get_time_col_name(data)
        return data.sort_values(time_col)

    def process_time_unix_column(data:pd.DataFrame, method:str='create') -> pd.DataFrame:
        
        time_unix_col = 'time_unix'

        if method == 'create':
            time_col = ProcAusWaves.get_time_col_name(data)
            data[time_unix_col] = data[time_col].astype("int64") // 10**9

        elif method == 'drop':
            reverse_map = {
                v[0]: k
                for k, v in ProcAusWaves.COLS_MAPPING.items()
                if "unix" in v[0].lower()
            }
            data = data.rename(columns=reverse_map)
            data = data.drop(columns=time_unix_col)

        return data
        

    def drop_redundant_columns(data:pd.DataFrame) -> pd.DataFrame:
        redundant_columns = ['latitude', 'longitude', 'processing_source']
        return data.drop(columns=redundant_columns)

    def process_partition_data(data:pd.DataFrame) -> pd.DataFrame:
        
        partitions = ['swell', 'sea']

        data[partitions] = pd.DataFrame(data["partitions"].tolist(), index=data.index)
        data = data.drop(columns='partitions')

        for partition in partitions:
            expanded_partition = pd.json_normalize(data[partition])
            expanded_partition.columns = [f"{c}_{partition}" for c in expanded_partition.columns]
            data = pd.concat([data, expanded_partition], axis=1)

        return data
    
    def merge_on_time(data:list[pd.DataFrame], method:str="merge") -> pd.DataFrame:
        
        time_col = ProcAusWaves.get_time_col_name(data[0])

        if method == "concat":
            data = [df.set_index(time_col) for df in data]
            data_merged = pd.concat(data, axis=1).reset_index()

        elif method == "merge":
            from functools import reduce  # your list of DataFrames

            data_merged = reduce(
                lambda left, right: pd.merge(left, right, on=time_col, how="outer"),
                data
            )

        return data_merged
    
    def conform_cols_to_auswaves(data: pd.DataFrame, strip: bool=True) -> pd.DataFrame:
        
        mapping = {}

        for k, (name, to_strip) in ProcAusWaves.COLS_MAPPING.items():
            if not strip and not to_strip:
                mapping[name] = f" {name}"
            else:
                mapping[k] = name

        return data.rename(columns=mapping)
    
    
    def fill_nan_9999(data:pd.DataFrame) -> pd.DataFrame:
        return data.fillna(-9999.0)
    
    def replace_9999_nan(data:pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        return data.replace(-9999.0, np.nan)
    
    def conform_timestamp_format_auswaves(dfs:list[pd.DataFrame]) -> list[pd.DataFrame]:

        for date in dfs:
            data = date['data']
            time_col = ProcAusWaves.get_time_col_name(data)
            data[time_col] = pd.to_datetime(data[time_col]).dt.strftime("%d-%b-%Y %H:%M:%S")
            date['data'] = data

        return dfs
    
    # Smart Mooring ===========================================================
    
    def convert_sm_to_dataframe(raw_data:dict) -> pd.DataFrame:
        return pd.DataFrame(raw_data)
    
    def pivot_sm_data(data:pd.DataFrame) -> pd.DataFrame:

        return (
            data.pivot(
                    index='timestamp',
                    columns=['data_type_name', 'sensorPosition'],
                    values='value'
                    )
                .reset_index()
            )

    def conform_sm_cols_to_auswaves(data:pd.DataFrame) -> pd.DataFrame:

        time_col = 'timestamp'
        time_data = data['timestamp']

        sensor_data = data.drop(columns=time_col)

        sensor_data.columns = [
            ProcAusWaves.SM_COLS_MAPPING.get((unit, pos), f"{unit}_pos{pos}")
            for unit, pos in sensor_data.columns if unit != time_col
        ]

        data = pd.concat([time_data, sensor_data], axis=1)

        return data

    def create_missing_columns(data:pd.DataFrame) -> pd.DataFrame:

        missing_cols = [col for col in ProcAusWaves.COLS_ORDER if col not in data.columns]

        if not missing_cols:
            return data
        
        for col in missing_cols:
            if "QF" in col:
                data[col] = 2
            else:
                data[col] = np.nan

        return data

    def drop_unwanted_columns(data:pd.DataFrame) -> pd.DataFrame:
        
        cols_to_drop = [
            col for col in data.columns
            if col not in ProcAusWaves.COLS_ORDER
        ]
        
        return data.drop(columns=cols_to_drop)

    def reorder_cols(data:pd.DataFrame) -> pd.DataFrame:
        
        columns_new_order = [col for col in ProcAusWaves.COLS_ORDER if col in data.columns]
        return data[columns_new_order]


    def combine_spotter_sm(data:pd.DataFrame, sm_data:pd.DataFrame, method:str="interpolation", merge_asof_tolerance:int=10) -> pd.DataFrame:

        time_col = ProcAusWaves.get_time_col_name(data) 

        if method == "interpolation":

            sm_data_indexed = sm_data.set_index(time_col)
            
            sm_data_interp = (
                sm_data_indexed
                .reindex(sm_data_indexed.index.union(data[time_col]))  # keep original + target
                .sort_index()
                .interpolate(method="time")
                .reindex(data[time_col])  # now extract only low timestamps
            )

            return pd.merge(data, sm_data_interp, on=time_col, how='outer')
                    
        elif method == "merge_asof":
            
            tolerance = pd.Timedelta(merge_asof_tolerance, unit='minutes')
            return pd.merge_asof(data, sm_data, on=time_col, tolerance=tolerance) 
        

    # QC Processing ================================

    def conform_AODN_cols(data:pd.DataFrame, direction:str='forwards') -> pd.DataFrame:
        
        if direction == 'forwards':
            return data.rename(columns=ProcAusWaves.QC_MAP)
        
        elif direction == 'backwards':
            # invert dictionary: values -> keys
            reverse_map = {v: k for k, v in ProcAusWaves.QC_MAP.items()}
            return data.rename(columns=reverse_map)
        
        else:
            raise ValueError("direction must be either 'forwards' or 'backwards'")
        
    def split_parameters_types(data:pd.DataFrame,
                               qc_waves:bool=True,
                               qc_temp:bool=True,
                               qc_temp_bottom:bool=True,
                               qc_winds:bool=False,
                               qc_currents:bool=False) -> dict:

        waves = data[ProcAusWaves.WAVES_PARAMS_QC_SPLIT]
        temp = data[ProcAusWaves.TEMP_PARAMS_QC_SPLIT]
        temp_bottom = data[ProcAusWaves.TEMP_BOTTOM_PARAMS_QC_SPLIT]
        winds = data[ProcAusWaves.WINDS_PARAMS_QC_SPLIT]
        currents = data[ProcAusWaves.CURRENTS_PARAMS_QC_SPLIT]

        return {
            "waves": {
                "data": waves,
                "data_qced": None,
                "to_qc": qc_waves
            },
            "temp": {
                "data": temp,
                "data_qced": None,
                "to_qc": qc_temp
            },
            "temp_bottom": {
                "data": temp_bottom,
                "data_qced": None,
                "to_qc": qc_temp_bottom
            },
            "winds": {
                "data": winds,
                "data_qced": None,
                "to_qc": qc_winds
            },
            "currents": {
                "data": currents,
                "data_qced": None,
                "to_qc": qc_currents
            }
        }
    
    def combine_parameters_types(data:pd.DataFrame) -> pd.DataFrame:

        dfs = []
        for param_type, metadata in data.items():
            
            if metadata['to_qc']:
                dfs.append(metadata['data_qced'])
            
            else:
                dfs.append(metadata['data'])
        
        dfs_combined = pd.concat(dfs, axis=1)
        dfs_combined = dfs_combined.loc[:, ~dfs_combined.columns.duplicated()]
        
        return dfs_combined
    

    # Process previous data =======================================

    def strip_columns(data:pd.DataFrame) -> pd.DataFrame:

        stripped_cols = data.columns.str.strip()
        data.columns = stripped_cols

        return data
    
    def concat_previous_new(previous_data:pd.DataFrame, new_data:pd.DataFrame,) -> pd.DataFrame:

        if not previous_data.columns.equals(new_data.columns):
            raise KeyError(f"Mismatch of columns between new_data and previous data. new_data columns: {new_data.columns}; previous_data columns:{previous_data.columns}")

        concat_data = pd.concat([previous_data, new_data], ignore_index=True)

        return concat_data
    
    def drop_timestamp_duplicates(data:pd.DataFrame) -> pd.DataFrame:

        time_col = ProcAusWaves.get_time_col_name(data)
        return data.drop_duplicates(subset=time_col, keep=False)
    

    # Split dataframes into daily dataframes =====================

    def extract_dates_from_dataframe(data:pd.DataFrame) -> list[datetime]:
        
        time_col = ProcAusWaves.get_time_col_name(data)
        return data[time_col].dt.date.unique()
    
    def split_data_daily(data:pd.DataFrame) -> pd.DataFrame:

        time_col = ProcAusWaves.get_time_col_name(data)

        dates = ProcAusWaves.extract_dates_from_dataframe(data)

        dfs = []
        for date in dates:
            df = (data
                  .set_index(time_col)
                  .loc[str(date)]
                  .reset_index()
            )

            # swap order between timestamp and timeunix columns
            cols = df.columns.tolist()
            cols[0], cols[1] = cols[1], cols[0]
            df = df[cols]

            dfs.append({"date":date, "data":df})

        return dfs

