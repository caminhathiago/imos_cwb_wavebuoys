from typing import List, Dict
import os
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import glob
import polars as pl


class csvConcat:
    """
    A class for processing multiple CSV files using Polars lazy dataframes.

    Attributes:
        files (List[str]): List of file paths to be processed.
        data_list (List[pl.LazyFrame]): List of lazy dataframes processed from the files.
    """

    EXPECTED_SCHEMAS = {"FLT":{"millis": pl.Int64,
                                "GPS_Epoch_Time(s)": pl.Float64,
                                "outx(mm)": pl.Float64,
                                "outy(mm)": pl.Float64,
                                "outz(mm)": pl.Float64
                                },
                        'SPC':{"type": pl.Utf8,
                                "millis": pl.Int64,
                                "t0_GPS_Epoch_Time(s)": pl.Float64,
                                "tN_GPS_Epoch_Time(s)": pl.Float64,
                                "ens_count": pl.Int64,
                                "Sxx_re": pl.Float64,
                                "Syy_re": pl.Float64,
                                "Szz_re": pl.Float64,
                                "Sxy_re": pl.Float64,
                                "Szx_re": pl.Float64,
                                "Szy_re": pl.Float64,
                                "Sxx_im": pl.Float64,
                                "Syy_im": pl.Float64,
                                "Szz_im": pl.Float64,
                                "Sxy_im": pl.Float64,
                                "Szx_im": pl.Float64,
                                "Szy_im": pl.Float64
                                },
                        'LOC':{"GPS_Epoch_Time(s)": (pl.Float64, pl.Int64),
                                "lat(deg)": pl.Int64,
                                "lat(min*1e5)": pl.Int64,
                                "long(deg)": pl.Int64,
                                "long(min*1e5)": pl.Int64
                                },
                        'SST':{"timestamp (ticks/UTC)": pl.Float64,
                                "temperature (C)": pl.Float64 
                                },
                        'HDR':{"GPS_Epoch_Time(s)": pl.Float64,
                                "dmx(mm)": pl.Int64,  
                                "dmy(mm)": pl.Int64,
                                "dmz(mm)": pl.Int64,
                                "dmn(mm)": pl.Int64
                                },
                        'BARO':{"timestamp (ticks/UTC)": pl.Float64,  
                                "pressure (mbar)": pl.Float64 
                                },
                        'SENS_AGG':{
                                'bm_node_id': pl.Utf8,
                                'node_position': pl.Int64,
                                'node_app_name': pl.Utf8, 
                                "timestamp (ticks/UTC)": pl.Float64, 
                                'reading_count': pl.Int64
                                 },
                        'SENS_IND':{
                                'bm_node_id': pl.Utf8,
                                'node_position': pl.Int64,
                                'node_app_name': pl.Utf8, 
                                'reading_uptime_millis': pl.Int64,
                                "reading_time_utc_s": pl.Float64, 
                                'sensor_reading_time_s': pl.Float64
                                 }
                        }

    def __init__(self,
                 files_path: List[str],
                 suffixes_to_concat: List[str] = None,
                 suffixes_to_parse: List[str] = None) -> None:
        """
        Initialize the DataProcessor with a list of file paths.

        Args:
            files (List[str]): A list of file paths to process.
        """
        self.files_path = files_path
        
        if suffixes_to_concat:
            self.suffixes_to_concat = suffixes_to_concat
        else:
            self.suffixes_to_concat = ['FLT','SPC','LOC','SST','HDR','BARO','SMD','SENS_AGG']

        if suffixes_to_parse:
            self.suffixes_to_parse = suffixes_to_concat
        else:
            self.suffixes_to_parse = ['FLT','SPC','LOC','SST']

        self.files_suffixes = self.map_files_suffixes()
        self.suffixes_schemas = self.scan_schemas()
        
        self.expected_schemas = {
            'FLT': {'outx(mm)': 'x', 'outy(mm)': 'y', 'outz(mm)': 'z'},
            'SPC': {},
            'LOC': {'lat(deg)':'lat', 'lat(min*1e5)':'lat_min', 'long(deg)':'lon', 'long(min*1e5)':'lon_min'},
            'SST': {'temperature (C)':'temperature'},
            'HDR': {'dmx(mm)':'x', 'dmy(mm)':'y', 'dmz(mm)':'z', 'dmn(mm)':'n'},
            'BARO': {'pressure (mbar)':'baro_pressure'},
            'SENS_AGG': {
                'bm_node_id':'bm_node_id', 'node_position':'node_position', 
                'node_app_name':'node_app_name', 'timestamp(ticks/UTC)':'timestamp',
                'reading_count':'reading_count'
                },
            'SENS_IND': {
                'bm_node_id':'bm_node_id',
                'node_position':'node_position',
                'node_app_name':'node_app_name',	
                'reading_uptime_millis':'reading_uptime_millis',	
                'reading_time_utc_s':'reading_time_utc_s',	
                'sensor_reading_time_s':'sensor_reading_time_s'	
                }
            }

    @staticmethod
    def validate_schema_dep(file_path: str, suffix: str) -> bool:
        """Check if a CSV file matches the expected schema."""
        try:
            df = pl.scan_csv(file_path)
            inferred_schema = df.collect_schema()
            expected_schema = csvConcat.EXPECTED_SCHEMAS[suffix]
            
            inferred_types = list(inferred_schema.values())
            expected_types = list(expected_schema.values())

            if len(inferred_types) != len(expected_schema):
                return False

            for inferred_field, expected_field in zip(inferred_schema, expected_schema):
                if inferred_field != expected_field:
                    return False

            for inferred_type, expected_type in zip(inferred_types, expected_types):
                if inferred_type != expected_type:
                    return False

            return True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
        
    def validate_schema(self, file_path: str, suffix: str) -> bool:
        """Check if a CSV file matches the expected schema and log mismatches."""
        try:
            df = pl.scan_csv(file_path)
            inferred_schema = df.collect_schema()
            expected_schema = csvConcat.EXPECTED_SCHEMAS[suffix]

            inferred_columns = list(inferred_schema.keys())
            expected_columns = list(expected_schema.keys())

            missing = set(expected_columns) - set(inferred_columns)
            extra = set(inferred_columns) - set(expected_columns)
            
            # if missing:
            #     print(f"Trying forcing header to {file_path}")
            #     # df, missing, extra, inferred_schema = self._force_header_corrupted_file(file_path, expected_columns)
            #     df, missing, extra, inferred_schema = self.fix_corrupted_csv(file_path, expected_columns)

            if missing or extra:
                print(f"Schema mismatch in {file_path}:")
                if missing:
                    print(f"  Missing columns: {missing}")
                if extra:
                    print(f"  Extra columns: {extra}")
                return False

            for col in expected_columns:
                inferred_type = inferred_schema[col]
                expected_type = expected_schema[col]
                if isinstance(expected_type, tuple):
                    if inferred_type not in expected_type:
                        print(f"Schema mismatch in {file_path}: column '{col}' "
                            f"has type {inferred_type}, expected one of {expected_type}")
                        return False
                else:
                    if inferred_type != expected_type:
                        print(f"Schema mismatch in {file_path}: column '{col}' "
                            f"has type {inferred_type}, expected {expected_type}")
                        return False

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def _force_header_corrupted_file(self, file_path:str, expected_columns):

        df_lazy = pl.scan_csv(file_path, has_header=False, new_columns=expected_columns)
        inferred_schema = df_lazy.collect_schema()
        inferred_columns = list(inferred_schema.keys())
        missing = set(expected_columns) - set(inferred_columns)
        extra = set(inferred_columns) - set(expected_columns)

        if not missing or extra:
            base, ext = os.path.splitext(file_path)
            backup_path = f"{base}_header_corrupted{ext}"

            os.rename(file_path, backup_path)
            print(f"[Info] Original corrupted file renamed to: {backup_path}")

            df = pl.read_csv(backup_path, has_header=False, new_columns=expected_columns)

            df.write_csv(file_path)
            print(f"[Info] Fixed CSV with headers saved as original filename: {file_path}")

        return df_lazy, missing, extra, inferred_schema

    def fix_corrupted_csv(self, file_path, expected_columns):
        # Step 1: backup original
        base, ext = os.path.splitext(file_path)
        backup_path = f"{base}_header_corrupted{ext}"
        os.rename(file_path, backup_path)
        print(f"Original file backed up as: {backup_path}")

        # Step 2: read lines manually and parse
        cleaned_rows = []
        with open(backup_path, newline='') as f:
            import csv
            reader = csv.reader(f)
            for row in reader:
                # skip completely empty rows
                if not row:
                    continue
                # keep only the first N columns (truncate extra)
                truncated_row = row[:len(expected_columns)]
                # pad missing columns with empty string or None
                while len(truncated_row) < len(expected_columns):
                    truncated_row.append(None)
                cleaned_rows.append(truncated_row)

        # Step 3: convert list of lists to dict of columns
        columns_dict = {col: [row[i] for row in cleaned_rows] for i, col in enumerate(expected_columns)}

        # Step 4: create Polars DataFrame
        df = pl.DataFrame(columns_dict)

        inferred_schema = df.collect_schema()
        inferred_columns = list(inferred_schema.keys())
        missing = set(expected_columns) - set(inferred_columns)
        extra = set(inferred_columns) - set(expected_columns)

        # Step 5: save cleaned CSV as original filename
        df.write_csv(file_path)
        print(f"Cleaned CSV saved as: {file_path}")

        return df.lazy(), missing, extra, inferred_columns

    @staticmethod
    def _correct_corrupted_no_headers(file_path: str, suffix: str) -> pl.LazyFrame:
            # Read the CSV as a LazyFrame without any assumptions about headers
        lf = pl.scan_csv(file_path, has_header=False, truncate_ragged_lines=True)
        inferred_columns = lf.collect_schema().names()
        print(inferred_columns)
        # Check if the schema matches the expected one
        if len(inferred_columns) != len(csvConcat.EXPECTED_SCHEMAS[suffix]):
            print("TEST")
            for i in range(1, len(inferred_columns)):  # Start from row 1 to ignore the first corrupted row
                # Try to exclude the current and previous rows and re-read the CSV
                new_lf = pl.scan_csv(file_path, has_header=False, truncate_ragged_lines=True).slice(i, None)
                # # Create the corrected DataFrame by renaming columns
                try:
                    new_lf = new_lf.rename({
                        f"column_{idx}": name for idx, name in enumerate(csvConcat.EXPECTED_SCHEMAS[suffix].keys())
                    })
                    
                    # Check if the new schema matches the expected schema
                    if set(lf.collect_schema().names()) == set(csvConcat.EXPECTED_SCHEMAS[suffix].keys()):
                        # If schema is valid, return the corrected LazyFrame
                        return new_lf

                except Exception as e:
                    # If an error occurs during renaming or processing, continue to the next iteration
                    continue
        
        # If no valid schema found, return the original LazyFrame
        return lf

    def ignore_files(self, file: str, size: int = 70, file_id: str = '0000'):
        """
        size in kbytes
        """
        if os.path.isfile(file):
            if os.path.getsize(file) < size or file_id in os.path.basename(file):
                return False
        
        return True

    def collect_schema(self, file:str , suffix: str):
        return self.load_csv(file=file).collect_schema()

    def scan_schemas(self) -> dict:
        suffixes_schemas = {}
        for suffix in self.files_suffixes.keys():
            files_schemas = []
            for file in self.files_suffixes[suffix]:
                schema = self.load_csv(file).collect_schema()
                files_schemas.append({file:schema})
            suffixes_schemas.update({suffix:files_schemas})
        return suffixes_schemas

    def load_csv(self, file: str, truncate_ragged_lines: bool = True) -> pl.LazyFrame:
        return pl.scan_csv(file,
                           truncate_ragged_lines=truncate_ragged_lines,
                           infer_schema_length=1000,
                           ignore_errors=True)        
    
    def filter_files_suffix(self, suffix:str, extension:str = ".csv") -> list:
        return glob.glob(os.path.join(self.files_path, "*" + suffix + extension))
    
    def map_files_suffixes(self) -> dict:
        
        map_suffixes_files = {suffix: [] for suffix in self.suffixes_to_concat}
        for suffix in map_suffixes_files.keys():
            files_list = self.filter_files_suffix(suffix=suffix)
            map_suffixes_files.update({suffix: files_list})

        if not map_suffixes_files.get("FLT") or not map_suffixes_files.get("LOC"):
            raise FileNotFoundError(f"No files found for FLT or LOC. Please check if path contains SD card files.")

        return map_suffixes_files

    def map_concat_results(self) -> dict:
        return {suffix: [] for suffix in self.suffixes_to_concat} 

    def lazy_concat_files(self) -> pl.LazyFrame:
      
        results = self.map_concat_results()

        for suffix in self.files_suffixes.keys():
            data_list = []
            
            for file in self.files_suffixes[suffix]:
                if self.ignore_files(file=file) and self.validate_schema(file, suffix):
                    df_lazy = self.load_csv(file)
                    data_list.append(df_lazy)

            if not data_list:
                continue

            concat_df_lazy = pl.concat(data_list, how="vertical")    
            results.update({suffix:concat_df_lazy})

        return results