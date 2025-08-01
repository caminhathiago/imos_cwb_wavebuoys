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
                        'LOC':{"GPS_Epoch_Time(s)": pl.Int64,
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
    def validate_schema(file_path: str, suffix: str) -> bool:
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
        
    def _validate_schemas():
        pass

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

    def ignore_files(self, file: str, size: int = 50, file_id: str = '0000'):
        """
        size in kbytes
        """
        if os.path.isfile(file):
            if os.path.getsize(file) < size * 1024 or file_id in os.path.basename(file):
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
        """
        Load a CSV file into a Polars lazy dataframe.

        Args:
            file (str): The file path to the CSV file.
            truncate_ragged_lines (bool): IF polars should truncated ragged lines.
                This is useful when working with spotters because sometimes CSV files contain rows
                with more fields than columns, so this argument makes sure those are ignored.
        Returns:
            pl.LazyFrame: A lazy dataframe containing the CSV data.
        """
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
        
        return map_suffixes_files

    def map_concat_results(self) -> dict:
        return {suffix: [] for suffix in self.suffixes_to_concat} 

    # def get_displacement_columns(self, df_lazy: pl.LazyFrame) -> list:
    #     columns_ignore = ["GPS_Epoch_Time(s)", "millis"]
    #     return [col for col in df_lazy.collect_schema().keys() if "out" in col]

    def lazy_concat_files(self) -> pl.LazyFrame:
        """
        Process all CSV files in the list by loading them, adding a datetime column,
        and concatenating them into a single lazy dataframe.

        Returns:
            pl.LazyFrame: A single lazy dataframe containing all processed data.
        """

        results = self.map_concat_results()

        for suffix in self.files_suffixes.keys():
            # print(suffix)
            data_list = []
            
            for file in self.files_suffixes[suffix]:
                if self.ignore_files(file=file): #and self.validate_schema(file, suffix):
                    df_lazy = self.load_csv(file)
                    # print(list(df_lazy.collect_schema()))
                    data_list.append(df_lazy)

            if not data_list:
                continue

            

            concat_df_lazy = pl.concat(data_list, how="vertical")    
            results.update({suffix:concat_df_lazy})

            # print("="*60)

        return results

    



# # Usage Example
# if __name__ == "__main__":
#     files = ['file1.csv', 'file2.csv', 'file3.csv']  # Replace with actual file paths
#     processor = csvConcatenator(files)
#     final_df_lazy = processor.process_files()

#     # Show schema of the final dataframe
#     print(final_df_lazy.collect_schema())
