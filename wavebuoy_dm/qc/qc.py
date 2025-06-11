from datetime import datetime, timedelta
import os
import logging

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from ioos_qc import qartod

SITE_LOGGER = logging.getLogger("site_logger")

load_dotenv()

class WaveBuoyQC():
    waves_parameters = ['SSWMD', 'WMDS', 'WPDI', 'WPDS', 'WPFM', 'WPPE', 'WSSH']
    
    def __init__(self, config_id: int = 1):
        self.qc_configs = self.get_qc_configs()
        self.qc_config = self.select_qc_config(qc_configs=self.qc_configs, config_id=config_id)
        self.qc_config_dict = self.convert_qc_config_to_dict(qc_config=self.qc_config )

    def get_qc_configs(self, file_name: str = "qc_config.csv"):
        # file_path = os.path.join(os.getenv('METADATA_PATH'), file_name)
        file_path = r"\\drive.irds.uwa.edu.au\OGS-COD-001\CUTTLER_wawaves\Data\aodn_nrt_python\qc_config_TC.csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            error_message = "qc limits file not found, make sure it is in the relevant path."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)
        
    def select_qc_config(self, qc_configs: pd.DataFrame, config_id: int) -> pd.DataFrame:
        return qc_configs.loc[qc_configs["config_id"] == config_id]
    
    def convert_qc_config_to_dict(self, qc_config: pd.DataFrame) -> dict:
        return (qc_config
                .set_index("parameter")
                .drop(columns=["id","config_id"])
                .to_dict(orient="index")
            )
    
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data

    def drop_unwanted_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        
        variables_to_drop = ['TIME', 'TIME_TEMP', 'timeSeries', 'LATITUDE', 'LONGITUDE', 'processing_source']

        columns_to_drop = [col for col in variables_to_drop if col in data.columns]

        qc_cols = ['WAVE_quality_control', 'TEMP_quality_control']
        qc_cols = [col for col in qc_cols if col in data.columns]

        columns_to_drop.extend(qc_cols)

        if "flag_previous_new" in data.columns:
            columns_to_drop.append("flag_previous_new")

        data = data.drop(columns=columns_to_drop) 

        return data

    def get_parameters_to_qc(self, data: pd.DataFrame, qc_config: pd.DataFrame) -> list:
       
        data = self.drop_unwanted_variables(data=data)
        data_variables = data.columns
        
        qc_config_parameters = qc_config.loc[qc_config["enable_checks"] == 1,"parameter"].values

        # check = [param in qc_config_parameters for param in data_variables]
        return [param for param in qc_config_parameters if param in data_variables]

        if all(check):
            return params_to_qc
        
        if any(check) or not all(check):
            print("not all parameters are present")
            
            params_missing = [param for param in data_variables if param not in qc_config_parameters]
            error_message = f"{params_missing} not set in the desired qc_config. Please check qc_config file"
            SITE_LOGGER.error(error_message)
            raise KeyError(error_message)

    def check_qc_limits(self, qc_config: pd.DataFrame):
        
        nan_indexes = qc_config.set_index("parameter").isna().stack()
        nan_locs = (nan_indexes[nan_indexes]
                        .reset_index()
                        .rename(columns={"level_1":"threshold"})
                        .drop(columns=0)
                        .set_index("parameter")
                        # .to_dict(orient="records")
                    )
        
        if not nan_locs.empty:
            error_message = f"tresholds not provided for some parameters and tests:\n{nan_locs}"
            # SITE_LOGGER.error(error_message)
            raise ValueError(error_message)    

    def _create_parameters_qc_column(self, data: pd.DataFrame, parameter: str, test: str) -> pd.DataFrame:
        
        not_eval_flag = 2.0
        # wave_qc_column = "WAVE_quality_control"
        
        # if parameter in self.waves_parameters:
        #     if not wave_qc_column in data.columns:
        #         data[wave_qc_column] = not_eval_flag
        # else:
        param_qc_column =  parameter + f"_{test}"
        if parameter in self.waves_parameters:
            param_qc_column = "WAVE_QC_" + param_qc_column
        else:
            param_qc_column = "TEMP_QC_" + param_qc_column
        
        data[param_qc_column] = not_eval_flag 
        
        return param_qc_column, data

    def fill_not_eval_flag(self, data: pd.DataFrame, not_eval_flag: int=2) -> pd.DataFrame:
        
        for col in data.columns:
            if col.endswith("_quality_control"):
                data[col] = not_eval_flag
        
        return data

    def qualify(self,
                data: pd.DataFrame,
                parameters: list,
                parameter_type: str,
                start_date: datetime = None,
                end_date: datetime = None,
                gross_range_test: bool = True,
                rate_of_change_test: bool = True,
                flat_line_test:bool = True,
                mean_std_test:bool = True,
                spike_test: bool = True) -> pd.DataFrame:
        
        self.check_qc_limits(qc_config=self.qc_config)

        if start_date and end_date:
            data = (data
                    .set_index("TIME")
                    .loc[start_date:end_date]
                    .reset_index()
                )

        tests = [
            (gross_range_test, self.gross_range_test),
            (rate_of_change_test, self.rate_of_change_test),
            (flat_line_test, self.flat_line_test),
            (mean_std_test, self.mean_std_test),
            (spike_test, self.spike_test),
        ]

        for param in parameters:
            for test_enabled, qc_test_func in tests:
                if test_enabled:
                    data = qc_test_func(data=data, parameter=param, qc_config=self.qc_config_dict)

        data = self.summarize_flags(data=data, parameter_type=parameter_type)

        return data
    
    def gross_range_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
        
        print(f"{parameter} - {qc_config[parameter]["gross_range_fail_min"]}")

        results = qartod.gross_range_test(
            inp=data[parameter],
            suspect_span=[qc_config[parameter]["gross_range_suspect_min"],
                          qc_config[parameter]["gross_range_suspect_max"]],
            fail_span=[qc_config[parameter]["gross_range_fail_min"],
                       qc_config[parameter]["gross_range_fail_max"]]
        )

        test_name = "gross_range_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        SITE_LOGGER.info(f"{parameter} | gross range test completed")
        return data
    
    def rate_of_change_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
       
        time_col = [col for col in data.columns if "TIME" in col][0]

        time_freq = data[time_col].diff().mean().seconds

        threshold = qc_config[parameter]["rate_of_change_threshold"] / time_freq

        results = qartod.rate_of_change_test(
            inp=data[parameter],
            tinp=data[time_col],
            threshold=threshold
        )
        
        test_name = "rate_of_change_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        qc_basic_report = self.flags_counter(results=results)
        
        SITE_LOGGER.info(f"{parameter} | rate of change test completed") #| {qc_basic_report}")
        return data

    # def flat_line(self,
    #               data: pd.Series,
    #               time: pd.Series,
    #               suspect_threshold: int,
    #               fail_threshold:int,
    #               tolerance: int):
    
    #     def check_suspect(diff_vals, lim):
    #         return 3 if np.all(np.abs(diff_vals) < lim) else 1

    #     def check_fail(diff_vals, lim):
    #         return 4 if np.all(np.abs(diff_vals) < lim) else 1

    #     def compare_qcflag(qc_sus, qc_fail):
    #         if qc_sus == 2 and qc_fail == 2:
    #             return 2
    #         elif qc_sus == 1 and qc_fail == 1:
    #             return 1
    #         else:
    #             return max(qc_sus, qc_fail)

    #     data = data.values
    #     n = len(data)
        
    #     if n < suspect_threshold:
    #         print("Dataset not long enough for FLATLINE TEST")
    #         return np.full(n, 2)

    #     QCFlag_suspect = np.zeros(n, dtype=int)
    #     for i in range(n):
    #         if i < suspect_threshold:
    #             QCFlag_suspect[i] = 2  # not assessed
    #         else:
    #             check = np.diff(data[i - suspect_threshold: i + 1])
    #             QCFlag_suspect[i] = check_suspect(check, tolerance)

    #     QCFlag_fail = np.zeros(n, dtype=int)
    #     for i in range(n):
    #         if i < fail_threshold:
    #             QCFlag_fail[i] = 2  # not assessed
    #         else:
    #             check = np.diff(data[i - fail_threshold: i + 1])
    #             QCFlag_fail[i] = check_fail(check, tolerance)

    #     QCFlag = np.zeros(n, dtype=int)
    #     for i in range(n):
    #         QCFlag[i] = compare_qcflag(QCFlag_suspect[i], QCFlag_fail[i])

    #     return QCFlag
    
    def flat_line(self,
              data: pd.Series,
              time: pd.Series,
              suspect_threshold: int,  # in hours
              fail_threshold: int,     # in hours
              tolerance: int):

        # Ensure time is datetime and data has it as index
        data = data.copy()
        data.index = pd.to_datetime(time)

        if data.shape[0] < suspect_threshold:
            print("Dataset not long enough for FLATLINE TEST")
            return pd.Series([2] * len(data), index=data.index)

        # Define subfunctions
        def make_check(threshold_val, window_hours):
            def check(vals):
                if vals.isna().any():
                    return 2  # not assessed
               
                diffs = vals.diff().dropna().abs()
                if (diffs < tolerance).all():
                    return threshold_val
                return 1
            return check

        check_suspect_fn = make_check(3, suspect_threshold)  # suspect = 3
        check_fail_fn = make_check(4, fail_threshold)     # fail = 4

        # Apply rolling checks
        print("checking suspect")
        rolling_suspect = data.rolling(f"{suspect_threshold}h").apply(check_suspect_fn, raw=False)
        print("checking fail")
        rolling_fail = data.rolling(f"{fail_threshold}h").apply(check_fail_fn, raw=False)
        print("checks done")
        # Fill leading unassessed values
        rolling_suspect.iloc[:suspect_threshold] = 2
        rolling_fail.iloc[:fail_threshold] = 2

        # Compare flags
        def compare_qcflag(row):
            qc_sus, qc_fail = row
            if qc_sus == 2 and qc_fail == 2:
                return 2
            elif qc_sus == 1 and qc_fail == 1:
                return 1
            else:
                return max(qc_sus, qc_fail)

        combined_qc = pd.concat([rolling_suspect, rolling_fail], axis=1).apply(compare_qcflag, axis=1)
        return combined_qc.astype(int)

    def flat_line_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
       
        time_col = [col for col in data.columns if "TIME" in col][0]

        results = self.flat_line(
            data=data[parameter],
            time=data[time_col],
            suspect_threshold=qc_config[parameter]["flat_line_suspect_time_dm"],
            fail_threshold=qc_config[parameter]["flat_line_fail_time_dm"],
            tolerance=qc_config[parameter]["flat_line_tol_dm"]
        )
        
        
        test_name = "flat_line_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results.values

        qc_basic_report = self.flags_counter(results=results)
        
        SITE_LOGGER.info(f"{parameter} | flat line test completed") #| {qc_basic_report}")
        return data

    def mean_std(self, time:pd.DatetimeIndex, time_window:int, std:float, data: pd.Series):
       
         # Make sure the DataFrame index matches the given time
        data = data.copy()
        data.index = time
        # Apply rolling window based on time
        rolling_obj = data.rolling(window=f"{int(time_window)}h", center=True, min_periods=1)
        

        # Calculate rolling mean and std
        rolling_mean = rolling_obj.mean()
        rolling_std = rolling_obj.std()

        # Calculate high and low thresholds
        Mhi = rolling_mean + std * rolling_std
        Mlow = rolling_mean - std * rolling_std
        
        # Initialize QC flags: 2 (NOT ASSESSED)
        QCFlag = pd.Series(2, index=data.index, dtype=int)

        # Assign flags based on comparison
        # If data > Mhi or data < Mlow => SUSPECT (3)
        suspect = (data > Mhi) | (data < Mlow)
        QCFlag[suspect] = 3

        # If data within thresholds => GOOD (1)
        good = (~suspect) & (~data.isna())
        QCFlag[good] = 1

        # NaNs stay as NOT ASSESSED (2)

        return QCFlag.values

    def mean_std_test(self, data: pd.DataFrame, parameter:str, qc_config: dict):
        
        time_col = [col for col in data.columns if "TIME" in col][0]

        results = self.mean_std(data=data[parameter], 
                                     time=data[time_col],
                                     time_window=qc_config[parameter]["mean_std_time_window"],
                                     std=qc_config[parameter]["mean_std_std"])
        
        test_name = "mean_std_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        SITE_LOGGER.info(f"{parameter} | mean std test completed")
        return data

    def spike(self, time: pd.DatetimeIndex, data: pd.Series, roc: float) -> pd.Series:
        """
        Spike test using a rolling window approach (centered on current point).
        
        Parameters:
        - time: pd.DatetimeIndex of the data
        - data: pd.Series with the data values
        - roc: rate of change threshold to flag spikes
        
        Returns:
        - pd.Series of QC flags (1=GOOD, 2=NOT ASSESSED, 4=SPIKE)
        """
        # Ensure data has proper index
        data = data.copy()
        data.index = time
        
        # Initialize QC flags to NOT ASSESSED (2)
        QCFlag = pd.Series(2, index=data.index, dtype=int)

        # Shifted differences
        diff_prev = data.diff()
        diff_next = data[::-1].diff()[::-1]  # equivalent to forward diff

        # Spike detection: central point much higher/lower than neighbors
        spike = (
            ((diff_prev > 0) & (diff_next < 0) & (diff_prev.abs() > roc) & (diff_next.abs() > roc)) |
            ((diff_prev < 0) & (diff_next > 0) & (diff_prev.abs() > roc) & (diff_next.abs() > roc))
        )

        QCFlag[spike] = 4  # SPIKE
        QCFlag[~spike & ~data.isna()] = 1  # GOOD
        # NaN values remain 2 (NOT ASSESSED)

        # First and last points: not enough context to assess
        QCFlag.iloc[0] = 2
        QCFlag.iloc[-1] = 2

        return QCFlag.values
    
    def spike_test(self, data: pd.DataFrame, parameter:str, qc_config: dict):
        
        time_col = [col for col in data.columns if "TIME" in col][0]

        results = self.spike(data=data[parameter], 
                            time=data[time_col],
                            roc=qc_config[parameter]["spike_test_roc"])
        
        
        test_name = "spike_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        SITE_LOGGER.info(f"{parameter} | spike test completed")
        return data

    def flags_counter(self, results: np.array):
        unique, counts = np.unique(results, return_counts=True)
        return dict(zip(unique, counts))

    def summarize_flags(self,
                        data: pd.DataFrame,
                        parameter_type: str = "waves",
                        drop_parameters_qc_columns: bool = True) -> pd.DataFrame:
        
        """
        the main idea is to summarize qc flags for each parameter in one flag to be stored in WAVES_quality_control
        - select only qc related columns with filter(regex)
        - choose the highest value of the parameters (except for 9)
        - if Peak Period fails, everything fails
        - 
        
        """
        
        if parameter_type == "waves":
            qc_col_prefix = "WAVE_QC_"
            global_qc_column = "WAVE_quality_control"
        elif parameter_type == "temp":
            qc_col_prefix = "TEMP_QC_"
            global_qc_column = "TEMP_quality_control"


        parameter_type_qc_columns = data.filter(regex=qc_col_prefix).columns
        
        if parameter_type_qc_columns.empty:
            return data

        # for idx, row in data[parameter_type_qc_columns].iterrows():
        #     data.loc[idx, global_qc_column] = row.max()
        data[global_qc_column] = data[parameter_type_qc_columns].max(axis=1)


        if drop_parameters_qc_columns:
            data = self.drop_parameters_qc_columns(data=data, qc_col_prefix=qc_col_prefix)

        return data

    def drop_parameters_qc_columns(self, data: pd.DataFrame, qc_col_prefix: str) -> pd.DataFrame:
        qc_columns = data.filter(regex=qc_col_prefix).columns
        parameters_qc_columns = [col for col in qc_columns if not col.endswith("_quality_control")]
        return data.drop(columns=parameters_qc_columns)

    def create_global_qc_columns(self, data: pd.DataFrame) -> pd.DataFrame:

        wave_qc_column = "WAVE_quality_control"
        temp_qc_column = "TEMP_quality_control"

        if "TEMP" not in data.columns:
            data[wave_qc_column] = 2.0
        else:
            data[temp_qc_column] = 2.0

        return data


   # def compose_config(self,
    #                    data: pd.DataFrame,
    #                     parameters: list,
    #                     start_date: datetime,
    #                     end_date: datetime):

    #     return
    
    # def compose_variable_stream_block(self, parameter: str) -> str:
    #     config_stream_base = """streams:
    #                             {parameter}:
    #                                 qartod:
    #                                 aggregate:
    #                                 {test_block}
    #                                 """
        
    # def compose_test_block(self, test: str, qc_limits: list) -> str:
    #     if not "rate_of_change_test":
    #         config_test_base = """{test}:
    #                             suspect_span: [{qc_limit_suspect_min}, {qc_limit_suspect_max}]
    #                             fail_span: [{qc_limit_fail_min}, {qc_limit_fail_max}]
    #                             """
    #     else:
    #         config_test_base = """{test}:
    #                             suspect_span: [{qc_limit_suspect_min}, {qc_limit_suspect_max}]
    #                             fail_span: [{qc_limit_fail_min}, {qc_limit_fail_max}]
    #                             """


    # def create_flags_columns(self, data: pd.DataFrame, parameters: list) -> pd.DataFrame:
        
    #     not_eval_flag = 2.0
    #     wave_qc_column = "WAVE_quality_control"
        
    #     for param in parameters:
    #         if param in self.waves_parameters:
    #             if not wave_qc_column in data.columns:
    #                 data[wave_qc_column] = not_eval_flag
    #         else:
    #             param_qc_column = param + "_quality_control"
    #             data[param_qc_column] = not_eval_flag 
        
    #     return data