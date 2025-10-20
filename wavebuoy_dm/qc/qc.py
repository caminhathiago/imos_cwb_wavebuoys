from datetime import datetime, timedelta
import os
import logging
import warnings

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
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    def get_qc_configs(self, file_name: str = "qc_config.csv"):
        
        file_path = os.path.join(os.getenv('METADATA_PATH'), file_name)
        
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        else:
            error_message = "qc limits file not found, make sure it is in the relevant path."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)
        
    def select_qc_config(self, qc_configs: pd.DataFrame, config_id: int) -> pd.DataFrame:
        
        if not (isinstance(config_id, int) or isinstance(config_id, float)):
            raise ValueError(f"config id passed ({config_id}) is not valid. Check buoys_to_process.csv")
        
        return qc_configs.loc[qc_configs["config_id"] == config_id]
    
    def convert_qc_config_to_dict(self, qc_config: pd.DataFrame) -> dict:
        
        return (qc_config
                .set_index("parameter")
                .drop(columns=["id","config_id"])
                .to_dict(orient="index")
            )

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

        return [param for param in qc_config_parameters if param in data_variables]

    def check_qc_limits(self, qc_config: pd.DataFrame):
        
        nan_indexes = qc_config.set_index("parameter").isna().stack()
        nan_locs = (nan_indexes[nan_indexes]
                        .reset_index()
                        .rename(columns={"level_1":"threshold"})
                        .drop(columns=0)
                        .set_index("parameter")
                    )
        
        if not nan_locs.empty:
            error_message = f"tresholds not provided for some parameters and tests:\n{nan_locs}"
            raise ValueError(error_message)    

    def _create_parameters_qc_column(self, data: pd.DataFrame, parameter: str, test: str) -> pd.DataFrame:
        
        not_eval_flag = 2.0
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

        time_col = [col for col in data.columns if "TIME" in col.upper()][0]

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
    
    def gross_range(self,
                    parameter:str,
                    inp:pd.Series,
                    max_threshold:float,
                    min_threshold:float):
        
        inp = np.asarray(inp)
        QCFlag = np.ones(len(inp), dtype=int)

        for i in range(len(inp)):
            if inp[i] > max_threshold or inp[i] < min_threshold:
                if parameter == "WSSH" or parameter == "WPPE":
                    QCFlag[i] = 4
                else:
                    QCFlag[i] = 3
                    
        return QCFlag
    
    def gross_range_test(self,
                        data:pd.DataFrame,
                        parameter:str,
                        qc_config:dict) -> pd.DataFrame:

        results = self.gross_range(
            parameter=parameter,
            inp=data[parameter],
            max_threshold=qc_config[parameter]["gross_range_fail_max"],
            min_threshold=qc_config[parameter]["gross_range_fail_min"]
            )

        test_name = "gross_range_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        SITE_LOGGER.info(f"{parameter} | gross range test completed")
        return data
    
    def rate_of_change(self, inp, threshold, directional=False):
        inp = np.asarray(inp)
        QCFlag = np.zeros(len(inp), dtype=int)

        for i in range(len(inp)):
            if i == 0:
                QCFlag[i] = 2
            else:
                if directional:
                    diff = abs((inp[i] - inp[i - 1] + 180) % 360 - 180)
                else:
                    diff = abs(inp[i] - inp[i - 1])
                
                QCFlag[i] = 4 if diff >= threshold else 1

        return QCFlag

    def classify_directional_parameter(self, parameter: str) -> bool:
        return "D" in parameter

    def rate_of_change_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
       
        time_col = [col for col in data.columns if "TIME" in col][0]

        time_freq = data[time_col].diff().mean().seconds

        threshold = qc_config[parameter]["rate_of_change_threshold"]# / time_freq (timefreq needs to be used with qartod.rate_of_change_test)

        directional = self.classify_directional_parameter(parameter)

        results = self.rate_of_change(
            inp=data[parameter],
            threshold=threshold,
            directional=directional,
        )
        
        test_name = "rate_of_change_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results
        
        SITE_LOGGER.info(f"{parameter} | rate of change test completed")
        return data

    def flat_line(self,
              data: pd.Series,
              time: pd.Series,
              suspect_threshold: int,  # in hours
              fail_threshold: int,     # in hours
              tolerance: float) -> pd.Series:
        
        def check_suspect(check_data, lim_sus):
            if np.all(np.abs(check_data) < lim_sus):
                return 3
            else:
                return 1

        def check_fail(check_data, lim_fail):
            if np.all(np.abs(check_data) < lim_fail):
                return 4
            else:
                return 1

        def compare_qcflag(qc_sus, qc_fail):
            if qc_sus == 2 and qc_fail == 2:
                return 2
            elif qc_sus == 1 and qc_fail == 1:
                return 1
            elif qc_sus < qc_fail:
                return qc_fail
            elif qc_sus > qc_fail:
                return qc_sus

        
        if len(data) >= suspect_threshold:
            
            QCFlag_suspect = np.zeros(len(data), dtype=int)
            for ii in range(len(data)):
                if ii <= suspect_threshold:
                    QCFlag_suspect[ii] = 2
                else:
                    check_data = np.diff(data[ii - suspect_threshold: ii + 1])
                    QCFlag_suspect[ii] = check_suspect(check_data, tolerance)

            QCFlag_fail = np.zeros(len(data), dtype=int)
            for ii in range(len(data)):
                if ii <= fail_threshold:
                    QCFlag_fail[ii] = 2
                else:
                    check_data = np.diff(data[ii - fail_threshold: ii + 1])
                    QCFlag_fail[ii] = check_fail(check_data, tolerance)

            QCFlag = np.zeros((len(data), 1), dtype=int)
            for ii in range(len(data)):
                QCFlag[ii] = compare_qcflag(QCFlag_suspect[ii], QCFlag_fail[ii])

        else:
            print('Dataset not long enough for FLATLINE TEST')
            QCFlag = np.ones(len(data), dtype=int) * 2

        return QCFlag

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

        data[param_qc_column] = results

        
        SITE_LOGGER.info(f"{parameter} | flat line test completed")
        return data

    def mean_std(self, data: pd.Series, time: pd.Series, time_window: int, std: float) -> list[int]:

        qc_flags = []
        for i in range(len(time)):
            tnow = time.iloc[i]
            tstart = tnow - timedelta(hours=time_window)
            tend = tnow + timedelta(hours=time_window)

            if tstart >= time.iloc[0] and tend <= time.iloc[-1]:

                idx = (time >= tstart) & (time <= tend)
                ddata = data[idx].to_numpy()

                mean = np.nanmean(ddata)
                Mhi = mean + (std * np.nanstd(ddata, ddof=1))
                Mlow = mean - (std * np.nanstd(ddata, ddof=1))

                if data.iloc[i] > Mhi or data.iloc[i] < Mlow:
                    qc_flags.append(3) 
                else:
                    qc_flags.append(1) 
            else:
                qc_flags.append(2) 

        return qc_flags

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
    
    def spike(self, time: pd.DatetimeIndex, data: pd.Series, roc: float, directional:bool = False) -> pd.Series:
        
        QCFlags = []

        for i in range(len(time)):
            if i == 0 or i == len(time) - 1:
                QCFlags.append(2)
            else:
                if directional:
                    diff1 = diff2 = abs((data[i] - data[i - 1] + 180) % 360 - 180)

                else:
                    diff1 = data[i] - data[i-1]
                    diff2 = data[i+1] - data[i]
                
                if abs(diff1) > roc and abs(diff2) > roc:
                    QCFlags.append(4)
                else:
                    QCFlags.append(1)
            
        return QCFlags

    def spike_test(self, data: pd.DataFrame, parameter:str, qc_config: dict):
        
        time_col = [col for col in data.columns if "TIME" in col][0]
        
        directional = self.classify_directional_parameter(parameter)
        
        results = self.spike(data=data[parameter], 
                            time=data[time_col],
                            roc=qc_config[parameter]["spike_test_roc"],
                            directional=directional)
        
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

        temp_flags = data[parameter_type_qc_columns].replace(2, 1)
        max_flags = temp_flags.max(axis=1)
        data[global_qc_column] = max_flags

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

    def report_qc(self, data:list[pd.DataFrame], parameters:list, output_path:str) -> None:
        
        qc_reports = []
        for df in data:
            if df is None or df.empty:
                continue
            
            tests = list(df.filter(regex="_test").columns)
            
            import re
            index_tests = [
                    re.sub(r'^(WAVE_QC_|TEMP_QC_)', '', test, flags=re.IGNORECASE) 
                    for test in tests
            ]
            
            columns = [
                'n',
                '1_count', '1_perc',
                '2_count', '2_perc',
                '3_count', '3_perc',
                '4_count', '4_perc'
            ]
            
            qc_report = pd.DataFrame(index=index_tests, columns=columns)
            
            qc_report['n'] = df.shape[0]
            
            for test, index in zip(tests, index_tests):
                counts = df[test].value_counts()
                percs = df[test].value_counts(normalize=True) * 100
                
                for idx, row in counts.items():
                    qc_report.loc[index, f'{idx}_count'] = int(counts.loc[idx])
                    qc_report.loc[index, f'{idx}_perc'] = round(percs.loc[idx],2)

            qc_reports.append(qc_report)

        qc_report = pd.concat(qc_reports)

        file_name = f'QC_subflags_reports.csv'
        file_path = os.path.join(output_path, file_name)
        qc_report.to_csv(file_path)

