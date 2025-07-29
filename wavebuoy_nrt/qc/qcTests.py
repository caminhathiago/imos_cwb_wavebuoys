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
        file_path = os.path.join(os.getenv('METADATA_PATH'), file_name)
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
        variables_to_drop = ['TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE', 'processing_source'] # TEMPORARY SETUP
            
        qc_cols = ['WAVE_quality_control', 'TEMP_quality_control']
        qc_cols = [col for col in qc_cols if col in data.columns]

        variables_to_drop.extend(qc_cols)

        if "flag_previous_new" in data.columns:
            variables_to_drop.append("flag_previous_new")

        data = data.drop(columns=variables_to_drop) 

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

    def _extract_qualification_window(self, data: pd.DataFrame, window=int) -> pd.DataFrame:
        
        time_col = [col for col in data.columns if "TIME" in col][0]
        
        window_start = data[time_col].max() - timedelta(hours=window)

        data_to_ignore = (data
                    .set_index(time_col)
                    .loc[:window_start-timedelta(hours=1)]
                    .reset_index()
                )
        
        data_to_qualify = (data
                    .set_index(time_col)
                    .loc[window_start:]
                    .reset_index()
                )
        

        return data_to_ignore, data_to_qualify

    def _concatenate_qualified_ignored(self, data_to_qualify:pd.DataFrame, data_to_ignore:pd.DataFrame) -> pd.DataFrame:
        return pd.concat([data_to_ignore,data_to_qualify])

    def qualify(self,
                data: pd.DataFrame,
                parameters: list,
                parameter_type:str,
                window: int,
                gross_range_test: bool = True,
                rate_of_change_test: bool = True,
                # flat_line_test:bool = True,
                mean_std_test:bool = True,
                spike_test: bool = True) -> pd.DataFrame:
        
        self.check_qc_limits(qc_config=self.qc_config)

        data_to_ignore, data_to_qualify = self._extract_qualification_window(data, window)

        # for param in parameters:
        #     # implement static method approach
        #     if gross_range_test:
        #         data_to_qualify = self.gross_range_test(data=data_to_qualify,
        #                                     parameter=param,
        #                                     qc_config=self.qc_config_dict)
        #     if rate_of_change_test:
        #         data_to_qualify = self.rate_of_change_test(data=data_to_qualify,
        #                                         parameter=param,
        #                                         qc_config=self.qc_config_dict)
        
        tests = [
            (gross_range_test, self.gross_range_test),
            (rate_of_change_test, self.rate_of_change_test),
            # (flat_line_test, self.flat_line_test),
            (mean_std_test, self.mean_std_test),
            (spike_test, self.spike_test),
        ]

        for param in parameters:
            for test_enabled, qc_test_func in tests:
                if test_enabled:
                    data_to_qualify = qc_test_func(data=data_to_qualify, parameter=param, qc_config=self.qc_config_dict)


        data_to_qualify = self.summarize_flags(data=data_to_qualify, parameter_type=parameter_type)
        # data_to_qualify = self.summarize_flags(data=data_to_qualify, parameter_type="temp")

        data_qualified_ignored = self._concatenate_qualified_ignored(data_to_qualify, data_to_ignore)

        return data_qualified_ignored
    
    def gross_range(self,
                    parameter:str,
                    inp:pd.Series,
                    max_threshold:float,
                    min_threshold:float):
        
        inp = np.asarray(inp)
        QCFlag = np.ones(len(inp), dtype=int)

        for i in range(len(inp)):
            if inp[i] > max_threshold or inp[i] < min_threshold:
                if parameter == "WSSH":
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
        #     suspect_span=[qc_config[parameter]["gross_range_suspect_min"],
        #                 qc_config[parameter]["gross_range_suspect_max"]],
        #     fail_span=[qc_config[parameter]["gross_range_fail_min"],
        #             qc_config[parameter]["gross_range_fail_max"]]
        # )

        test_name = "gross_range_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        SITE_LOGGER.info(f"{parameter} | gross range test completed")
        return data
    
    def rate_of_change(self, inp, tinp, threshold):
        
        inp = np.asarray(inp)
        QCFlag = np.zeros(len(inp), dtype=int)

        for i in range(len(inp)):
            if i == 0:
                QCFlag[i] = 2  # Not evaluated
            else:
                if abs(inp[i] - inp[i - 1]) >= threshold:
                    QCFlag[i] = 4  # Fail
                else:
                    QCFlag[i] = 1  # Pass

        return QCFlag

    def rate_of_change_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
       
        time_col = [col for col in data.columns if "TIME" in col][0]

        time_freq = data[time_col].diff().mean().seconds

        threshold = qc_config[parameter]["rate_of_change_threshold"]# / time_freq (timefreq needs to be used with qartod.rate_of_change_test)

        results = self.rate_of_change(
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

    def mean_std(self, data: pd.Series, time: pd.Series, time_window: int, std: float) -> list[int]:

        qc_flags = []
        for i in range(len(time)):
            tnow = time.iloc[i]
            tstart = tnow - timedelta(hours=time_window)
            tend = tnow + timedelta(hours=time_window)

            if tstart >= time.iloc[0] and tend <= time.iloc[-1]:

                idx = (time >= tstart) & (time <= tend)
                ddata = data[idx].to_numpy()
                
                # idx = np.where((time >= tstart) & (time <= tend))[0]
                # ddata = data.iloc[idx].to_numpy()
                # ddata = ddata[~np.isnan(ddata)]  # remove NaNs

                # if len(ddata) == 0:
                #     qc_flags.append(2)  # Not enough data to assess
                #     continue

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

    def spike(self, time: pd.DatetimeIndex, data: pd.Series, roc: float) -> pd.Series:
        
        QCFlags = []

        for i in range(len(time)):
            if i == 0 or i == len(time) - 1:
                QCFlags.append(2)
            else:
                dum = np.diff(data[i-1:i+2])

                # spike chec
                if dum[0] > 0 and dum[1] < 0 and all(np.abs(dum) > roc):
                    QCFlags.append(4)
                elif dum[0] < 0 and dum[1] > 0 and all(np.abs(dum) > roc):
                    QCFlags.append(4)
                else:
                    QCFlags.append(1)
        
        return QCFlags

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

        # data = data.reset_index(drop=True)
        # for idx, row in data[parameter_type_qc_columns].iterrows():
        #     data.loc[idx, global_qc_column] = row.max()
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

    def create_global_qc_columns(self, data: pd.DataFrame, parameter_type:str) -> pd.DataFrame:

        if parameter_type == "waves":
            global_qc_column = "WAVE_quality_control"
        elif parameter_type == "temp":
            global_qc_column = "TEMP_quality_control"

        global_qc_columns = [col for col in data.columns if col.endswith("quality_control")]

        if not global_qc_columns:
            data[global_qc_column] = 2.0

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