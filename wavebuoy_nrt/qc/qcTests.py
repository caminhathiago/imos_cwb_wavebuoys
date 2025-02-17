from datetime import datetime, timedelta
import os
import logging

import pandas as pd
import numpy as np

from ioos_qc import qartod

from wavebuoy_nrt.config.config import FILES_PATH

SITE_LOGGER = logging.getLogger("site_logger")


class WaveBuoyQC():
    waves_parameters = ['SSWMD', 'WMDS', 'WPDI', 'WPDS', 'WPFM', 'WPPE', 'WSSH']
    
    def __init__(self, config_id: int = 1):
        self.qc_configs = self.get_qc_configs()
        self.qc_config = self.select_qc_config(qc_configs=self.qc_configs, config_id=config_id)
        self.qc_config_dict = self.convert_qc_config_to_dict(qc_config=self.qc_config )

    def get_qc_configs(self, file_name: str = "qc_config.csv"):
        file_path = os.path.join(FILES_PATH, file_name)
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
        
        qc_config_parameters = qc_config["parameter"].values
        
        check = [param in qc_config_parameters for param in data_variables]
        params_to_qc = [param for param in data_variables if param in qc_config_parameters]

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
                start_date: datetime = None,
                end_date: datetime = None,
                gross_range_test: bool = True,
                rate_of_change_test: bool = True) -> pd.DataFrame:
        
        self.check_qc_limits(qc_config=self.qc_config)

        if start_date and end_date:
            data = (data
                    .set_index("TIME")
                    .loc[start_date:end_date]
                    .reset_index()
                )

        for param in parameters:
            # implement static method approach
            if gross_range_test:
                data = self.gross_range_test(data=data,
                                            parameter=param,
                                            qc_config=self.qc_config_dict)
            if rate_of_change_test:
                data = self.rate_of_change_test(data=data,
                                                parameter=param,
                                                qc_config=self.qc_config_dict)
        
        data = self.summarize_flags(data=data, parameter_type="waves")
        data = self.summarize_flags(data=data, parameter_type="temp")


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
       
              
        results = qartod.rate_of_change_test(
            inp=data[parameter],
            tinp=data["TIME"],
            threshold=[qc_config[parameter]["rate_of_change_threshold"]]
        )
        
        
        test_name = "rate_of_change_test"
        param_qc_column, data = self._create_parameters_qc_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        qc_basic_report = self.flags_counter(results=results)
        
        SITE_LOGGER.info(f"{parameter} | rate of change test completed") #| {qc_basic_report}")
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

        SITE_LOGGER.warning(f"SUMMARIZE: {data.columns}")
        SITE_LOGGER.warning(f"SUMMARIZE: {parameter_type_qc_columns}")
        
        if parameter_type_qc_columns.empty:
            return data

        for idx, row in data[parameter_type_qc_columns].iterrows():
            data.loc[idx, global_qc_column] = row.max()

        if drop_parameters_qc_columns:
            data = self.drop_parameters_qc_columns(data=data, qc_col_prefix=qc_col_prefix)

        return data

    def drop_parameters_qc_columns(self, data: pd.DataFrame, qc_col_prefix: str) -> pd.DataFrame:
        qc_columns = data.filter(regex=qc_col_prefix).columns
        parameters_qc_columns = [col for col in qc_columns if not col.endswith("_quality_control")]
        SITE_LOGGER.warning(f"DROP PARAMETERS: {parameters_qc_columns}")
        return data.drop(columns=parameters_qc_columns)

    def create_global_qc_columns(self, data: pd.DataFrame) -> pd.DataFrame:

        wave_qc_column = "WAVE_quality_control"
        temp_qc_column = "TEMP_quality_control"

        data[wave_qc_column] = 2.0
        if "TEMP" in data.columns:
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