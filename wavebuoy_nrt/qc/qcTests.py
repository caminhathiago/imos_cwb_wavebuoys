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
            raise Exception(error_message)
        
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
        variables_to_drop = ['TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE', 'WAVE_quality_control', 'check', 'processing_source'] # TEMPORARY SETUP
        
        try:
            data = data.drop(columns=variables_to_drop) 
        except:
            variables_to_drop.remove("WAVE_quality_control")
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
        

    def create_flags_columns(self, data: pd.DataFrame, parameters: list) -> pd.DataFrame:
        
        not_eval_flag = 2.0
        wave_qc_column = "WAVE_quality_control"
        
        for param in parameters:
            if param in self.waves_parameters:
                if not wave_qc_column in data.columns:
                    data[wave_qc_column] = not_eval_flag
            else:
                param_qc_column = param + "_quality_control"
                data[param_qc_column] = not_eval_flag 
        
        return data

    def _create_flags_column(self, data: pd.DataFrame, parameter: str, test: str) -> pd.DataFrame:
        
        not_eval_flag = 2.0
        # wave_qc_column = "WAVE_quality_control"
        
        # if parameter in self.waves_parameters:
        #     if not wave_qc_column in data.columns:
        #         data[wave_qc_column] = not_eval_flag
        # else:
        param_qc_column = parameter + f"_{test}"
        data[param_qc_column] = not_eval_flag 
        
        return data

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
            if gross_range_test:
                data = self.gross_range_test(data=data,
                                            parameter=param,
                                            qc_config=self.qc_config_dict)
            if rate_of_change_test:
                data = self.rate_of_change_test(data=data,
                                                parameter=param,
                                                qc_config=self.qc_config_dict)

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

        param_qc_column = f"{parameter}_gross_range_test"
        if not param_qc_column in data.columns:
            data = self._create_flags_column(data=data, parameter=parameter, test="gross_range_test")

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
        param_qc_column = f"{parameter}_{test_name}"
        if not param_qc_column in data.columns:
            data = self._create_flags_column(data=data, parameter=parameter, test=test_name)

        data[param_qc_column] = results

        qc_basic_report = self.flags_counter(results=results)
        
        SITE_LOGGER.info(f"{parameter} | rate of change test completed") #| {qc_basic_report}")
        return data

    def flags_counter(self, results: np.array):
        unique, counts = np.unique(results, return_counts=True)
        return dict(zip(unique, counts))

    def summarize_flags(self, data: pd.DataFrame, parameter_type: str = "waves") -> pd.DataFrame:
        
        """
        the main idea is to summarize qc flags for each parameter in one flag to be stored in WAVES_quality_control
        - select only qc related columns with filter(regex)
        - choose the highest value of the parameters (except for 9)
        - if Peak Period fails, everything fails
        - 
        
        """
        # qualified_data_summarized = pd.DataFrame([])
        # return qualified_data_summarized
        pass





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