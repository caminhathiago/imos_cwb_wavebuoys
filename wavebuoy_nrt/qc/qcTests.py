from datetime import datetime, timedelta
import os
import logging

import pandas as pd
from ioos_qc import qartod

from wavebuoy_nrt.config.config import FILES_PATH

SITE_LOGGER = logging.getLogger("site_logger")


class WaveBuoyQC():
    waves_parameters = ['SSWMD', 'WMDS', 'WPDI', 'WPDS', 'WPFM', 'WPPE', 'WSSH']
    
    def __init__(self):
        self.qc_configs = self.get_qc_configs()
    
    def get_qc_configs(self, file_name: str = "qc_config.csv"):
        file_path = os.path.join(FILES_PATH, file_name)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            error_message = "qc limits file not found, make sure it is in the relevant path."
            SITE_LOGGER.error(error_message)
            raise Exception(error_message)
        # where are they stored?
        
    def select_qc_config(self, qc_configs: pd.DataFrame, config_id: int) -> pd.DataFrame:
        return qc_configs.loc[qc_configs["config_id"] == config_id]
    
    def convert_qc_config_to_dict(self, qc_config: pd.DataFrame) -> dict:
        return (qc_config
                .set_index("parameter")
                .drop(columns=["id","config_id"])
                .to_dict(orient="index")
            )
    
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
                start_date: datetime,
                end_date: datetime) -> pd.DataFrame:
        
        for param in parameters:
            results = self.gr
        
        
        return
    
    def gross_range_test(self,
                        data: pd.DataFrame,
                        parameter: str,
                        qc_config: dict) -> pd.DataFrame:
        
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