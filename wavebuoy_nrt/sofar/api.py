from datetime import datetime

from pysofar.sofar import SofarApi
from pysofar.spotter import Spotter
import pandas as pd


class SofarAPI:
    def __init__(self, buoys_metadata:pd.DataFrame):
        self.buoys_metadata = buoys_metadata
        self.api = self._initiate_sofar_api()
        self.spotter_grid = self._load_spotter_grid()
        self.devices = self._load_spotter_devices()

    def _get_first_token(self):
        return (self.buoys_metadata
                        .sort_values("sofar_token")
                        .iloc[0]
                        .sofar_token
        )
        
    def _initiate_sofar_api(self):
        return SofarApi(custom_token=self._get_first_token())
    
    def _reload_sofar_api(self, token:str):
        return SofarApi(custom_token=token)

    def _validate_token(self, token:str) -> bool:
        if not isinstance(token, str) or len(token) != 30:
            raise ValueError("provide a valid token.")

    def check_token_iteration(self, next_token:str) -> bool:
        
        self._validate_token(token=next_token)

        if next_token != self.api.token:
            self.api = self._reload_sofar_api(token=next_token)
            self.spotter_grid = self._load_spotter_grid()
            self.devices = self._load_spotter_devices()
        
    def get_spot_id(self, site_id:str, buoys_metadata:pd.DataFrame) -> str:
        return buoys_metadata.loc[site_id,"serial"]

    def _load_spotter_grid(self) -> list:
        return self.api.get_spotters()
    
    def _load_spotter_devices(self):
        return self.api.devices
    
    def select_spotter_obj_from_spotter_grid(self, spot_id:str, spotter_grid:list, devices:list) -> Spotter:
        for device in devices:
            if device["spotterId"] == spot_id:
                idx = devices.index(device)
        return spotter_grid[idx]
    
    def get_latest_available_datetime(self, spotter_obj) -> datetime:
        latest_available_datetime = spotter_obj.latest_data()["wave"]["timestamp"]
        return datetime.strptime(latest_available_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    def grab_raw_data(self, spotter_obj, include_bulk:bool=True, include_spectral:bool=False) -> pd.DataFrame:
        
        # TEMPORARY - Working with kwargs as possible approach
        # def function_x(argument1, argument2):
        #     return argument1 + argument2
        # kwargs = {"argument1":2, "argument2":2}
        # function_x(**kwargs) -> 4
        
        kwargs = {}

        if include_bulk:
            pass
        if include_spectral:
            pass

        data = spotter_obj.grab_data(**kwargs)

        return data