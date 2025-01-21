from datetime import datetime, timedelta
import requests

from pysofar.sofar import SofarApi
from pysofar.spotter import Spotter
import pandas as pd


class SofarAPI:
    def __init__(self, buoys_metadata: pd.DataFrame):
        self.buoys_metadata = buoys_metadata
        # self.api = self._initiate_sofar_api()
        # self.spotter_grid = self._load_spotter_grid()
        # self.devices = self._load_spotter_devices()

        self._base_url = "https://api.sofarocean.com/api"
        self._endpoints = {"waves": "/wave-data",
                           "latest": "/latest-data",
                           "smart_mooring": "/sensor-data",
                           "devices": "devices"}

    def _get_first_token(self):
        return (self.buoys_metadata
                        .sort_values("sofar_token")
                        .iloc[0]
                        .sofar_token
        )
        
    def _initiate_sofar_api(self):
        return SofarApi(custom_token=self._get_first_token())
    
    def _reload_sofar_api(self, token: str):
        return SofarApi(custom_token=token)

    def _validate_token(self, token: str) -> bool:
        if not isinstance(token, str) or len(token) != 30:
            raise ValueError("provide a valid token.")

    def check_token_iteration(self, next_token: str) -> bool:
        
        self._validate_token(token=next_token)

        if next_token != self.api.token:
            self.api = self._reload_sofar_api(token=next_token)
            self.spotter_grid = self._load_spotter_grid()
            self.devices = self._load_spotter_devices()
        
    def get_spot_id(self, site_id: str, buoys_metadata: pd.DataFrame) -> str:
        return buoys_metadata.loc[site_id,"serial"]

    def _load_spotter_grid(self) -> list:
        return self.api.get_spotters()
    
    def _load_spotter_devices(self):
        return self.api.devices
    
    def select_spotter_obj_from_spotter_grid(self, 
                                             spot_id: str, 
                                             spotter_grid: list,
                                             devices: list) -> Spotter:
        for device in devices:
            if device["spotterId"] == spot_id:
                idx = devices.index(device)
        return spotter_grid[idx]
    
   
    # ============================
    def _compose_header(self, token:str) -> dict:
        return {"token" : token}
    
    def grab_data(self, spot_id: str,
                     token: str,
                     query_params: dict = None,
                     data_type: str = "waves") -> dict:

        request_url = self._compose_request_url(base_url=self._base_url, endpoint=self._endpoints[data_type])
        headers = self._compose_header(token=token)

        kwargs = {"url":request_url, "headers": headers}
        if not data_type=="devices":
            kwargs.update({"params" : query_params})
        response = requests.get(**kwargs)

        if response.status_code == 200:
            return response.json()["data"]
        else:
            # ELABORATE ERROR HANDLING  
            print(f"Unsuccessfull API call, status {response.status_code}")
            return
   
    def get_wave_data(self,
                    spot_id: str,
                    token: str,
                    start_date: datetime = datetime.now() - timedelta(hours=24), 
                    end_date: datetime = datetime.now(),
                    add_wave_params: bool = True,
                    include_surface_temp_data: bool = True,
                    include_wind_data: bool = True,
                    include_frequency_data: bool = True,
                    include_directional_moments: bool = True,
                    include_partition_data: bool = True,
                    include_barometer_data: bool = True,
                    include_track: bool = True,
                    processing_sources="all"
                    )-> dict:
        
        kwargs_query_params = {
                key: value for key, value in locals().items() 
                if key not in ('self','token') and value is not None
            }        
        query_params = self._compose_query_parameters(**kwargs_query_params)
        
        request_url = self._compose_request_url(base_url=self._base_url, endpoint=self._endpoints["waves"])
        headers = self._compose_header(token=token)
        
        kwargs_request = {"url":request_url,
                          "headers": headers,
                          "params": query_params}

        response = requests.get(**kwargs_request)

        if response.status_code == 200:
            return response.json()["data"]
        else:
            # ELABORATE ERROR HANDLING  
            print(f"Unsuccessfull API call, status {response.status_code}")
            return

    def get_latest_data(self, spot_id: str, token: str) -> dict:
        
        request_url = self._compose_request_url(base_url=self._base_url,
                                               endpoint=self._endpoints["latest"])
        headers = self._compose_header(token=token)
        query_params = self._compose_query_parameters(spot_id=spot_id, add_wave_params=False)

        kwargs_request = {"url":request_url,
                          "headers": headers,
                          "params": query_params}

        response = requests.get(**kwargs_request)
        print(response.request.url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            # ELABORATE ERROR HANDLING  
            print(f"Unsuccessfull API call, status {response.status_code}")
            return

    def get_sensor_data(self,
                        spot_id:str,
                        token: str,
                        start_date: datetime = datetime.now() - timedelta(hours=24), 
                        end_date: datetime = datetime.now()
                        ) -> dict:
        
        request_url = self._compose_request_url(base_url=self._base_url,
                                               endpoint=self._endpoints["smart_mooring"])
        headers = self._compose_header(token=token)
        query_params = self._compose_query_parameters(spot_id=spot_id,
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      add_wave_params=False)

        kwargs_request = {"url":request_url,
                          "headers": headers,
                          "params": query_params}

        response = requests.get(**kwargs_request)
        print(response.request.url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            # ELABORATE ERROR HANDLING  
            print(f"Unsuccessfull API call, status {response.status_code}")
            return

    def get_latest_available_time(self, spot_id: str, token: str) -> datetime:
        """
        CONSIDER SMART MOORING
        """
        latest_data = self.get_latest_data(spot_id=spot_id, token=token)
        try:
            latest_available_time = latest_data["waves"][-1]["timestamp"]
        except:
            latest_available_time = latest_data["track"][-1]["timestamp"]
        
        return datetime.strptime(latest_available_time, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _get_status_code(self, response: requests.Response) -> int:
        return response.status_code

    def _compose_query_parameters(self,
                                spot_id: str,
                                start_date: datetime = None, 
                                end_date: datetime = None,
                                add_wave_params: bool = True,
                                include_surface_temp_data: bool = True,
                                include_wind_data: bool = True,
                                include_frequency_data: bool = True,
                                include_directional_moments: bool = True,
                                include_partition_data: bool = True,
                                include_barometer_data: bool = True,
                                include_track: bool = True,
                                processing_sources="all"):
        
        query_params = {"spotterId" : spot_id}
        
        if start_date and end_date:
            query_params.update({ 
                "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            })

        if add_wave_params:
            query_params.update({
                "includeSurfaceTempData": str(include_surface_temp_data).lower(),
                "includeWindData": str(include_wind_data).lower(),
                "includeFrequencyData": str(include_frequency_data).lower(),
                "includeDirectionalMoments": str(include_directional_moments).lower(),
                "includePartitionData": str(include_partition_data).lower(),
                "includeBarometerData": str(include_barometer_data).lower(),
                "includeTrack": str(include_track).lower(),
                "processingSources": processing_sources
            })
        
        return query_params
    
    def _compose_request_url(self, base_url: str, endpoint: str) -> str:
        return base_url + endpoint
        
    


 # def grab_latest_data(self, spot_id: str, token: str) -> dict:
        
    #     request_url = self.compose_request_url(base_url=self._base_url, endpoint=self._latest_data_endpoint)
    #     headers = self.compose_header(token=token)
    #     response = requests.get(request_url, headers=headers)

    #     if response.status_code == 200:
    #         return json.loads(response)
    #     else:
    #         # ELABORATE ERROR HANDLING  
    #         print(f"Unsuccessfull API call, status {response.status_code}")
    #         return

     # def get_latest_available_time2(self, spot_id: str, token: str) -> datetime:
        
    #     query_params = self.compose_query_parameters(spot_id=spot_id, add_wave_params=False)
    #     print(query_params)
    #     print(token)
    #     latest_data = self.request_api(spot_id=spot_id, token=token, data_type="latest", query_params=query_params)
        
    #     try:
    #         print("grabing from waves")
    #         print(latest_data)
    #         latest_time = latest_data["waves"]["timestamp"]
    #     except:
    #         print("grabing from track")
    #         latest_time = latest_data["track"][-1]["timestamp"]

    #     return datetime.strptime(latest_time, "%Y-%m-%dT%H:%M:%S.%fZ")
