from datetime import datetime, timedelta
import requests
import logging

from pysofar.sofar import SofarApi
from pysofar.spotter import Spotter
import pandas as pd

GENERAL_LOGGER = logging.getLogger("general_logger")
SITE_LOGGER = logging.getLogger("site_logger")

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
   
    def _get_wave_data(self,
                    spot_id: str,
                    token: str,
                    start_date: datetime = datetime.now() - timedelta(hours=24), 
                    end_date: datetime = datetime.now(),
                    add_wave_params: bool = True,
                    limit: int = 500,
                    include_waves: bool = True,
                    include_surface_temp_data: bool = True,
                    include_wind_data: bool = True,
                    include_frequency_data: bool = True,
                    include_directional_moments: bool = True,
                    include_partition_data: bool = True,
                    include_barometer_data: bool = True,
                    include_track: bool = False,
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

    def fetch_wave_data(self,
                            spot_id: str,
                            token: str,
                            start_date: datetime = datetime.now() - timedelta(hours=24), 
                            end_date: datetime = datetime.now(),
                            **kwargs) -> dict:
        page = 1
        current_start_date = start_date
        needs_pagination = True
        raw_data = None

        # if kwargs["include_waves"] and not kwargs["include_frequency_data"]:
        #     data_type = "waves"
        # elif not kwargs["include_waves"] and kwargs["include_frequency_data"]:
        #     data_type = "frequencyData"

        SITE_LOGGER.info("Starting Sofar API requests, paginating if needed:")
        while needs_pagination:
            SITE_LOGGER.info(f"Page: {page} | Start: {current_start_date} | End: {end_date}")
            new_raw_data = self._get_wave_data(
                spot_id=spot_id,
                token=token,
                start_date=current_start_date,
                end_date=end_date,
                **kwargs
            )
            
            if not new_raw_data["waves"]:
                    SITE_LOGGER.info(f"No more data available after Page: {page}")
                    break
            
            if page == 1:
                raw_data = new_raw_data
                
            else:
                raw_data = self._extend_raw_data(global_output=raw_data, current_page=new_raw_data)

            latest_extracted_time = datetime.strptime(new_raw_data["waves"][-1]["timestamp"],
                                                    "%Y-%m-%dT%H:%M:%S.%fZ")
            
            # Dealing with not owned spotters
            test_not_owned = self._test_not_owned_spotter(raw_data, new_raw_data)
            if test_not_owned:
                SITE_LOGGER.info(f"Not owned Spotter, pagination not necessary")
                break



            if latest_extracted_time >= end_date:
                SITE_LOGGER.info("End of API calls pagination.")
                needs_pagination = False
            else:
                current_start_date = latest_extracted_time
                page += 1
                print(page)

        return raw_data
    
    def _test_not_owned_spotter(self, raw_data: dict, new_raw_data: dict) -> bool:
        if raw_data is None or new_raw_data is None:
            return False  

        if "waves" not in raw_data or "waves" not in new_raw_data:
            return False

        if not raw_data["waves"] or not new_raw_data["waves"]:
            return False

        raw_data_times = (
            datetime.strptime(raw_data["waves"][0]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"),
            datetime.strptime(raw_data["waves"][-1]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
        )

        new_raw_data_times = (
            datetime.strptime(new_raw_data["waves"][0]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"),
            datetime.strptime(new_raw_data["waves"][-1]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
        )

        return raw_data_times == new_raw_data_times

    def _extend_raw_data(self, 
                        global_output: dict,
                        current_page: dict,
                        keys_to_extend: list = ['partitionData', 
                                                'wind',
                                                'surfaceTemp',
                                                'barometerData',
                                                'waves']
                    ) -> dict:

        for key in keys_to_extend:
            if key in global_output and current_page:
                global_output[key].extend(current_page[key])
        
        return global_output

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

    def get_latest_available_time(self, spot_id: str, token: str, dataset_type: str = "bulk") -> datetime:
      
        latest_data = self.get_latest_data(spot_id=spot_id, token=token)
        SITE_LOGGER.warning(latest_data)
        # return latest_data[dataset_type]
        if dataset_type == "bulk":
            parameters_type = "waves"
        elif dataset_type == "spectral":
            parameters_type = "frequencyData"

        SITE_LOGGER.warning("LATEST DATA:")
        SITE_LOGGER.warning(latest_data)


        if latest_data[parameters_type]:
            latest_available_time = latest_data[parameters_type][-1]["timestamp"]
        elif latest_data["track"]:
            latest_available_time = latest_data["track"][-1]["timestamp"]
        else:
            message = f"Latest data empty for {parameters_type} and track. Probably spotter is under a gap"
            SITE_LOGGER.warning(message)
            raise Exception(message)

        # try:
        #     latest_available_time = latest_data["waves"][-1]["timestamp"]
        # except:
        #     latest_available_time = latest_data["track"][-1]["timestamp"]

        latest_available_time = datetime.strptime(latest_available_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        return latest_available_time

    def _get_status_code(self, response: requests.Response) -> int:
        return response.status_code

    def _compose_query_parameters(self,
                                spot_id: str,
                                start_date: datetime = None, 
                                end_date: datetime = None,
                                add_wave_params: bool = True,
                                limit: int = 500,
                                include_waves: bool = True,
                                include_surface_temp_data: bool = True,
                                include_wind_data: bool = True,
                                include_frequency_data: bool = False,
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
                "limit": str(limit),
                "includeWaves": str(include_waves).lower(),
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
        

    def check_new_data(self, raw_data: dict, dataset_type: str = "waves") -> bool:
        return bool(raw_data and raw_data.get(dataset_type))

                


    
    # def get_wave_data2(self,
    #                 spot_id: str,
    #                 token: str,
    #                 start_date: datetime = datetime.now() - timedelta(hours=24), 
    #                 end_date: datetime = datetime.now(),
    #                 limit: int = 500,
    #                 add_wave_params: bool = True,
    #                 include_surface_temp_data: bool = True,
    #                 include_wind_data: bool = True,
    #                 include_frequency_data: bool = True,
    #                 include_directional_moments: bool = True,
    #                 include_partition_data: bool = True,
    #                 include_barometer_data: bool = True,
    #                 include_track: bool = True,
    #                 processing_sources="all"
    #                 )-> dict:
        
    #     kwargs_query_params = {
    #             key: value for key, value in locals().items() 
    #             if key not in ('self','token') and value is not None
    #         }        
    #     query_params = self._compose_query_parameters(**kwargs_query_params)
        
    #     request_url = self._compose_request_url(base_url=self._base_url, endpoint=self._endpoints["waves"])
    #     headers = self._compose_header(token=token)
        
    #     kwargs_request = {"url":request_url,
    #                       "headers": headers,
    #                       "params": query_params}
    #     page = 1
    #     try:
    #         response = requests.get(**kwargs_request)
    #         SITE_LOGGER(f"Page: {page} | Start: {start_date} | End: {end_date}")
    #     except Exception as e:
    #         SITE_LOGGER.error(str(e), exc_info=True)
    #         return

    #     raw_data = []
    #     latest_retrieved_time = response.json()["data"]["waves"][-1]["timestamp"]

    #     if latest_retrieved_time >= end_date:
    #         return response.json()["data"]
    #     else:
    #         page += 1
    #         raw_data.extend(response.json()["data"])
    #         SITE_LOGGER.info("API Calls pagination needed:")
    #         SITE_LOGGER.infor(f"Page: {page} | Start: {start_date} | End: {end_date}")
            
            
    #         start_date = end_date
    #         end_date += timedelta(days=30)
    #         pagination_data = self.get_wave_data(spot_id=spot_id,
    #                            token=token,
    #                            start_date=start_date,
    #                            end_date=end_date)
    #         raw_data

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
