import os
import logging
import json

import netCDF4
import xarray as xr
import pandas as pd

import wavebuoy_nrt.config as config
from wavebuoy_nrt.wavebuoy import WaveBuoy


SITE_LOGGER = logging.getLogger("site_logger")

class GeneralAttrs:
    def __init__():
        return

class BulkAttrs:
    def __init__():
        return
    
class SpectralAttrs:
    def __init__():
        return
    
class ncWriter(WaveBuoy):
    def __init__(self, buoy_type):
        super().__init__(buoy_type=buoy_type)
        self.template_imos = self._get_template_imos(file_name="general_attrs.json")
    
    def _get_template_imos(self, file_name: str) -> dict:
        file_path = os.path.join(os.path.dirname(config.__file__), file_name)
        with open(file_path) as j:
            return json.load(j)

    def process_df(self, data: pd.DataFrame) -> pd.DataFrame:
        # check if timeseries exists then create
        # 
        return

    def save_nc_file(self, file_path: str):
        return
    
    def _compose_coords_dimensions(self, data: pd.DataFrame) -> dict:
        return {
            "TIME":("TIME", data["TIME"]),
            "timeSeries":("timeSeries", data["timeSeries"]),
            "LATITUDE":("TIME", data["LATITUDE"]),
            "LONGITUDE":("TIME", data["LONGITUDE"])
        }
        
    def _compose_data_vars(self, data: pd.DataFrame, dimensions: list) -> dict:
        data_vars = {}
        vars_to_include = data.drop(columns=['TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE']).columns
        for var in vars_to_include:
            data_vars.update({var:(dimensions, data[var])})
        return data_vars  
    
    def compose_dataset(self, data: pd.DataFrame) -> xr.Dataset:
        
        coords = self._compose_coords_dimensions(data=data)
        data_vars = self._compose_data_vars(data=data, dimensions=["TIME"])
        
        dataset = xr.Dataset(coords=coords, data_vars=data_vars)

        dataset = self._assign_attributes_variables(dataset=dataset)
        dataset = self._assign_general_attributes(dataset=dataset)

        return dataset

    def _assign_attributes_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        for variable in self.template_imos['_variables'].keys():
            dataset[variable] = (dataset[variable]
                                 .assign_attrs(
                                     self.template_imos['_variables'][variable]
                                            )
                            )
        
        return dataset

    def _assign_general_attributes(self, dataset: xr.Dataset, site_name: str) -> xr.Dataset:
        
        general_attributes = self.template_imos
        del general_attributes["_dimensions"]
        del general_attributes["_variables"]
        
        generla_attributes  = self._compose_general_attributes(general_attributes=general_attributes)


        dataset = dataset.assign_attrs(general_attributes)

        return dataset
    
    def _compose_general_attributes(self,
                                    site_name: str,
                                    dataset: xr.Dataset) -> dict:
        deployment_metadata = self._load_deployment_metadata()
        
        general_attributes = {}

        for name in dir(self): 
        
            if name.startswith("_extract_"): 
                method = getattr(self, name)
                if callable(method):
                    if name.startswith("_extract_buoys_metadata_"):
                        key = name.removeprefix("_extract_buoys_metadata_")
                        kwargs = {"site_name": site_name,
                                  "buoys_metadata":self.buoys_metadata}
                         
                    elif name.startswith("_extract_data_"):
                        key = name.removeprefix("_extract_data_") 
                        kwargs = {"dataset":dataset}
                        
                    elif name.startswith("_extract_deployment_metadata_"):
                        key = name.removeprefix("_extract_deployment_metadata_") 
                        kwargs = {"deployment_metadata":deployment_metadata}
                    
                    elif name.startswith("_extract_general_"):
                        key = name.removeprefix("_extract_general_") 
                        kwargs = {}

                    try:                        
                        extracted = method(**kwargs)
                    except:
                        SITE_LOGGER.warning(f"grabing attribute from general_attrs.json for {key}")
                        extracted = self._get_attribute_from_template(attribute_name=key, attributes_template=self.template_imos)
                    
                    general_attributes.update({key:extracted})

        return general_attributes  

    def _get_attribute_from_template(self, attribute_name: str, attributes_template: dict) -> dict:
        return attributes_template[attribute_name]

    # Compose methods for each general attribute key ======================================================

    def _compose_title(self, institution: str, instrument: str, site_name:str) -> str:
        base_title = """Near real time integral wave parameters from wave buoys collected by {institution} using a {instrument} at {site_name}"""
        return base_title.format(institution=institution,
                                instrument=instrument,
                                site_name=site_name)

    def _compose_abstract(self, institution: str, instrument: str, site_name:str) -> str:
        base_abstact = """Near real time integral wave parameters from wave buoys collected by {institution} using a {instrument} at {site_name}"""
        return base_abstact.format(institution=institution,
                                instrument=instrument,
                                site_name=site_name)
    
    # from the data itself -------------
    def _extract_data_time_coverage_start(self, dataset: xr.Dataset) -> str:
        return str(dataset["TIME"].min().values)
    
    def _extract_data_time_coverage_end(self, dataset: xr.Dataset) -> str:
        return str(dataset["TIME"].max().values)
    
    def _extract_data_geospatial_lat_min(self, dataset: xr.Dataset) -> str:
        return float(dataset["LATITUDE"].min().values)
    
    def _extract_data_geospatial_lat_max(self, dataset: xr.Dataset) -> str:
        return float(dataset["LATITUDE"].max().values)
    
    def _extract_data_geospatial_lon_min(self, dataset: xr.Dataset) -> str:
        return float(dataset["LONGITUDE"].min().values)
    
    def _extract_data_geospatial_lon_max(self, dataset: xr.Dataset) -> str:
        return float(dataset["LONGITUDE"].max().values)

    def _extract_data_geospatial_lat_units(self, dataset: xr.Dataset) -> str:
        return "degrees_north"

    def _extract_data_geospatial_long_units(self, dataset: xr.Dataset) -> str:
        return "degrees_east"

    def _extract_data_history(self, dataset: xr.Dataset) -> str:
        return
    
    def _extract_data_date_created(self, dataset: xr.Dataset) -> str:
        return

    # from buoys metadata ------------------
    def _extract_buoys_metadata_site_name(self, site_name: str, buoys_metadata: pd.DataFrame) -> str:
        return buoys_metadata.loc[site_name].name
    
    def _extract_buoys_metadata_spot_id(self, site_name: str, buoys_metadata: pd.DataFrame) -> str:
        return buoys_metadata.loc[site_name].serial

    # from deployment metadata -------------
    def _load_deployment_metadata(self) -> pd.DataFrame:
        deployment_metadata_path = "\\\\drive.irds.uwa.edu.au\\OGS-COD-001\\CUTTLER_wawaves\\Data\\wawaves\\Hillarys\\metadata\\Hillarys_dep08_20240703.xlsx"
        deployment_metadata = pd.read_excel(deployment_metadata_path)
        
        metadata_wave_buoy_col = deployment_metadata.filter(regex="Metadata").columns
        deployment_metadata = deployment_metadata.rename(columns={metadata_wave_buoy_col[0]:"metadata_wave_buoy",
                                                                  "Parameter":"parameter"})
        deployment_metadata = deployment_metadata.set_index("parameter")
        return deployment_metadata
    
    def  _extract_deployment_metadata_instrument(self, deployment_metadata: pd.DataFrame):
        return deployment_metadata.loc["Instrument", "metadata_wave_buoy"]

    def  _extract_deployment_metadata_transmission(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Transmission", "metadata_wave_buoy"]

    def _extract_deployment_metadata_hull_serial_number(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Hull serial number", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_water_depth(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Water depth", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_water_depth_units(self, deployment_metadata: pd.DataFrame) -> str:
        return "m"

    def _extract_deployment_metadata_principal_investigator(self, deployment_metadata: pd.DataFrame) -> str:
        return

    def _extract_deployment_metadata_principal_investigator_email(self, deployment_metadata: pd.DataFrame) -> str:
        return

    def _extract_deployment_metadata_project(self, deployment_metadata: pd.DataFrame) -> str:
        return
    
    def _extract_deployment_metadata_instrument_burst_duration(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst duration", "metadata_wave_buoy"]

    def _extract_deployment_metadata_instrument_burst_interval(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst interval", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_instrument_burst_units(self, deployment_metadata: pd.DataFrame) -> str:
        return "s"
    
    def _extract_deployment_metadata_instrument_sampling_interval(self, deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument sampling interval", "metadata_wave_buoy"]

    # generally pre-defined --------------------
    def _extract_general_author(self):
        return
    
    def _extract_general_author_email(self) -> str:
        return
    
    def _extract_general_data_centre(self) -> str:
        return

    def _extract_general_data_centre_email(self) -> str:
        return

    def _extract_general_conventions(self) -> str:
        return

    def _extract_general_standard_name_vocabulary(self) -> str:
        return
    
    def _extract_general_naming_authority(self) -> str:
        return
    
    def _extract_general_citation(self) -> str:
        return
    
    def _extract_general_acknowledgement(self) -> str:
        return
    
    def _extract_general_disclaimer(self) -> str:
        return
    
    def _extract_general_license(self) -> str:
        return
    
    def _extract_general_cdm_data_type(self) -> str:
        return
    
    def _extract_general_platform(self) -> str:
        return
    
    def _extract_general_references(self) -> str:
        return  
    
    def _extract_general_wave_buoy_type(self) -> str:
        return
    
    def _extract_general_wave_motion_sensor_type(self) -> str:
        return
    
    def _extract_general_wave_sensor_serial_number(self) -> str:
        return

    

    