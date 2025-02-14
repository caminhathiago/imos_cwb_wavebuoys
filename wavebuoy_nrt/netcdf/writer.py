import os
import logging
import json
import re
from datetime import datetime, timezone
from typing import List, Tuple

from netCDF4 import date2num
import xarray as xr
import pandas as pd
from pandas.core.indexes.period import PeriodIndex
import numpy as np
import glob

import wavebuoy_nrt.config as config
from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.config.config import NC_FILE_NAME_TEMPLATE, IRDS_PATH, OPERATING_INSTITUTIONS



SITE_LOGGER = logging.getLogger("site_logger")


class ncMetaDataLoader:
    def __init__(self, buoys_metadata: pd.DataFrame):
        self.buoys_metadata = buoys_metadata

    def _get_deployment_metadata_region_folders(self, site_name: str) -> list:
        region_folder = self.buoys_metadata.loc[site_name, "region"].lower()
        region_folder += "waves"
        return region_folder

    def _get_deployment_metadata_files(self, site_name: str, region_folder: str, file_extension: str = "*.xlsx") -> list:
        
        deployment_metadata_files_extension = file_extension
        site_name_corrected = site_name.replace("_","")
        files_path = os.path.join(IRDS_PATH, "Data", region_folder, site_name_corrected)

        if os.path.exists(files_path):
            if os.path.exists(os.path.join(files_path, "metadata")):
                metadata_folder = "metadata"
            elif os.path.exists(os.path.join(files_path, "AODN_metadata")):
                SITE_LOGGER.warning(f"deployment metadata path for {site_name} is currently named as /AODN_metadata; rename it to /metadata")
                metadata_folder = "AODN_metadata"
            else:
                SITE_LOGGER.error(f"metadata folder for {site_name} not found. Please make sure it exists named as metadata and that it contains the relevant deployment metadata files.")
        else:
            SITE_LOGGER.error(f"folder for {site_name} not found. Please make sure it exists and has the same name as in buoys_metadata.")

        files = glob.glob(os.path.join(files_path, metadata_folder, deployment_metadata_files_extension))

        if files:
            return files
        else:
            error_message = f"No deployment metadata files provided for {site_name}. Please make sure at least the most recent one exists and matches the correct file naming standar."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)

    def _get_latest_deployment_metadata(self, file_paths: list) -> list:
        
        self._validate_deployment_metadata_file_name(file_paths=file_paths)
        
        file_paths.sort(key=os.path.getctime)
        latest_created_file = file_paths[-1]
        
        try:
            date_pattern = re.compile(r"(\d{8}).xlsx")
            latest_date_file = max(file_paths, key=lambda x: int(date_pattern.search(x).group(1)))
        except:
            SITE_LOGGER.warning("deployment metadata file date is set as YYYYmm. Try to include day of deployment.")
            date_pattern = re.compile(r"(\d{6}).xlsx")
            latest_date_file = max(file_paths, key=lambda x: int(date_pattern.search(x).group(1)))
        
        if latest_created_file != latest_date_file:
            SITE_LOGGER.warning("latest created deployment metadata file different from latest date.")
       
        return latest_date_file

    def _validate_deployment_metadata_file_name(self, file_paths: list):
        template = re.compile(r"metadata_[A-Za-z0-9]+_deploy(\d+).xlsx")

        matches = [file for file in file_paths if template.search(file)]
        if len(matches) < len(file_paths):
            error_message = "at least one of the deployment metadata files name is not conforming with the template: metadata_{site_name}_deploy{deployment_date}.xlsx. Make sure all of them are conforming."
            SITE_LOGGER.error(error_message)
            raise NameError(error_message)            

    def load_latest_deployment_metadata(self, site_name:str) -> pd.DataFrame:
        region_folder = self._get_deployment_metadata_region_folders(site_name=site_name)
        file_paths = self._get_deployment_metadata_files(site_name=site_name, region_folder=region_folder)
        file_path = self._get_latest_deployment_metadata(file_paths=file_paths)
        
        deployment_metadata = pd.read_excel(file_path)
        metadata_wave_buoy_col = deployment_metadata.filter(regex="Metadata").columns
        deployment_metadata = deployment_metadata.rename(columns={metadata_wave_buoy_col[0]:"metadata_wave_buoy",
                                                                  "Parameter":"parameter"})
        deployment_metadata = deployment_metadata.set_index("parameter")

        return deployment_metadata

    @staticmethod
    def load_deployment_metadata(site_name:str) -> pd.DataFrame:
        deployment_metadata_path = "\\\\drive.irds.uwa.edu.au\\OGS-COD-001\\CUTTLER_wawaves\\Data\\wawaves\\Hillarys\\metadata\\Hillarys_dep08_20240703.xlsx"
        deployment_metadata = pd.read_excel(deployment_metadata_path)
        
        metadata_wave_buoy_col = deployment_metadata.filter(regex="Metadata").columns
        deployment_metadata = deployment_metadata.rename(columns={metadata_wave_buoy_col[0]:"metadata_wave_buoy",
                                                                  "Parameter":"parameter"})
        deployment_metadata = deployment_metadata.set_index("parameter")
        return deployment_metadata
    
    @staticmethod
    def _get_template_imos(file_name: str) -> dict:
        file_path = os.path.join(os.path.dirname(config.__file__), file_name)
        with open(file_path) as j:
            return json.load(j)

class ncGeneralAttrs:
    def __init__():
        return

class ncBulkAttrs:
    def __init__():
        return
    
class ncSpectralAttrs:
    def __init__():
        return

class ncAttrsExtractor:
    # def __init__(self, buoys_metadata: pd.DataFrame):
        # self.deployment_metadata = metaDataLoader(buoys_metadata=buoys_metadata)._load_deployment_metadata()

    # from the data itself -------------
    def _extract_data_time_coverage_start(dataset: xr.Dataset) -> str:
        return pd.to_datetime(dataset["TIME"].min().values).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def _extract_data_time_coverage_end(dataset: xr.Dataset) -> str:
        return pd.to_datetime(dataset["TIME"].max().values).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def _extract_data_geospatial_lat_min(dataset: xr.Dataset) -> str:
        return float(dataset["LATITUDE"].min().values)
    
    def _extract_data_geospatial_lat_max(dataset: xr.Dataset) -> str:
        return float(dataset["LATITUDE"].max().values)
    
    def _extract_data_geospatial_lon_min(dataset: xr.Dataset) -> str:
        return float(dataset["LONGITUDE"].min().values)
    
    def _extract_data_geospatial_lon_max(dataset: xr.Dataset) -> str:
        return float(dataset["LONGITUDE"].max().values)

    def _extract_data_geospatial_lat_units(dataset: xr.Dataset) -> str:
        return "degrees_north"

    def _extract_data_geospatial_lon_units(dataset: xr.Dataset) -> str:
        return "degrees_east"
    
    def _extract_data_date_created(dataset: xr.Dataset) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # from buoys metadata ------------------
    

    # from deployment metadata -------------
    def _extract_deployment_metadata_site_name(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Site Name", "metadata_wave_buoy"]
    
    def  _extract_deployment_metadata_instrument(deployment_metadata: pd.DataFrame):
        return deployment_metadata.loc["Instrument", "metadata_wave_buoy"]

    def  _extract_deployment_metadata_transmission(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Transmission", "metadata_wave_buoy"]

    def _extract_deployment_metadata_hull_serial_number(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Hull serial number", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_institution(deployment_metadata: pd.DataFrame) -> str:
        return NetCDFFileHandler()._get_operating_institution(deployment_metadata=deployment_metadata)
    
    def _extract_deployment_metadata_water_depth(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Water depth", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_water_depth_units(deployment_metadata: pd.DataFrame) -> str:
        return "m"

    def _extract_deployment_metadata_principal_investigator(deployment_metadata: pd.DataFrame) -> str:
        return "PRINCIPAL_INVESTIGATOR_TEST"

    def _extract_deployment_metadata_principal_investigator_email(deployment_metadata: pd.DataFrame) -> str:
        return "PRINCIPAL_INVESTIGATOR_EMAIL_TEST"

    def _extract_deployment_metadata_project(deployment_metadata: pd.DataFrame) -> str:
        project = "IMOS"
        
        return project
    
    def _extract_deployment_metadata_instrument_burst_duration(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst duration", "metadata_wave_buoy"]

    def _extract_deployment_metadata_instrument_burst_interval(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst interval", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_instrument_burst_units(deployment_metadata: pd.DataFrame) -> str:
        return "s"
    
    def _extract_deployment_metadata_instrument_sampling_interval(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument sampling interval", "metadata_wave_buoy"]

    def _extract_deployment_metadata_spotter_id(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Spotter_id ", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_wave_sensor_serial_number(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Wave sensor serial number", "metadata_wave_buoy"]

    def _extract_deployment_metadata_title(deployment_metadata: pd.DataFrame) -> str:
        base_title = """Near real time integral wave parameters from wave buoys collected by {operating_institution} using a {instrument} at {site_name}"""
        operating_institution = ncAttrsExtractor._extract_deployment_metadata_institution(deployment_metadata=deployment_metadata)
        instrument = ncAttrsExtractor._extract_deployment_metadata_instrument(deployment_metadata=deployment_metadata)
        site_name = ncAttrsExtractor._extract_deployment_metadata_site_name(deployment_metadata=deployment_metadata)
        return base_title.format(operating_institution=operating_institution,
                                instrument=instrument,
                                site_name=site_name)

    def _extract_deployment_metadata_abstract(deployment_metadata: pd.DataFrame) -> str:
        return ncAttrsExtractor._extract_deployment_metadata_title(deployment_metadata=deployment_metadata)

    # generally pre-defined --------------------
    def _extract_data_history(dataset: xr.Dataset) -> str:
        # return f"this file was file created on: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M%SZ")}"
        return f"this file was file created on: {ncAttrsExtractor._extract_data_date_created(dataset=dataset)}"
    
    def _extract_general_author():
        return "AODN"
    
    def _extract_general_author_email() -> str:
        return "info@aodn.org.au"
    
    def _extract_general_data_centre() -> str:
        return "Australian Ocean Data Network (AODN)"

    def _extract_general_data_centre_email() -> str:
        return "info@aodn.org.au"

    def _extract_general_Conventions() -> str:
        return 'CF-1.6'

    def _extract_general_standard_name_vocabulary() -> str:
        return "NetCDF Climate and Forecast CF Standard Name Table Version 78"
    
    def _extract_general_naming_authority() -> str:
        naming_authority = "IMOS"
        
        return naming_authority
    
    def _extract_general_citation() -> str:
        return "_general_citation".upper()
    
    def _extract_general_acknowledgement() -> str:
        return "_general_acknowledgement".upper()
    
    def _extract_general_disclaimer() -> str:
        return "_general_disclaimer".upper()
    
    def _extract_general_license() -> str:
        return 'http://creativecommons.org/licenses/by/4.0/'
    
    def _extract_general_cdm_data_type() -> str:
        return "Station"
    
    def _extract_general_platform() -> str:
        return 'moored surface buoy'
    
    def _extract_general_references() -> str:
        return  'http://www.imos.org.au'
    
    def _extract_general_wave_buoy_type() -> str:
        return 'directional'
    
    def _extract_general_wave_motion_sensor_type() -> str:
        return "GPS"
    
    

class ncAttrsComposer:
    def __init__(self, buoys_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame):
        self.buoys_metadata = buoys_metadata
        self.deployment_metadata = deployment_metadata
        self.template_imos = ncMetaDataLoader(buoys_metadata=buoys_metadata)._get_template_imos(file_name="general_attrs.json")
        
    def assign_variables_attributes(self, dataset: xr.Dataset) -> xr.Dataset:
        variables = list(self.template_imos['variables'].keys())
        # variables.remove("timeSeries")
        for variable in variables:
            
            if variable in list(dataset.variables):
                variables_attributes = self.template_imos['variables'][variable]
                
                # for var_attr in variables_attributes:
                #     if var_attr.startswith("_"):
                #         del variables_attributes[var_attr] 
                # print(variables_attributes)
                # print("===================")
                dataset[variable] = (dataset[variable].assign_attrs(variables_attributes))
        
        return dataset

    def assign_variables_attributes_dataset_objects(self, dataset_objects: xr.Dataset) -> xr.Dataset:
        for dataset in dataset_objects:
            dataset = self.assign_variables_attributes(dataset=dataset)

        return dataset_objects

    def assign_general_attributes(self, dataset: xr.Dataset, site_name: str) -> xr.Dataset:
        
        # general_attributes = self.template_imos
        # # del general_attributes["_dimensions"]
        # # del general_attributes["_variables"]
        
        general_attributes  = self._compose_general_attributes(dataset=dataset, site_name=site_name)
        dataset = dataset.assign_attrs(general_attributes)

        return dataset
    
    def _compose_general_attributes(self, site_name: str, dataset: xr.Dataset) -> dict:
        
        general_attributes = {}

        for name in dir (ncAttrsExtractor): 
        
            if name.startswith("_extract_"): 
                method = getattr(ncAttrsExtractor, name)
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
                        kwargs = {"deployment_metadata":self.deployment_metadata}
                    
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

    

    def _compose_abstract(self, institution: str, instrument: str, site_name:str) -> str:
        base_abstact = """Near real time integral wave parameters from wave buoys collected by {institution} using a {instrument} at {site_name}"""
        return base_abstact.format(institution=institution,
                                instrument=instrument,
                                site_name=site_name)

class ncProcessor:

    @staticmethod
    def select_processing_source(data: pd.DataFrame, processing_source : str) -> pd.DataFrame:
        return data[data["processing_source"] == processing_source]

    @staticmethod
    def _compose_coords_dimensions(data: pd.DataFrame) -> dict:
        # hdr = self.select_processing_source(data=data, processing_source="hdr")
        # embedded = self.select_processing_source(data=data, processing_source="embedded")

        # times = [hdr["TIME"].to_list(),embedded["TIME"].to_list()]
        # latitudes = [hdr["TIME"].to_list(),embedded["TIME"].to_list()]
        # longitudes = [hdr["TIME"].to_list(),embedded["TIME"].to_list()]
        # timeSeries = [hdr["TIME"].to_list(),embedded["TIME"].to_list()]

        # return {
        #     "processing_source":["hdr", "embedded"],
        #     "TIME": (["processing_source", "TIME"], times),
        #     "LATITUDE": (["processing_source", "TIME"], latitudes),
        #     "LONGITUDE": (["processing_source", "TIME"], longitudes),
        #     "timeSeries": (["processing_source", "TIME"], timeSeries),

        # }

        # return {
        #     "processing_source":("processing_source",["hdr", "embedded"]),
        #     "TIME":("TIME", [hdr["TIME"], embedded["TIME"]], {"nested":True}),
        #     "timeSeries":("timeSeries", [hdr["timeSeries"], embedded["timeSeries"]], {"nested":True}),
        #     "LATITUDE":("LATITUDE", [hdr["LATITUDE"], embedded["LATITUDE"]], {"nested":True}),
        #     "LONGITUDE":("LONGITUDE", [hdr["LONGITUDE"], embedded["LONGITUDE"]], {"nested":True}),
        # }
        
        # return {
        #     "processing_source":("processing_source", ["hdr"]),
        #     "TIME":("TIME", data["TIME"]),
        #     "timeSeries":("timeSeries", data["timeSeries"]),
        #     "LATITUDE":("TIME", data["LATITUDE"]),
        #     "LONGITUDE":("TIME", data["LONGITUDE"])
        # }
    
        return {
            "TIME":("TIME", data["TIME"]),
            # "timeSeries":("timeSeries", data["timeSeries"]),
            "LATITUDE":("TIME", data["LATITUDE"]),
            "LONGITUDE":("TIME", data["LONGITUDE"])
        }

    @staticmethod
    def _compose_data_vars(data: pd.DataFrame, dimensions: list) -> dict:

        # hdr = self.select_processing_source(data=data, processing_source="hdr")
        # embedded = self.select_processing_source(data=data, processing_source="embedded")

        # data_vars = {}
        # vars_to_include = data.drop(columns=['processing_source','TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE']).columns
        # for var in vars_to_include:
        #     data_list = [hdr[var].to_list(),embedded[var].to_list()]
        #     data_vars.update({var:(dimensions, data_list)})
        # return data_vars 

        # hdr = self.select_processing_source(data=data, processing_source="hdr")
        # embedded = self.select_processing_source(data=data, processing_source="embedded")
        
        # data_vars = {}
        # vars_to_include = data.drop(columns=['processing_source','TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE']).columns
        # for var in vars_to_include:
        #     data_vars.update({var:(dimensions, [hdr[var],embedded[var]])})
        # return data_vars  
        
        data_vars = {}
        vars_to_include = data.drop(columns=['TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE', "processing_source"]).columns
        for var in vars_to_include:
            data_vars.update({var:(dimensions, data[var])})
        return data_vars  
    
    @staticmethod
    def compose_dataset(data: pd.DataFrame) -> xr.Dataset:
        
        coords = ncProcessor._compose_coords_dimensions(data=data)
        data_vars = ncProcessor._compose_data_vars(data=data, dimensions=["TIME"])
        
        dataset = xr.Dataset(coords=coords, data_vars=data_vars)

        # dataset = self._assign_attributes_variables(dataset=dataset)
        # dataset = self._assign_general_attributes(dataset=dataset)

        return dataset

    @staticmethod
    def select_processing_source(dataset: xr.Dataset, processing_source: str = "hdr") -> xr.Dataset:
        subset_variables = list(dataset.keys()) 
        subset_variables = [variable for variable in subset_variables 
                            if not variable.endswith("_test") 
                                and variable != "check"]
        
        return (dataset
                .sel(processing_source=processing_source)
                .dropna("TIME", how="all", subset=subset_variables)
                .drop_vars("processing_source")
        )

    @staticmethod
    def create_timeseries_variable(dataset: xr.Dataset) -> xr.Dataset:
        dataset["timeSeries"] = [np.int64(1)]
        return dataset

    @staticmethod
    def combine_datasets(dataset1: xr.Dataset, dataset2: xr.Dataset) -> xr.Dataset:
        return xr.concat([dataset1, dataset2], dim="processing_source")
    
    @staticmethod
    def assing_processing_source_as_coord(combined_dataset: xr.Dataset) -> xr.Dataset:
        processing_sources = np.array(["hdr", "embedded"], dtype=object)
        return combined_dataset.assign_coords(processing_source=("processing_source", processing_sources))

    @staticmethod
    def extract_monthly_periods_dataset(dataset: xr.Dataset) -> PeriodIndex:
        return pd.to_datetime(dataset["TIME"].data).to_period("M").unique()

    @staticmethod
    def split_dataset_monthly(dataset: xr.Dataset, periods: PeriodIndex) -> Tuple[xr.Dataset, ...]:
        dataset_objects = []
        for period in periods:
            print(period)
            monthly_dataset = dataset.sel(TIME=str(period))
            dataset_objects.append(monthly_dataset)
        return tuple(dataset_objects)

    @staticmethod
    def process_time_to_CF_convention(dataset_objects: tuple) -> List[xr.Dataset]:
        for dataset in dataset_objects:
            time = np.array(dataset["TIME"]
                            .to_dataframe()["TIME"]
                            .dt.to_pydatetime()
            )
            dataset["TIME"] = date2num(time,
                                "days since 1950-01-01 00:00:00 UTC",
                                "gregorian")
            
        return dataset_objects

class ncWriter(WaveBuoy):

    def __init__(self, buoy_type):
        super().__init__(buoy_type=buoy_type)

    def generate_nc_output_paths(self, dataframe: pd.DataFrame) -> list:
        nc_output_paths = []
        return nc_output_paths

    def _get_operating_institution(self, deployment_metadata: pd.DataFrame) -> str:
        operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        print(operating_institution)
        if "IMOS" in operating_institution:
            operating_institution = "IMOS_COASTAL-WAVE-BUOYS"
        elif "IMOS" not in operating_institution:
            try:
                operating_institution = OPERATING_INSTITUTIONS[operating_institution]
            except:
                raise ValueError(f"{operating_institution} not valid. Please make sure the operating institution code is valid in the deployment metadata file")

        return operating_institution

    # def _validate_operating_institution(self, deployment_metadata: pd.DataFrame):
    #     operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
    #     if not operating_institution in self.OPERATING_INSTITUTIONS.values():
    #         error_message = f""
        
    def _format_periods(self, periods: PeriodIndex) -> PeriodIndex:
        day = "01"
        return periods.strftime("%Y%m") + day
        

    def compose_file_names(self,
                            site_id: str,
                            deployment_metadata: pd.DataFrame,
                            periods: PeriodIndex) -> List[str]:
        periods_formated = self._format_periods(periods=periods)
        file_names = []

        operating_institution = self._get_operating_institution(deployment_metadata=deployment_metadata)

        for period in periods_formated:
            file_name = NC_FILE_NAME_TEMPLATE.format(operating_institution=operating_institution,
                                                     site_id=site_id,
                                                     monthly_datetime=period
                                                )
            file_names.append(file_name)
        return file_names
    
    def compose_file_names_processing_source(self, file_names: list, processing_source: str) -> list:
        return [file_name.replace(".nc", f"_{processing_source}.nc") 
                    for file_name in file_names]

    def _compose_file_paths(self, file_names: list, output_path: str) -> list:
        return [os.path.join(output_path, file_name) for file_name in file_names]

    def save_nc_file(self, 
                     output_path: str,
                     file_names: str,
                     dataset_objects: xr.Dataset):
        file_paths = self._compose_file_paths(output_path=output_path,
                                                 file_names=file_names)
        for file_path, dataset in zip(file_paths, dataset_objects):
            dataset.to_netcdf(file_path, engine="netcdf4")

        
    


    # Compose methods for each general attribute key ======================================================



    

    