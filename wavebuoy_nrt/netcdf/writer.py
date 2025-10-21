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
from dotenv import load_dotenv

import wavebuoy_nrt.config as config
from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.netcdf.lookup import NetCDFFileHandler
from wavebuoy_nrt.config.config import NC_FILE_NAME_TEMPLATE, NC_SPECTRAL_FILE_NAME_TEMPLATE

load_dotenv()

SITE_LOGGER = logging.getLogger("site_logger")


class ncMetaDataLoader:
    def __init__(self, buoys_metadata: pd.DataFrame):
        self.buoys_metadata = buoys_metadata

    def load_buoys_metadata(self, buoy_type:str):
        try:
            file_path = os.path.join(os.getenv('IRDS_PATH'), "Data", "website", "auswaves", "buoys_metadata.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError("No such directory for buoys metadata: {}")
           
            buoys_metadata = pd.read_csv(file_path)
            
            buoys_metadata = buoys_metadata.loc[buoys_metadata["type"] == buoy_type]
            
            buoys_metadata["region"] = (buoys_metadata['archive_path']
                                    .str.extract(r'\\auswaves\\([a-z]+)waves')[0]
                                    .str.upper())
            
            buoys_metadata = buoys_metadata.set_index('name')
            
            name_constraint = "drift".upper()
            indexes = [index for index in buoys_metadata.index if name_constraint not in index.upper()]
            buoys_metadata = buoys_metadata.loc[indexes]
            
            return buoys_metadata

        except:
            error_message = "Loading and processing buoys_metadata.csv unsuccessful. Check if the file is corrupted or if its structure has been changed"

    def load_regional_metadata(self) -> pd.DataFrame:
        metadata_path = os.path.normpath(os.getenv("METADATA_PATH"))
        regional_metadata_path = os.path.join(metadata_path, "regional_metadata.csv")
        SITE_LOGGER.warning(regional_metadata_path)
        return pd.read_csv(regional_metadata_path)

    def _get_deployment_metadata_region_folders(self, site_name: str) -> list:
        region_folder = self.buoys_metadata.loc[site_name, "region"].lower()
        region_folder += "waves"
        return region_folder

    def _get_deployment_metadata_files(self, site_name: str, region_folder: str, file_extension: str = "*.xlsx") -> list:
        
        deployment_metadata_files_extension = file_extension
        site_name_corrected = site_name.replace("_","")
        files_path = os.path.join(os.getenv('IRDS_PATH'), "Data", region_folder, site_name_corrected)

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
        
        file_paths = self._validate_deployment_metadata_file_name(file_paths=file_paths)
        
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

        matches, not_matches, temp_files = [], [], []

        for file in file_paths:
            filename = os.path.basename(file)
            if filename.startswith("~$"):
                temp_files.append(file)
                continue  # Skip lock files entirely

            if template.search(filename):
                matches.append(file)
            else:
                not_matches.append(file)

        if not matches:
            error_message = "No deployment metadata files found."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)

        if temp_files:
            SITE_LOGGER.warning(
                f"The following deployment metadata sheets weren't closed properly and were ignored: {temp_files}"
            )

        if not_matches:
            SITE_LOGGER.warning(f"Files provided: {file_paths}")
            SITE_LOGGER.warning(f"Valid matches: {matches}")
            error_message = (
                "At least one of the deployment metadata file names does not conform "
                "to the expected template (metadata_{site_name}_deploy{YYYYmmdd}.xlsx). "
                "Make sure all of them conform."
            )
            SITE_LOGGER.error(error_message)
            raise NameError(error_message)

        return matches

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


class ncSpectralAttrsExtractor:
    def _extract_general_spectral_analysis_technique() -> str:
        return "Fast Fourier Transform"
    
    def _extract_general_spectral_analysis_technique_reference() -> str:
        return "Kuik, A. J., van Vledder, G. P., & Holthuijsen, L. H. (1988).\
                A Method for the Routine Analysis of Pitch-and-Roll Buoy Wave Data,\
                Journal of Physical Oceanography, 18(7), 1020-1034. \
                Retrieved Feb 21, 2022, from https://journals.ametsoc.org/view/journals/phoc/18/7/1520-0485_1988_018_1020_amftra_2_0_co_2.xml"
    

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
        site_name = deployment_metadata.loc["Site Name", "metadata_wave_buoy"]
        return re.sub(r'(?<!^)(?=[A-Z0-9])', '-', site_name).upper().replace(" ", "")#re.sub(r'\d+', '', site_name).strip()
    
    def  _extract_deployment_metadata_instrument(deployment_metadata: pd.DataFrame):
        return deployment_metadata.loc["Instrument", "metadata_wave_buoy"]

    def  _extract_deployment_metadata_transmission(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Transmission", "metadata_wave_buoy"]

    def _extract_deployment_metadata_hull_serial_number(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Hull serial number", "metadata_wave_buoy"]
        # return NetCDFFileHandler()._get_operating_institution(deployment_metadata=deployment_metadata)
    
    def _extract_deployment_metadata_water_depth(deployment_metadata: pd.DataFrame, drifter:bool = False) -> float:
        
        if drifter:
            return np.nan
        
        try:
            water_depth = deployment_metadata.loc["Water depth", "metadata_wave_buoy"]
        except KeyError:
            raise ValueError("Water depth metadata is missing") 

        if isinstance(water_depth, str):
            
            match = re.search(r'(\d+(\.\d+)?)', water_depth)
            if match:
                water_depth = match.group(1)
            else:
                raise ValueError("No numeric water depth found")

        try:
            return np.float32(water_depth)
        except (TypeError, ValueError):
            raise ValueError("Invalid water depth format")
    
    def _extract_deployment_metadata_water_depth_units(deployment_metadata: pd.DataFrame, drifter:bool = False) -> str:
        return np.nan if drifter else "m"
    
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

    def _extract_deployment_metadata_title(deployment_metadata: pd.DataFrame, regional_metadata: pd.DataFrame) -> str:
        base_title = """Near real time integral wave parameters from wave buoys collected by {operating_institution} using a {instrument} at {site_name}"""
        operating_institution = ncAttrsExtractor._extract_regional_metadata_institution(deployment_metadata=deployment_metadata, regional_metadata=regional_metadata)
        instrument = ncAttrsExtractor._extract_deployment_metadata_instrument(deployment_metadata=deployment_metadata)
        site_name = ncAttrsExtractor._extract_deployment_metadata_site_name(deployment_metadata=deployment_metadata)
        return base_title.format(operating_institution=operating_institution,
                                instrument=instrument,
                                site_name=site_name)

    def _extract_deployment_metadata_abstract(deployment_metadata: pd.DataFrame, regional_metadata:pd.DataFrame) -> str:
        return ncAttrsExtractor._extract_deployment_metadata_title(deployment_metadata=deployment_metadata, regional_metadata=regional_metadata)

    # from regional metadata -------------
    def _process_operating_institution(deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        if "IMOS" in operating_institution_code:
            return "IMOS"
        else:
            return operating_institution_code

    def _extract_regional_metadata_principal_investigator(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "principal_investigator"].values[0]

    def _extract_regional_metadata_principal_investigator_email(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "principal_investigator_email"].values[0]
    
    def _extract_regional_metadata_citation(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "citation"].values[0]
    
    def _extract_regional_metadata_acknowledgement(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        # SITE_LOGGER.warning(f"OPERATING INSTITUTION: {operating_institution}")
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "acknowledgement"].values[0]

    def _extract_regional_metadata_project(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "project"].values[0]

    def  _extract_regional_metadata_institution(regional_metadata: pd.DataFrame, deployment_metadata: pd.DataFrame) -> str:
        operating_institution_code = ncAttrsExtractor._process_operating_institution(deployment_metadata=deployment_metadata)
        return regional_metadata.loc[regional_metadata["operating_institution"]==operating_institution_code, "operating_institution_long_name"].values[0]
        # op_inst_code = {"UWA":"The University of Western Australia",
        #                   "Deakin":"Deakin University",
        #                   "NSW-DCCEEW" : "New South Wales Department of Climate Change, Energy, the Environment and Water",
        #                   "IMOS":"IMOS Coastal Wave Buoys",
        #                   "SARDI": "South Australian Research and Development Institute",
        #                   "SARDI-DEW": "TEST",
        #                   "SARDI-Flinders": "SARDI-Flinders University",
        #                   "DEW": "DEW"
        #                   }
        # operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        # if "IMOS" in operating_institution:
        #     return op_inst_code["IMOS"]
        # else:
        #     return op_inst_code[operating_institution]


    # generally pre-defined --------------------
    def _extract_data_history(dataset: xr.Dataset) -> str:
        # return f"this file was file created on: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M%SZ")}"
        return f"this file was file created on: {ncAttrsExtractor._extract_data_date_created(dataset=dataset)}"
    
    def _extract_general_author():
        return "Thiago Caminha"
    
    def _extract_general_author_email() -> str:
        return "thiago.caminha@uwa.edu.au"
    
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
    
    def _extract_general_license() -> str:
        return 'http://creativecommons.org/licenses/by/4.0/'
    
    def _extract_general_cdm_data_type(drifter:bool = False) -> str:
        return "Trajectory" if drifter else "Station"
    
    def _extract_general_platform(drifter:bool = False) -> str:
        return 'drifter surface buoy' if drifter else 'moored surface buoy'
    
    def _extract_general_references() -> str:
        return  'http://www.imos.org.au'
    
    def _extract_general_wave_buoy_type() -> str:
        return 'directional'
    
    def _extract_general_wave_motion_sensor_type() -> str:
        return "GPS"
    
    def _extract_general_disclaimer() -> str:
        return 'Data, products and services from IMOS are provided \\"as is\\" without any warranty as to fitness for a particular purpose.'


class ncAttrsComposer:
    def __init__(self, 
                 buoys_metadata: pd.DataFrame,
                 deployment_metadata: pd.DataFrame,
                 regional_metadata: pd.DataFrame,
                 parameters_type: str = "bulk"):
        
        self.parameters_type = parameters_type
        self.buoys_metadata = buoys_metadata
        self.deployment_metadata = deployment_metadata
        self.regional_metadata = regional_metadata
        self.attrs_templates_files = {"bulk": "bulk_attrs.json",
                                "spectral":"spectral_attrs.json"}
        self.attrs_template = (ncMetaDataLoader(buoys_metadata=buoys_metadata)
                              ._get_template_imos(file_name=self.attrs_templates_files[self.parameters_type])
                        )
    
    def _match_valid_min_max_dtype(self, variable:str, variables_attributes:dict , dataset: xr.Dataset):
        return (np.dtype(dataset[variable]).type(variables_attributes["valid_min"]),
                np.dtype(dataset[variable]).type(variables_attributes["valid_max"]))
    


    def assign_variables_attributes(self, dataset: xr.Dataset) -> xr.Dataset:
        variables = list(self.attrs_template['variables'].keys())
        # variables.remove("timeSeries")
        for variable in variables:
            if variable in list(dataset.variables):
                variables_attributes = self.attrs_template['variables'][variable]
                
                if "quality_control" in variable:
                    variables_attributes["flag_values"] = np.int8(variables_attributes["flag_values"])
                
                if "valid_min" in variables_attributes or variable not in ("TIME","timeSeries"):
                    variables_attributes["valid_min"], variables_attributes["valid_max"] = self._match_valid_min_max_dtype(
                                                                                variable=variable,
                                                                                dataset=dataset,
                                                                                variables_attributes=variables_attributes
                                                                            )
                    
                dataset[variable] = (dataset[variable].assign_attrs(variables_attributes))
        
        return dataset

    def assign_variables_attributes_dataset_objects(self, dataset_objects: xr.Dataset) -> xr.Dataset:
        for dataset in dataset_objects:
            dataset = self.assign_variables_attributes(dataset=dataset)

        return dataset_objects

    def assign_general_attributes(self, dataset: xr.Dataset, site_name: str, drifter:bool = False) -> xr.Dataset:
        
        # general_attributes = self.template_imos
        # # del general_attributes["_dimensions"]
        # # del general_attributes["_variables"]
        
        general_attributes  = self._compose_general_attributes(dataset=dataset, site_name=site_name, drifter=drifter)
        dataset = dataset.assign_attrs(general_attributes)

        return dataset
    
    def _compose_general_attributes(self, site_name: str, dataset: xr.Dataset, drifter=False) -> dict:
        
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
                        
                        if name.endswith("_title") or name.endswith("_abstract"):
                            kwargs.update({"regional_metadata":self.regional_metadata})
                    
                        if name.endswith(("_water_depth","_water_depth_units")):
                            kwargs.update({"drifter":drifter})

                    elif name.startswith("_extract_general_"):
                        key = name.removeprefix("_extract_general_") 
                        kwargs = {}
                        
                        if name.endswith(("_platform", "cdm_data_type")):
                            kwargs.update({"drifter":drifter})

                    elif name.startswith("_extract_regional_metadata_"):
                        key = name.removeprefix("_extract_regional_metadata_") 
                        kwargs = kwargs = {"regional_metadata": self.regional_metadata,
                                        "deployment_metadata":self.deployment_metadata}

                    try:                        
                        extracted = method(**kwargs)
                    except:
                        SITE_LOGGER.warning(f"grabing attribute from general_attrs.json for {key}")
                        extracted = self._get_attribute_from_template(attribute_name=key, attributes_template=self.attrs_template)
                    
                    general_attributes.update({key:extracted})

        if self.parameters_type == "spectral":
            # extracted = ncSpectralAttrsExtractor._extract_general_spectral_analysis_technique()
            for name in dir(ncSpectralAttrsExtractor): 
                if name.startswith("_extract_"): 
                    method = getattr(ncSpectralAttrsExtractor, name)
                    if callable(method):
                        if name.startswith("_extract_general_"):
                                key = name.removeprefix("_extract_general_") 
                                kwargs = {}

                        extracted = method(**kwargs)
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

    DTYPES_BULK = {'WSSH':{"dtype":np.float64},
                        'WPPE':{"dtype":np.float64},
                        'WPFM':{"dtype":np.float64},
                        'WPDI':{"dtype":np.float64},
                        'WPDS':{"dtype":np.float64},
                        'SSWMD':{"dtype":np.float64},
                        'WMDS':{"dtype":np.float64},
                        'TEMP':{"dtype":np.float64},
                        'WAVE_quality_control':{"dtype":np.int8},
                        'TEMP_quality_control':{"dtype":np.int8},
                        # 'TIME':{"dtype":np.float64},
                        'LATITUDE':{"dtype":np.float64},
                        'LONGITUDE':{"dtype":np.float64},
                        'timeSeries':{"dtype":np.int16}}\
                        
    
    DTYPES_SPECTRAL = {
                "FREQUENCY": {"dtype": np.float32},
                "LATITUDE": {"dtype": np.float64},
                "LONGITUDE": {"dtype": np.float64},
                "A1": {"dtype": np.float32},
                "B1": {"dtype": np.float32},
                "A2": {"dtype": np.float32},
                "B2": {"dtype": np.float32},
                "ENERGY": {"dtype": np.float32},
                'timeSeries':{"dtype":np.int16}
            }

    @staticmethod
    def select_processing_source(data: pd.DataFrame, processing_source : str) -> pd.DataFrame:
        return data[data["processing_source"] == processing_source]

    @staticmethod
    def _compose_coords_dimensions(data: pd.DataFrame, parameters_type: str = "bulk") -> dict:

        coords = {
                "TIME":("TIME", data["TIME"]),
                "LATITUDE":("TIME", data["LATITUDE"]),
                "LONGITUDE":("TIME", data["LONGITUDE"])
            }

        if parameters_type == "spectral":
            coords.update({"FREQUENCY": ("FREQUENCY", data["FREQUENCY"].iloc[-1])})

        return coords 

    @staticmethod
    def _compose_data_vars(data: pd.DataFrame, parameters_type: str = "bulk") -> dict:
      
        data_vars = {}
        cols_to_drop = ['TIME', 'timeSeries', 'LATITUDE', 'LONGITUDE', "processing_source"]
        dimensions = ["TIME"]

        if parameters_type == "spectral":
            cols_to_drop.extend(["FREQUENCY", "DIFFREQUENCY", "DIRECTION", "DIRSPREAD"])
            dimensions.append("FREQUENCY")

        vars_to_include = data.drop(columns=cols_to_drop).columns

        for var in vars_to_include:
            if parameters_type == "bulk":
                data_vars.update({var:(tuple(dimensions), data[var])})
            elif parameters_type == "spectral":
                data_vars.update({var:(tuple(dimensions), np.vstack(data[var].values))})
        
        return data_vars
    
    @staticmethod
    def compose_dataset(data: pd.DataFrame, parameters_type: str = "bulk") -> xr.Dataset:
        
        coords = ncProcessor._compose_coords_dimensions(data=data,parameters_type=parameters_type)
        data_vars = ncProcessor._compose_data_vars(data=data, parameters_type=parameters_type)
        
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
        dataset["timeSeries"] = [np.int16(1)]
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

    @staticmethod
    def convert_dtypes(dataset:tuple, parameters_type:str = "bulk") -> xr.Dataset:
        
        if parameters_type == 'bulk':
            dtypes_dict = ncProcessor.DTYPES_BULK
        elif parameters_type == 'spectral':
            dtypes_dict = ncProcessor.DTYPES_SPECTRAL
        else:
            raise ValueError("Invalid parameters_type. Choose 'bulk' or 'spectral'.")

        dataset = dataset.copy()

        for var_name, target_dtype in dtypes_dict.items():
            if var_name in dataset:
                dataset[var_name] = dataset[var_name].astype(target_dtype["dtype"])

        SITE_LOGGER.warning(dataset.dtypes)

        return dataset
    
    @staticmethod
    def convert_dtypes_dataset_objects(dataset_objects: tuple, parameters_type:str = "bulk") -> List[xr.Dataset]:
        
        if parameters_type == 'bulk':
            dtypes_dict = ncProcessor.DTYPES_BULK
        elif parameters_type == 'spectral':
            dtypes_dict = ncProcessor.DTYPES_SPECTRAL

        for dataset in dataset_objects:
            dataset = dataset.map(lambda x: x.astype(dtypes_dict[x.name]["dtype"]) if x.name in dtypes_dict else x )
            SITE_LOGGER.warning(dataset.dtypes)
        return dataset_objects

    def convert_drifter_name(site_name: str) -> str:

        site_lower = site_name.lower()
        if "drift" not in site_lower:
            return site_name, False

        match = re.match(r"^TIDE_SouthAfricaDrifting(\d+)$", site_name)
        if match:
            number = match.group(1)
            return f"UWA-Drifter_{number}", True

        return site_name, True


class ncWriter(WaveBuoy):

    ENCODING_ENFORCEMENT_BULK = {"TIME":{"_FillValue":None},
                            'WSSH':{"dtype":np.float64},
                            'WPPE':{"dtype":np.float64},
                            'WPFM':{"dtype":np.float64},
                            'WPDI':{"dtype":np.float64},
                            'WPDS':{"dtype":np.float64},
                            'SSWMD':{"dtype":np.float64},
                            'WMDS':{"dtype":np.float64},
                            'TEMP':{"dtype":np.float64},
                            'WAVE_quality_control':{"dtype":np.int8},
                            'TEMP_quality_control':{"dtype":np.int8},
                            'TIME':{"dtype":np.float64},
                            'LATITUDE':{"dtype":np.float64},
                            'LONGITUDE':{"dtype":np.float64},
                            'timeSeries':{"dtype":np.int16}
                    }

    ENCODING_ENFORCEMENT_SPECTRAL = {
                "TIME": {"dtype": np.float64, "_FillValue":None},
                "FREQUENCY": {"dtype": np.float32, "_FillValue":None},
                "LATITUDE": {"dtype": np.float64},
                "LONGITUDE": {"dtype": np.float64},
                "A1": {"dtype": np.float32},
                "B1": {"dtype": np.float32},
                "A2": {"dtype": np.float32},
                "B2": {"dtype": np.float32},
                "ENERGY": {"dtype": np.float32},
                'timeSeries':{"dtype":np.int16}
            }


    def __init__(self, buoy_type):
        super().__init__(buoy_type=buoy_type)

    def generate_nc_output_paths(self, dataframe: pd.DataFrame) -> list:
        nc_output_paths = []
        return nc_output_paths

    def _get_operating_institution(self, deployment_metadata: pd.DataFrame, regional_metadata: pd.DataFrame) -> str:
        operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        print(operating_institution)
        if "IMOS" in operating_institution:
            return "IMOS_COASTAL-WAVE-BUOYS"
        elif "IMOS" not in operating_institution:
            try:
                # operating_institution = OPERATING_INSTITUTIONS[operating_institution]
                return (regional_metadata
                        .loc[regional_metadata["operating_institution"] == operating_institution,"operating_institution_nc_preffix"]
                        .values[0]
                    )

            except:
                raise ValueError(f"{operating_institution} not valid. Please make sure the operating institution code is valid in the deployment metadata file")


    def _format_periods(self, periods: PeriodIndex) -> PeriodIndex:
        day = "01"
        return periods.strftime("%Y%m") + day
        
    # def _format_site_id_to_filename(self, site_id:str) -> str:
    #     return re.sub(r'(?<!^)(?=[A-Z0-9])', '-', site_id)
    def _format_site_id_to_filename(self, site_id: str) -> str:
        site_id = re.sub(r'([a-z])([A-Z])', r'\1-\2', site_id)   # lower→UPPER boundary
        site_id = re.sub(r'([A-Za-z])(\d+)', r'\1-\2', site_id)
        return site_id.upper()

    def compose_file_names(self,
                            site_id: str,
                            deployment_metadata: pd.DataFrame,
                            regional_metadata: pd.DataFrame,
                            periods: PeriodIndex,
                            parameters_type: str = "bulk") -> List[str]:
        
        if parameters_type == "bulk":
            file_name_template = NC_FILE_NAME_TEMPLATE
        elif parameters_type == "spectral":
            file_name_template = NC_SPECTRAL_FILE_NAME_TEMPLATE
        
        periods_formated = self._format_periods(periods=periods)
        file_names = []

        operating_institution = self._get_operating_institution(deployment_metadata=deployment_metadata,
                                                                regional_metadata=regional_metadata)
        site_id = re.sub(r'_+', '', site_id).strip()

        for period in periods_formated:
            file_name = file_name_template.format(operating_institution=operating_institution,
                                                     site_id=self._format_site_id_to_filename(site_id),
                                                     monthly_datetime=period
                                                )
            file_names.append(file_name)
        return file_names
    
    def compose_file_names_processing_source(self, file_names: list, processing_source: str) -> list:
        return [file_name.replace(".nc", f"_{processing_source}.nc") 
                    for file_name in file_names]

    def _compose_file_paths(self,
                            site_id: str,
                            file_names: list,
                            output_path: str,
                            stage: str = "production") -> list:

        output_path = os.path.join(output_path, "sites")        
        site_id_processed = site_id.replace("_","")

        if stage == "production":
            return [os.path.join(output_path, site_id_processed, file_name) for file_name in file_names]
        
        elif stage == "backup":
            backup_path = os.path.join(output_path, site_id_processed, "backup_files")
            if not os.path.isdir(backup_path):
                os.makedirs(backup_path)
            return [os.path.join(backup_path, file_name) for file_name in file_names]
            
    
    def _remove_coordinates_qc_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        qc_variables = [var for var in list(dataset.variables.keys()) if var.endswith("quality_control")]
        for qc_var in qc_variables:
            dataset[qc_var].encoding["coordinates"] = None
        return dataset

    def _remove_fillvalue_attributes(self, dataset: xr.Dataset) -> xr.Dataset:
        time_variables = [var for var in list(dataset.variables.keys()) if var.startswith("TIME")]
        for time_var in time_variables:
            dataset[time_var].encoding["_FillValue"] = None

        if "FREQUENCY" in list(dataset.variables.keys()):
            dataset["FREQUENCY"].encoding["_FillValue"] = None
        
        return dataset

    def _process_encoding(self, dataset: xr.Dataset, parameters_type: str) -> dict:
                
        if parameters_type == "bulk":
            encoding = self.ENCODING_ENFORCEMENT_BULK.copy()
            if "TEMP" not in list(dataset.variables):
                del encoding["TEMP"]
                del encoding["TEMP_quality_control"]

        elif parameters_type == "spectral":
            encoding = self.ENCODING_ENFORCEMENT_SPECTRAL.copy()
        SITE_LOGGER.warning(encoding)
        return encoding

    
    def _is_file_locked(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        
        import platform
        system = platform.system()

        if system == "Windows":
            try:
                import msvcrt
                with open(file_path, 'a') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                return False
            except OSError:
                return True

        elif system in ("Linux", "Darwin"):  # Darwin = macOS
            try:
                import fcntl
                with open(file_path, 'a') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return False
            except (OSError, IOError):
                return True

        return False
    
    def save_nc_file(self, 
                     site_id: str,
                     output_path: str,
                     file_names: str,
                     dataset_objects: xr.Dataset,
                     parameters_type: str = "bulk"):
        file_paths = self._compose_file_paths(site_id=site_id,
                                                output_path=output_path,
                                                file_names=file_names)       
        backup_file_paths = self._compose_file_paths(site_id=site_id,
                                                        output_path=output_path,
                                                        file_names=file_names,
                                                        stage="backup") 
        
        for file_path, backup_file_path, dataset in zip(file_paths, backup_file_paths, dataset_objects):
            dataset = self._remove_coordinates_qc_variables(dataset=dataset)
            dataset = self._remove_fillvalue_attributes(dataset=dataset)

            encoding = self._process_encoding(dataset=dataset, parameters_type=parameters_type)
            
            if not self._is_file_locked(file_path):
                dataset.to_netcdf(file_path, engine="netcdf4",
                                    encoding=encoding)
                
            else:
                dataset.to_netcdf(backup_file_path, engine="netcdf4",
                                    encoding=encoding)
                raise RuntimeError(f"File was locked, saving to {backup_file_path}")
                



    

    