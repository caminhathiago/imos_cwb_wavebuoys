import os
import re 
from typing import List, Tuple
import json
from datetime import datetime, timedelta, timezone 

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
from pandas.core.indexes.period import PeriodIndex
import polars as pl

import wavebuoy_dm.config as config
from wavebuoy_dm.config.config import (NC_DISPLACEMENTS_FILE_NAME_TEMPLATE, 
                                        NC_SPECTRAL_FILE_NAME_TEMPLATE,
                                        NC_BULK_FILE_NAME_TEMPLATE,
                                        OPERATING_INSTITUTIONS)

class ncSpectralAttrsExtractor:
    def _extract_general_spectral_analysis_technique() -> str:
        return "Fast Fourier Transform"
    
    def _extract_general_spectral_analysis_technique_reference() -> str:
        return """Kuik, A. J., van Vledder, G. P., & Holthuijsen, L. H. (1988).
                A Method for the Routine Analysis of Pitch-and-Roll Buoy Wave Data,
                Journal of Physical Oceanography, 18(7), 1020-1034.
                Retrieved Feb 21, 2022, from https://journals.ametsoc.org/view/journals/phoc/18/7/1520-0485_1988_018_1020_amftra_2_0_co_2.xml"""
    
class ncAttrsExtractor:

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
        return re.sub(r'(?<!^)(?=[A-Z0-9])', '-', site_name).upper().replace(" ", "")
    
    def  _extract_deployment_metadata_instrument(deployment_metadata: pd.DataFrame):
        return deployment_metadata.loc["Instrument", "metadata_wave_buoy"]


    def _extract_deployment_metadata_hull_serial_number(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Hull serial number", "metadata_wave_buoy"]
    
    # def _extract_deployment_metadata_institution(deployment_metadata: pd.DataFrame) -> str:
    #     op_inst_code = {"UWA":"The University of Western Australia",
    #                       "Deakin":"Deakin University",
    #                       "NSW-DCCEEW" : "New South Wales Department of Climate Change, Energy, the Environment and Water",
    #                       "IMOS":"IMOS Coastal Wave Buoys Facility",
    #                       "SARDI": "South Australian Research and Development Institute"
    #                       }
        
    #     operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        
    #     if "IMOS" in operating_institution:
    #         return op_inst_code["IMOS"]
       
    #     else:
    #         return op_inst_code[operating_institution]
    
    def _extract_deployment_metadata_water_depth(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Water depth", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_water_depth_units(deployment_metadata: pd.DataFrame) -> str:
        return "m"
    
    def _extract_deployment_metadata_instrument_burst_duration(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst duration", "metadata_wave_buoy"]

    def _extract_deployment_metadata_instrument_burst_interval(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument burst interval", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_instrument_burst_unit(deployment_metadata: pd.DataFrame) -> str:
        return "s"
    
    def _extract_deployment_metadata_instrument_sampling_interval(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Instrument sampling interval", "metadata_wave_buoy"]

    def _extract_deployment_metadata_spotter_id(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Spotter_id ", "metadata_wave_buoy"]
    
    def _extract_deployment_metadata_wave_sensor_serial_number(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["Wave sensor serial number", "metadata_wave_buoy"]

    # def _extract_deployment_metadata_title(deployment_metadata: pd.DataFrame) -> str:
        
    #     base_title = """Delayed mode integral wave parameters from wave buoys collected by {operating_institution} using a {instrument} at {site_name}"""
    #     operating_institution = ncAttrsExtractor._extract_deployment_metadata_institution(deployment_metadata=deployment_metadata)
    #     instrument = ncAttrsExtractor._extract_deployment_metadata_instrument(deployment_metadata=deployment_metadata)
    #     site_name = ncAttrsExtractor._extract_deployment_metadata_site_name(deployment_metadata=deployment_metadata)
        
    #     return base_title.format(operating_institution=operating_institution,
    #                             instrument=instrument,
    #                             site_name=site_name)

    def _extract_deployment_metadata_title(deployment_metadata: pd.DataFrame, regional_metadata: pd.DataFrame) -> str:
        base_title = """Delayed mode integral wave parameters from wave buoys collected by {operating_institution} using a {instrument} at {site_name}"""
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
        
    def _extract_deployment_metadata_watch_circle(deployment_metadata: pd.DataFrame) -> str:
        return deployment_metadata.loc["watch_circle", "metadata_wave_buoy"]

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

    # generally pre-defined --------------------
    def _extract_data_history(dataset: xr.Dataset) -> str:
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
        return 'IMOS'
    
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
    
    def _extract_general_disclaimer() -> str:
        return 'Data, products and services from UWA are provided \\"as is\\" without any warranty as to fitness for a particular purpose.'

    def _extract_general_water_depth_reference() -> str:
        return "mean sea level"
    
    def _extract_general_buoy_specification_url() -> str:
        return "https://s3-ap-southeast-2.amazonaws.com/content.aodn.org.au/Documents/AODN/Waves/Instruments_manuals/Spotter_SpecSheet%20Expanded.pdf"

class ncAttrsComposer:
    def __init__(self, 
                 buoys_metadata: pd.DataFrame,
                 deployment_metadata: pd.DataFrame,
                 regional_metadata: pd.DataFrame,
                 parameters_type: str = "spectral"):
        
        self.parameters_type = parameters_type
        self.buoys_metadata = buoys_metadata
        self.deployment_metadata = deployment_metadata
        self.regional_metadata = regional_metadata
        self.attrs_templates_files = {"displacements": "displacements_attrs.json",
                                "spectral":"spectral_dm_attrs.json",
                                "bulk":"bulk_attrs.json"}
        self.attrs_template = self._get_template_imos(file_name=self.attrs_templates_files[parameters_type])
    
    def _get_template_imos(self, file_name: str) -> dict:
        
        file_path = os.path.join(os.path.dirname(config.__file__), file_name)
        with open(file_path) as j:
            return json.load(j)

    def _match_valid_min_max_dtype(self, variable:str, variables_attributes:dict , dataset: xr.Dataset,
                                   min_attribute_name:str, max_attribute_name:str):
        
        return (np.dtype(dataset[variable]).type(variables_attributes[min_attribute_name]),
                np.dtype(dataset[variable]).type(variables_attributes[max_attribute_name]))

    def assign_variables_attributes(self, dataset: xr.Dataset) -> xr.Dataset:
        
        variables = list(self.attrs_template['variables'].keys())
        
        for variable in variables:
            if variable in list(dataset.variables):
                variables_attributes = self.attrs_template['variables'][variable]
                
                if "quality_control" in variable:
                    variables_attributes["flag_values"] = np.int8(variables_attributes["flag_values"])
                
                if "valid_min" in variables_attributes or variable not in ("TIME", "TIME_TEMP", "timeSeries", "FREQUENCY"):
                    variables_attributes["valid_min"], variables_attributes["valid_max"] = self._match_valid_min_max_dtype(
                                                                                variable=variable,
                                                                                dataset=dataset,
                                                                                variables_attributes=variables_attributes,
                                                                                min_attribute_name="valid_min",
                                                                                max_attribute_name="valid_max"
                                                                            )
                
                if "min" in variables_attributes and variable == "FREQUENCY":
                    variables_attributes["min"], variables_attributes["max"] = self._match_valid_min_max_dtype(
                                                                                variable=variable,
                                                                                dataset=dataset,
                                                                                variables_attributes=variables_attributes,
                                                                                min_attribute_name="min",
                                                                                max_attribute_name="max"
                                                                            )

                dataset[variable] = (dataset[variable].assign_attrs(variables_attributes))
        
        return dataset

    def assign_variables_attributes_dataset_objects(self, dataset_objects: xr.Dataset) -> xr.Dataset:
        for dataset in dataset_objects:
            dataset = self.assign_variables_attributes(dataset=dataset)

        return dataset_objects

    def assign_general_attributes(self, dataset: xr.Dataset, site_name: str) -> xr.Dataset:
                
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
                        if name.endswith("_title") or name.endswith("_abstract"):
                            kwargs.update({"regional_metadata":self.regional_metadata})
                    
                    elif name.startswith("_extract_general_"):
                        key = name.removeprefix("_extract_general_") 
                        kwargs = {}

                    elif name.startswith("_extract_regional_metadata_"):
                        key = name.removeprefix("_extract_regional_metadata_") 
                        kwargs = kwargs = {"regional_metadata": self.regional_metadata,
                                        "deployment_metadata":self.deployment_metadata}

                    try:                        
                        extracted = method(**kwargs)
                    except:
                        extracted = self._get_attribute_from_template(attribute_name=key, attributes_template=self.attrs_template)
                    
                    general_attributes.update({key:extracted})

        if self.parameters_type == "spectral":

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

class ncWriter:
    
    ENCODING_ENFORCEMENT_SPECTRA = {
                "TIME": {"dtype": np.float64, "_FillValue":None},
                "FREQUENCY": {"dtype": np.float32, "_FillValue":None},
                "LATITUDE": {"dtype": np.float64},
                "LONGITUDE": {"dtype": np.float64},
                "A1": {"dtype": np.float32},
                "B1": {"dtype": np.float32},
                "A2": {"dtype": np.float32},
                "B2": {"dtype": np.float32},
                "ENERGY": {"dtype": np.float32},
            }
    
    ENCODING_ENFORCEMENT_BULK = {"TIME":{"_FillValue":None},
                            "TIME_TEMP":{"_FillValue":None},
                            'WSSH':{"dtype":np.float32},
                            'WPPE':{"dtype":np.float32},
                            'WPFM':{"dtype":np.float32},
                            'WPDI':{"dtype":np.float32},
                            'WPDS':{"dtype":np.float32},
                            'SSWMD':{"dtype":np.float32},
                            'WMDS':{"dtype":np.float32},
                            'TEMP':{"dtype":np.float32},
                            'WAVE_quality_control':{"dtype":np.int8},
                            'TEMP_quality_control':{"dtype":np.int8},
                            'TIME':{"dtype":np.float64},
                            'LATITUDE':{"dtype":np.float64},
                            'LONGITUDE':{"dtype":np.float64},
                            'timeSeries':{"dtype":np.int16}
                    }

    ENCODING_ENFORCEMENT_DISPLACEMENTS = {
                "TIME": {"dtype": np.float64, "_FillValue":None},
                "TIME_LOCATION": {"dtype": np.float64, "_FillValue":None},
                "LATITUDE": {"dtype": np.float64},
                "LONGITUDE": {"dtype": np.float64},
                "XDIS": {"dtype": np.float32},
                "YDIS": {"dtype": np.float32},
                "ZDIS": {"dtype": np.float32},
            }

    def _get_operating_institution(self, deployment_metadata: pd.DataFrame) -> str:
        operating_institution = deployment_metadata.loc["Operating institution","metadata_wave_buoy"]
        if "IMOS" in operating_institution:
            operating_institution = "IMOS_COASTAL-WAVE-BUOYS"
        elif "IMOS" not in operating_institution:
            try:
                operating_institution = OPERATING_INSTITUTIONS[operating_institution]
            except:
                raise ValueError(f"{operating_institution} not valid. Please make sure the operating institution code is valid in the deployment metadata file")

        return operating_institution

    def extract_start_end_dates(self, dataset: xr.Dataset):
        start_date = dataset["TIME"].min().dt.strftime("%Y%m%d").values
        end_date =dataset["TIME"].max().dt.strftime("%Y%m%d").values

        return (start_date, end_date)

    def extract_start_end_dates_list(self, dataset_objects):
        periods = []
        for dataset in dataset_objects:       
            periods.append(self.extract_start_end_dates(dataset))
        return periods
    
    def _format_periods(self, periods: PeriodIndex) -> list:
        return [(period[0].strftime("%Y%m%d"), period[1].strftime("%Y%m%d")) for period in periods]

    def _format_site_id_to_filename(self, site_id: str) -> str:
        
        site_id = re.sub(r'([a-z])([A-Z])', r'\1-\2', site_id)   # lower→UPPER boundary
        site_id = re.sub(r'([A-Za-z])(\d+)', r'\1-\2', site_id)
        
        return site_id.upper()

    def compose_file_names(self,
                            site_id: str,
                            deployment_metadata: pd.DataFrame,
                            periods: PeriodIndex,
                            parameters_type: str = "spectral") -> List[str]:
        
        if parameters_type == "spectral":
            file_name_template = NC_SPECTRAL_FILE_NAME_TEMPLATE
        
        elif parameters_type == "displacements":
            file_name_template = NC_DISPLACEMENTS_FILE_NAME_TEMPLATE
            periods = self._format_periods(periods)
        
        elif parameters_type == "bulk":
            file_name_template = NC_BULK_FILE_NAME_TEMPLATE

        file_names = []

        operating_institution = self._get_operating_institution(deployment_metadata=deployment_metadata)
        site_id = re.sub(r'_+', '', site_id).strip()

        for period in periods:
            file_name = file_name_template.format(operating_institution=operating_institution,
                                                        site_id=self._format_site_id_to_filename(site_id),
                                                        start_date=period[0],
                                                        end_date=period[1]
                                                    )
            file_names.append(file_name)
        
        return file_names

    def _compose_file_paths(self, file_names: list, output_path: str) -> list:
        return [os.path.join(output_path, file_name) for file_name in file_names]

    def _process_encoding(self, dataset: xr.Dataset, parameters_type: str) -> dict:
                
        if parameters_type == "spectral":
            return self.ENCODING_ENFORCEMENT_SPECTRA.copy()
        
        elif parameters_type == "bulk":
            
            encoding = self.ENCODING_ENFORCEMENT_BULK.copy()
            
            if "TEMP" not in list(dataset.variables):
                del encoding["TEMP"]
                del encoding["TEMP_quality_control"]
                del encoding["TIME_TEMP"]
        
        elif parameters_type == "displacements":
            return self.ENCODING_ENFORCEMENT_DISPLACEMENTS.copy()
    
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

    def save_nc_file(self, 
                     output_path: str,
                     file_names: str,
                     dataset_objects: xr.Dataset,
                     parameters_type: str = "spectral"):
        
        file_paths = self._compose_file_paths(output_path=output_path,file_names=file_names)       
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for file_path, dataset in zip(file_paths, dataset_objects):
            dataset = self._remove_coordinates_qc_variables(dataset=dataset)
            dataset = self._remove_fillvalue_attributes(dataset=dataset)

            encoding = self._process_encoding(dataset=dataset, parameters_type=parameters_type)
            dataset.to_netcdf(file_path, engine="netcdf4",
                                encoding=encoding)
            
