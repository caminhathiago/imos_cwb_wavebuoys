import os
import re
import logging

from netCDF4 import Dataset

from wavebuoy_nrt.config.config import NC_FILE_NAME_TEMPLATE, NC_SPECTRAL_FILE_NAME_TEMPLATE, OPERATING_INSTITUTIONS
from wavebuoy_nrt.netcdf.writer import ncMetaDataLoader

LOGGER = logging.getLogger("aodn_ftp_push_logger")

class ncValidator():
    
    @staticmethod
    def validate_file_name(file_path: str=None, file_name: str=None):
        
        # General matching
        if file_path:
            file_name = os.path.basename(file_path)
        if not file_name:
            return ValueError("Either 'file_name' or 'file_path' must be provided.")
        
        pattern_bulk = r"^[A-Za-z_-]+_\d{8}_[A-Za-z0-9]+_RT_WAVE-PARAMETERS_monthly\.nc$"
        pattern_spectral = r"^[A-Za-z_-]+_\d{8}_[A-Za-z0-9]+_RT_WAVE-SPECTRA_monthly\.nc$"

        if not re.fullmatch(pattern_bulk, file_name) and not re.fullmatch(pattern_spectral, file_name):
            error = f"File name not matching templates {NC_FILE_NAME_TEMPLATE} or {NC_SPECTRAL_FILE_NAME_TEMPLATE}"
            LOGGER.error(error)
            return f"validate_file_name - failed: {error}"
        
        # Specific matching
        file_name_parts = file_name.split("_")
        if "IMOS" in file_name_parts:
            file_name_parts[0] = file_name_parts[0] + "_" + file_name_parts[1]
            del file_name_parts[1]

        if file_name_parts[0] not in OPERATING_INSTITUTIONS.values():
            error = f"Operating institution '{file_name_parts[0]}' not in {list(OPERATING_INSTITUTIONS.values())}"
            LOGGER.error(error)
            return f"validate_file_name - failed: {error}"
        
        return "validate_file_name - passed"
        # if file_name_parts[2] not in 

    @staticmethod        
    def validade_nc_integrity(file_path: str):
        try:
            ds = Dataset(file_path)
        except Exception as e:
            return f"validade_nc_integrity - failed: {str(e)}"
        
        return "validade_nc_integrity - passed"
        
    @staticmethod
    def validade_variables_attributes(Dataset: Dataset):
        pass
        
    @staticmethod
    def validade_general_attributes(Dataset: Dataset):
        pass

