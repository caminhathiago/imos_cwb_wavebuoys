import os
import re
import logging

from netCDF4 import Dataset

from wavebuoy_nrt.config.config import NC_FILE_NAME_TEMPLATE, NC_SPECTRAL_FILE_NAME_TEMPLATE
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






    @staticmethod
    def create_validation_log_contents(nc_file_name) -> list:
        return {"contents": [nc_file_name + " VALIDATION REPORT \n"], "all_passed": []}

    @staticmethod
    def generate_reports_path(site_id, incoming_path) -> str:
        processed_site_id = site_id.replace("_","")
        reports_path = os.path.join(incoming_path, "sites", processed_site_id, "reports")
        if not os.path.exists(reports_path):
            os.mkdir(reports_path)
        
        return reports_path

    @staticmethod
    def check_any_fail(log_contents):
        return all(log_contents["all_passed"])

    @staticmethod
    def save_validation_log(reports_path, nc_file_name, log_contents) -> None:
        reg_validation_txt_path = os.path.join(reports_path, f"{nc_file_name}_validation_report.txt") 
        with open(reg_validation_txt_path, "w", encoding="utf-8") as f:
            f.write("".join(log_contents["contents"]) + "\n\n")

    @staticmethod
    def validate_regional_metadata(nc_file_name, ds, regional_metadata, log_contents):
        
        log_contents["contents"].append("\nRegional Metadata Validation ------------------")
        # Extract datetime prefix
        datetime_prefix_match = re.search(r"\d{8}", nc_file_name)
        datetime_prefix = datetime_prefix_match.group(0) if datetime_prefix_match else None
        
        # Extract prefix
        if "IMOS" in nc_file_name:
            file_prefix = "_".join(nc_file_name.split("_", 2)[:2])
        else:    
            file_prefix = nc_file_name.split("_")[0]

        # Regional Metadata
        match_regional = regional_metadata.loc[
            regional_metadata["operating_institution_nc_preffix"] == file_prefix
        ].squeeze() 

        if match_regional.empty:
            log_contents["contents"].append(f" • {file_prefix} does not exist in regional metadata operating_institution_nc_preffix column.")
            log_contents["all_passed"].append(False)

        else:
            checks = [
                ("project", "project"),
                ("operating_institution_long_name", "institution"),
                ("principal_investigator", "principal_investigator"),
                ("principal_investigator_email", "principal_investigator_email"),
            ]

            check_results = [
                (reg, bulk, match_regional[reg] == ds.attrs.get(bulk))
                for reg, bulk in checks
            ]
            
            all_match = all(match for _, _, match in check_results)


            if not all_match:
                
                log_contents["contents"].append(f"Mismatch detected")
                for reg, bulk, match in check_results:
                    if not match:
                        expected = match_regional[reg]
                        got = ds.attrs.get(bulk)
                        log_contents["contents"].append(f" • {bulk}: expected {expected!r}, got {got!r}")
                        log_contents["all_passed"].append(False)

                # raise ValueError(f"Attributes mismatching with regional metadata: {log_contents}")
            
            else:
                log_contents["contents"].append(f"PASSED")
                log_contents["all_passed"].append(True)


      
    @staticmethod
    def validate_spot_id(site_id, ds, deployment_metadata, buoys_metadata, log_contents):
        
        log_contents["contents"].append("\nSpotter ID Validation ------------------")

        dep_spotter_id = deployment_metadata.loc["Spotter_id ", "metadata_wave_buoy"]
        dep_wave_sensor_serial_number = deployment_metadata.loc["Wave sensor serial number", "metadata_wave_buoy"]
        buoys_spotter_id = buoys_metadata.loc[site_id, "serial"]

        checks = (dep_spotter_id == buoys_spotter_id,
                  dep_spotter_id == ds.spotter_id,
                  buoys_spotter_id == ds.spotter_id,
                  dep_spotter_id == dep_wave_sensor_serial_number)
        
        if not all(checks):
            log_contents["contents"].append(f""" • SPOT-ID FAIL: {dep_spotter_id} (deployment metadata); 
                                                {dep_wave_sensor_serial_number} (deployment metadata serial number);
                                                {buoys_spotter_id} (buoys metadata); 
                                                {ds.spotter_id} (dataset); """)
            log_contents["all_passed"].append(False)
            
            # raise ValueError(f"{log_contents[0]}")
        else:
            log_contents["contents"].append(f"PASSED")
            log_contents["all_passed"].append(True)

    @staticmethod
    def validate_site_name(site_id, ds, deployment_metadata, log_contents) -> None:
        
        log_contents["contents"].append("\nSite Name Validation ------------------")

        dep_site_name = deployment_metadata.loc["Site Name", "metadata_wave_buoy"]

        if not dep_site_name == ds.site_name:
            log_contents["contents"].append(f" • Site name discrepancy: {dep_site_name} (deployment metadata); {ds.site_name} (dataset)")
            log_contents["all_passed"].append(False)

        else:
            log_contents["contents"].append("PASSED")
            log_contents["all_passed"].append(True)


    @staticmethod
    def validate_principal_investigator():
        pass

    @staticmethod
    def validate_principal_investigator_email():
        pass

    @staticmethod
    def validate_datatypes():
        pass    