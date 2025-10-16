from pathlib import Path
import os
import re
import logging
from datetime import datetime

import glob
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
SITE_LOGGER = logging.getLogger("DM_processing")

# path = Path(r"\\drive.irds.uwa.edu.au\OGS-COD-001\CUTTLER_wawaves\Data\wawaves\OceanBeach\delayedmode\OceanBeach_deploy20240701_retrieve20241205_SPOT31395C\log")

class WaveBuoy():

    def extract_region_site_name(self, path: str) -> str:
        parts = Path(path).parts 
        regions = {"wawaves", "sawaves", "qldwaves", "vicwaves", "nswwaves"}
        # return "CollaroyNarrabeen", "nswwaves"
        for i, part in enumerate(parts):
            if part in regions:
                if i + 1 < len(parts):
                    return parts[i + 1], part
                else:
                    raise Exception("Invalid path structure: no folder after region name")

        raise Exception("Invalid path: no recognized region found")

    def _get_buoys_metadata(self, buoy_type:str, buoys_metadata_file_name:str):
        try:
            file_path = os.path.normpath(os.path.join(os.getenv('IRDS_PATH'), "Data", "website", "auswaves"))
            if not os.path.exists(file_path):
                raise FileNotFoundError("No such directory for buoys metadata: {}")
            buoys_metadata_path = os.path.join(file_path, buoys_metadata_file_name)
            buoys_metadata = pd.read_csv(buoys_metadata_path)
            buoys_metadata = buoys_metadata.loc[buoys_metadata.type == buoy_type]
            # buoys_metadata["region"] = self._get_regions(buoys_metadata=buoys_metadata)
            buoys_metadata = buoys_metadata.set_index('name')
            # GENERAL_LOGGER.info("Buoys metadata grabbed successfully")
            return buoys_metadata

        except:
            raise TypeError("Loading and processing buoys_metadata.csv unsuccessful. Check if the file is corrupted or if its structure has been changed")
            # GENERAL_LOGGER.error(error_message, exc_info=True)

    def _get_deployment_metadata_files(self, site_name: str, region: str, file_extension: str = "*.xlsx") -> pd.DataFrame:
        
        files_path = os.path.join(os.getenv('IRDS_PATH'), "Data", region, site_name)
        
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
        
        print(files_path)

        files = glob.glob(os.path.join(files_path, metadata_folder, file_extension))

        if files:
            return files
        else:
            error_message = f"No deployment metadata files provided for {site_name}. Please make sure at least the most recent one exists and matches the correct file naming standar."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)

    def _validate_deployment_metadata_file_name(self, file_paths: list):
        template = re.compile(r"metadata_[A-Za-z0-9]+_deploy(\d+).xlsx")

        matches, not_matches, temp_files = [], [], []
        for file in file_paths:
            if template.search(file):
                matches.append(file)
            else:
                if os.path.basename(file).startswith("~$"):
                    temp_files.append(file)
                else:
                    not_matches.append(file)

        if not matches:
            error_message = "No deployment metadata files found."
            SITE_LOGGER.error(error_message)
            raise FileNotFoundError(error_message)

        elif temp_files and not not_matches:
            SITE_LOGGER.warning(f"the following deployment metadata sheets weren't closed properly: {temp_files}")

        elif not_matches:
            SITE_LOGGER.warning(file_paths)
            SITE_LOGGER.warning(matches)
            error_message = "At least one of the deployment metadata files name is not conforming with the expected template (metadata_{site_name}_deploy{YYYYmmdd}.xlsx). Make sure all of them are conforming."
            SITE_LOGGER.error(error_message)
            raise NameError(error_message)

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

    def load_latest_deployment_metadata(self, site_name:str, region:str) -> pd.DataFrame:
        file_paths = self._get_deployment_metadata_files(site_name=site_name, region=region)
        file_path = self._get_latest_deployment_metadata(file_paths=file_paths)
        
        deployment_metadata = pd.read_excel(file_path)
        metadata_wave_buoy_col = deployment_metadata.filter(regex="Metadata").columns
        deployment_metadata = deployment_metadata.rename(columns={metadata_wave_buoy_col[0]:"metadata_wave_buoy",
                                                                  "Parameter":"parameter"})
        deployment_metadata = deployment_metadata.set_index("parameter")

        return deployment_metadata

    def load_regional_metadata(self) -> pd.DataFrame:
        metadata_path = os.path.normpath(os.getenv("METADATA_PATH"))
        regional_metadata_path = os.path.join(metadata_path, "regional_metadata.csv")
        return pd.read_csv(regional_metadata_path)
    
    def load_buoys_to_process(self, region:str) -> pd.DataFrame:
        'waves'
        region_path = os.path.join(os.getenv('IRDS_PATH'), 'Data', region + 'waves')
        pattern = f"{region}_delayed_mode_buoys_to_process.csv"
        file_path = glob.glob(os.path.join(region_path, pattern))[0]
        if os.path.exists(file_path):
            buoys_to_process = pd.read_csv(file_path)
            buoys_to_process = buoys_to_process.loc[buoys_to_process['process']==1]
            return buoys_to_process
        else:
            raise NotADirectoryError(f"{file_path} does not exist")

    def extract_deploy_dates_spotid_from_path(self, deployment_folder_name:str) -> list[datetime]:
        
        pattern = r'deploy(\d{8})_retrieve(\d{8})_SPOT-(\w+)'
        match = re.search(pattern, deployment_folder_name)

        if match:
            deploy_str, retrieve_str, spot_id = match.groups()
            deploy_date = datetime.strptime(deploy_str, '%Y%m%d')
            retrieve_date = datetime.strptime(retrieve_str, '%Y%m%d')
            return deploy_date, retrieve_date, spot_id
        else:
            raise ValueError(f"Unable to extract deploy and retrieve dates, and spot id from dm deployment folder")