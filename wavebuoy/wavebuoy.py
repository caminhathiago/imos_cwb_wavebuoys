import os
from datetime import datetime, timedelta

import pandas as pd

from wavebuoy.netcdf.lookup import NetCDFFileHandler

from wavebuoy.config.config import FILES_PATH


class FilesHandler():
    def __init__(self):
        pass

    def _get_file_path(self, file_name):
        return os.path.join(FILES_PATH, file_name)
    

class WaveBuoy(FilesHandler, NetCDFFileHandler): #(CWBAWSs3):
    def __init__(self, buoy_type:str,buoys_metadata_file_name:str="buoys_metadata.csv"):
        self.buoys_metadata = self._get_buoys_metadata(buoy_type=buoy_type, buoys_metadata_file_name=buoys_metadata_file_name)
        self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)
        self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata_token_sorted)
        self.sites_per_region = self._get_sites_per_region(buoys_metadata=self.buoys_metadata_token_sorted)

    def _get_buoys_metadata(self, buoy_type:str,buoys_metadata_file_name:str):
        buoys_metadata_path = self._get_file_path(file_name=buoys_metadata_file_name)
        buoys_metadata = pd.read_csv(buoys_metadata_path)
        buoys_metadata = self._select_buoy_type(buoy_type=buoy_type, buoys_metadata=buoys_metadata)
        buoys_metadata["region"] = self._get_regions(buoys_metadata=buoys_metadata)
        buoys_metadata = buoys_metadata.set_index('name')
        
        return buoys_metadata

    def _select_buoy_type(self, buoy_type:str, buoys_metadata:pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.loc[buoys_metadata["type"] == buoy_type]

    def _get_site_ids(self, buoys_metadata=pd.DataFrame) -> list:
        return buoys_metadata.index.values
    
    def _get_sites(self, buoys_metadata=pd.DataFrame) -> list:
        return (buoys_metadata
                .reset_index()
                [["name","region"]]
                .to_dict(orient="records")
        )

    def _sort_sites_by_sofar_token(self, buoys_metadata=pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.sort_values("sofar_token")

    def get_latest_available_date_time(self):
        pass

    def generate_period_to_extract(self):
        pass

    def _get_regions(self, buoys_metadata:pd.DataFrame):
        return (buoys_metadata['archive_path']
                                    .str.extract(r'\\auswaves\\([a-z]+)waves')[0]
                                    .str.upper()
                    )
    
    def _get_sites_per_region(self, buoys_metadata:pd.DataFrame):
        pass 

if __name__ == "__main__":

    wb = WaveBuoy()