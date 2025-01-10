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
    def __init__(self, buoys_metadata_file_name:str="buoys_metadata.csv"):
        self.buoys_metadata = self._get_buoys_metadata(buoys_metadata_file_name=buoys_metadata_file_name)
        self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)

        self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata_token_sorted)

    def _get_buoys_metadata(self, buoys_metadata_file_name:str):
        buoys_metadata_path = self._get_file_path(file_name=buoys_metadata_file_name)
        buoys_metadata = pd.read_csv(buoys_metadata_path)
        buoys_metadata = self._select_sofar_buoys(buoys_metadata=buoys_metadata)
        buoys_metadata = buoys_metadata.set_index('name')
        
        return buoys_metadata

    def _select_sofar_buoys(self, buoys_metadata:pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.loc[buoys_metadata["type"] == "sofar"]

    def _get_site_ids(self, buoys_metadata=pd.DataFrame) -> list:
        return buoys_metadata.index.values

    def _sort_sites_by_sofar_token(self, buoys_metadata=pd.DataFrame) -> pd.DataFrame:
        return buoys_metadata.sort_values("sofar_token")

    # def get_latest_processed_date_time(self):
    #     pass

    def get_latest_available_date_time(self):
        pass

    def generate_period_to_extract(self):
        pass

if __name__ == "__main__":

    wb = WaveBuoy()