import pandas as pd
from datetime import datetime, timedelta
import os
# import importlib.resources as pkg_resources

# from aws.cwb import CWBAWSs3



class FilesHandler():
    def __init__(self):
        # self._root = os.path.dirname(os.path.relpath(__file__))
        pass

    def _get_file_path(self, file_name):
        parent_path = r"C:\Users\00116827\cwb\wavebuoy_aodn\wavebuoy"
        return os.path.join(parent_path, file_name)
    

class WaveBuoy(FilesHandler): #(CWBAWSs3):
    def __init__(self):
        super().__init__()
        self.buoys_metadata = self._get_buoys_metadata()
        self.buoys_metadata_token_sorted = self._sort_sites_by_sofar_token(buoys_metadata=self.buoys_metadata)

        self.site_ids = self._get_site_ids(buoys_metadata=self.buoys_metadata_token_sorted)

    def _get_buoys_metadata(self):
        buoys_metadata_path = self._get_file_path(file_name="buoys_metadata.csv")
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

    def get_latest_processed_date_time(self):
        pass

    def get_latest_available_date_time(self):
        pass

    def generate_period_to_extract(self):
        pass

if __name__ == "__main__":

    wb = WaveBuoy()