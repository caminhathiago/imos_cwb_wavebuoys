import os
from datetime import datetime, timedelta

import ftplib
import glob

class ncPusher:
    def __init__(self,
                 host: str,
                 user: str,
                 password: str):
        self._host = host
        self._user = user
        self._password = password
        # self._ftp_server = self.set_server()

    @property
    def host(self):
        return self._host
    
    @property
    def user(self):
        return self._user
    
    @property
    def password(self):
        return self._password
    
    def set_server(self):
        ftp_server = ftplib.FTP(host=self._host,
                                user=self._user,
                                passwd=self._password)
        return ftp_server
    
    def grab_nc_files_to_push(self, incoming_path: str) -> list:
        
        condition = os.path.join(incoming_path, "*.nc")
        file_paths = glob.glob(condition)
        file_times = []
        for file_path in file_paths:
            creation_date_time = datetime.fromtimestamp(os.path.getctime(file_path))
            modification_date_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            file_times.append({"creation_time": creation_date_time, 
                               "modification_time": modification_date_time,
                               "file_path": file_path})

        current_time = datetime.now()
        last_hour_time = current_time - timedelta(hours=0.8)
        print(last_hour_time)


        files_to_push = []
        for file_time in file_times:
            if file_time["creation_time"] > last_hour_time or file_time["modification_time"] > last_hour_time:
                files_to_push.append(file_time)

        return tuple(files_to_push)
    
    def read_files_as_binary(self):
        binary_objects = []
        file_paths = []
        for nc_file in file_paths:
            with open(nc_file, "rb") as bin_obj:
                binary_objects.append(bin_obj)
        return binary_objects
    
    def load_binary_objects(self, binary_objects: list):
        for binary_obj in binary_objects:
            self.ftp_server.storbinary()

    def get_server_dirs(self):
        return self._ftp_server.dir()