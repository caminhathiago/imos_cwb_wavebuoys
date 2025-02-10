import ftplib


class ncPusher:
    def __init__(self,
                 host: str,
                 user: str,
                 password: str):
        self._host = host
        self._user = user
        self._password = password
        self._ftp_server = self.set_server()

    def set_credentials(self):
        return
    
    def get_credentials(self):
        return
    
    def set_server(self):
        ftp_server = ftplib.FTP()
        return ftp_server
    
    def grab_files_to_push(self):
        file_paths = []
        return file_paths
    
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