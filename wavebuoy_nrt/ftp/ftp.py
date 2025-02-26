import os
from datetime import datetime, timedelta
import ssl
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
        self._context = ssl.create_default_context()
        self._context.check_hostname = False
        self._context.verify_mode = ssl.CERT_NONE
        self._ftp = self._start_server()
        self._configure_server()

    @property
    def ftp(self):
        return self._ftp
    
    @property
    def host(self):
        return self._host
    
    @property
    def user(self):
        return self._user
    
    @property
    def password(self):
        return self._password
    
    @property
    def server_dirs(self):
        return self._ftp.dir()

    def _start_server(self):
        return MyFTP_TLS(timeout=30, context=self._context)
    
    def _connect(self, port=21):
        return self._ftp.connect(host= self._host, port=port)
    
    def _auth(self):
        return self._ftp.auth()
    
    def _login(self):
        return self._ftp.login(self._user, self._password)
    
    def _secure_data_connection(self):
        return self._ftp.prot_p()

    def _configure_server(self):
        self._connect()
        self._auth()
        self._login()

    def change_dir(self, path: str):
        self._secure_data_connection()
        if path in self._ftp.nlst():
            return self._ftp.cwd(path)
        else:
            raise FileNotFoundError("Passed path does not exist in the FTP server.")
    
    def pwd(self):
        self._secure_data_connection()
        return self._ftp.pwd()

    def dir(self):
        self._secure_data_connection()
        return self._ftp.dir()

    def nlst(self):
        self._secure_data_connection()
        return self._ftp.nlst()

    def close(self):
        return self._ftp.close()
    
    def quit(self):
        return self._ftp.quit()

    def grab_nc_files_to_push(self, incoming_path: str, lookback_hours: timedelta = 1) -> list:
        
        condition = os.path.join(incoming_path, "*.nc")
        file_paths = glob.glob(condition)
        
        file_times = []
        for file_path in file_paths:
            creation_date_time = datetime.fromtimestamp(os.path.getctime(file_path))
            modification_date_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            file_times.append({"file_name": os.path.basename(file_path),
                                "creation_time": creation_date_time, 
                               "modification_time": modification_date_time,
                               "file_path": file_path})

        current_time = datetime.now()
        last_hour_time = current_time - timedelta(hours=lookback_hours)

        files_to_push = []
        for file_time in file_times:
            if file_time["creation_time"] > last_hour_time or file_time["modification_time"] > last_hour_time:
                files_to_push.append(file_time)

        return tuple(files_to_push)
    
    def push_files_to_ftp(self, files_to_push: list):
        self._secure_data_connection()
        for file in files_to_push:
            self.push_file_to_ftp(file=file)

    def push_file_to_ftp(self, file: dict):
        file_name_ftp = "STOR " + file["file_name"]
        with open(file["file_path"], "rb") as binary_obj:
            self._ftp.storbinary(file_name_ftp,binary_obj)

    def check_size(self, file1_name: str, file2_name: str) -> bool:
        return self._ftp.size(file1_name) == self._ftp.size(file2_name)
    

class MyFTP_TLS(ftplib.FTP_TLS):
    """Explicit FTPS, with shared TLS session"""
    def ntransfercmd(self, cmd, rest=None):
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn, server_hostname=self.host, session=self.sock.session)
        return conn, size