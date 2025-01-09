import boto3
from botocore.config import Config as Boto3Config
from dotenv import load_dotenv

import os

load_dotenv()

class CWBAWSs3():
    """
    A class to interact with Coastal Wave Buoy Facility's AWS S3 services.
    """
    
    def __init__(self):
        """
        Initializes the CWBAWSs3 instance by setting up the boto3 configuration,
        session, and client.
        """
        self.boto3_config = self.configure_boto3()
        self.boto3_session = self.create_boto3_session()
        self.s3 = self.create_boto3_client(session=self.boto3_session,
                                                    boto3_config=self.boto3_config)
        self.buckets = self._get_buckets_list()
        self.buckets_names = self._get_buckets_names(buckets_list=self.buckets)

    def configure_boto3(self, retries:int=5) -> Boto3Config:
        """
        Configures the boto3 client settings.
        
        Returns:
            Boto3Config: A configuration object containing boto3 settings.
        """
        return Boto3Config(
            region_name = os.getenv('CWB_AWS_REGION'),
                signature_version = 'v4',
                retries = {
                    'max_attempts': retries,
                    'mode': 'standard'
                }
            )
    
    def create_boto3_session(self):
        """
        Creates a boto3 session using environment variables for credentials.

        Returns:
            boto3.Session: The created boto3 session.
        """
        return boto3.Session(aws_access_key_id=os.getenv('CWB_AWS_ACCESS_KEY_ID'), 
                                aws_secret_access_key=os.getenv('CWB_AWS_ACCESS_KEY_SECRET'),
                            )

    def create_boto3_client(self, session:boto3.session, boto3_config:Boto3Config):
        """
        Creates a boto3 client for AWS S3 using the provided session and configuration.
                
        Args:
            session (boto3.session): The boto3 session to be used for the client.
            boto3_config (Boto3Config): The configuration settings for the boto3 client.
        
        Returns:
            boto3.client: The created boto3 S3 client.
        """
        return session.client("s3", config=boto3_config)


    def _get_buckets_list(self) -> list:
        """
        Retrieves a list of S3 buckets from Coastal Wave Buoys Facility.
        
        Returns:
            list: A list of dictionaries containing details of each S3 bucket.
        """
        return self.s3.list_buckets()["Buckets"]
    
    def _get_buckets_names(self, buckets_list:list) -> list:
        """
        Extracts the names of S3 buckets from a list of bucket metadata dictionaries.
        
        Args:
            buckets_list (list): A list of dictionaries, each containing metadata
                                for CWBs S3 buckets. 
        Returns:
            list: A list of strings representing the names of the S3 buckets.
        """
        return [bucket['Name'] for bucket in buckets_list]
        

    def _find_file(self, file_name:str):
        pass
    
    def get_bucket_content(self):
        pass

    def get_bucket_files_list(self):
        pass

    def retrieve_file(self):
        pass

    def load_file(self):
        pass

    def rename_file(self):
        pass

