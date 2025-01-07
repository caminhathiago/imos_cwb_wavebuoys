#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from wavebuoy.sofar.api import sofarApi

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    # Loading metadata
    """ 
    - Load buoys_metadata.csv as Pandas DataFrame
    - Get site_ids (site names)
    - Get Sofar API Tokens
    - 
    """

    # General ETL idea
    """
    - Iterate through each site_id
    - Handle error for each site_name
    - Reuse aodn logging approach
    - 

    """

    # Extract
    """
    Algoritm 1
    - Get current date_time (focus on current month)
    - Get last available date_time for that month
    - Get last downloaded date_time from the NetCDF files
    -

    Algoritm 2
    - Get current date_time
    - Get last available date_time from SofarAPI
    - Get last downloaded date_time (latest NetCDF file)
    . using with open OR loading the whole file
    - Extract data SofarAPI using previous date_times
    -
    """


    # Transform
    """
    - Convert to Pandas DataFrame
    - Convert date_times, date_types, etc
    - Outputs Pandas DataFrame
    
    """

    # Qualify
    """
    - check if current extraction has 24h hours 
        . if not, load last NetCDF file as a pandas DataFrame
            - Check if last NetCDF file has XX hours from current date
                . if not, load second latest NetCDF file as a Pandas DataFrame
    - 
    . range test
    . spike test
    - check if 
    

    """


    # Final Transform (Generate NetCDF)
    """
    - Generate monthly NetCDFs 
    *note: use AODN naming convention:
    template -> IMOS_COASTAL-WAVE-BUOYS_YYYYMMDD_SITENAME_RT_WAVE-PARAMETERS_monthly.nc
    e.g. IMOS_COASTAL-WAVE-BUOYS_20241101_HILLARYS_RT_WAVE-PARAMETERS_monthly.nc
    -
    
    """


    # Load



