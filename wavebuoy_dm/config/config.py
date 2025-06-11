OPERATING_INSTITUTIONS = {"UWA":"UWA",
                          "Deakin":"VIC-DEAKIN-UNI",
                          "NSW-DCCEEW" : "NSW-DPE",
                          "IMOS":"IMOS_COASTAL-WAVE-BUOYS",
                          "SARDI": "SARDI"
                          }

AODN_COLUMNS_TEMPLATE = {
    "timestamp": 'TIME',
    "longitude": 'LONGITUDE',
    "latitude": 'LATITUDE',
    "meanDirection": 'SSWMD',
    "meanDirectionalSpread": 'WMDS',
    "peakDirection": 'WPDI',
    "peakDirectionalSpread": 'WPDS',
    "meanPeriod": 'WPFM',
    "peakPeriod": 'WPPE',
    "significantWaveHeight": 'WSSH',
    "degrees": 'TEMP', # Spotter TEMP
    "value": 'TEMP', # SM surface temperature
    "temperature": 'TEMP'
}

NC_SPECTRAL_FILE_NAME_TEMPLATE = "{operating_institution}_{start_date}_{site_id}_DM_WAVE-SPECTRA_{end_date}.nc"
NC_BULK_FILE_NAME_TEMPLATE = "{operating_institution}_{start_date}_{site_id}_DM_WAVE-PARAMETERS_{end_date}.nc"
NC_DISPLACEMENTS_FILE_NAME_TEMPLATE = "{operating_institution}_{start_date}_{site_id}_DM_WAVE-RAW-DISPLACEMENTS_{end_date}.nc"