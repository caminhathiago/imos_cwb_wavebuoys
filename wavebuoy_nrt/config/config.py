OPERATING_INSTITUTIONS = {"UWA":"UWA",
                          "Deakin":"VIC-DEAKIN-UNI",
                          "NSW-DCCEEW" : "NSW-DPE",
                          "IMOS":"IMOS_COASTAL-WAVE-BUOYS",
                          "SARDI": "SARDI"
                          }

NC_FILE_NAME_TEMPLATE = "{operating_institution}_{monthly_datetime}_{site_id}_RT_WAVE-PARAMETERS_monthly.nc"
NC_SPECTRAL_FILE_NAME_TEMPLATE = "{operating_institution}_{monthly_datetime}_{site_id}_RT_WAVE-SPECTRA_monthly.nc"

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
    "value": 'TEMP' # SM surface temperature
}

AODN_SPECTRAL_COLUMNS_TEMPLATE = {
    "timestamp": 'TIME',
    "longitude": 'LONGITUDE',
    "latitude": 'LATITUDE',
    "frequency": 'FREQUENCY',
    "a1": 'A1',
    "b1": 'B1',
    "a2": 'A2',
    "b2": 'B2',
    "varianceDensity": 'ENERGY',
    "direction": 'DIRECTION',
    "directionalSpread": 'DIRSPREAD',
    "df": "DIFFREQUENCY"
}