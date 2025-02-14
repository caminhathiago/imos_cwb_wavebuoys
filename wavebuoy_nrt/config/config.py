FILES_OUTPUT_PATH = r"C:\Users\00116827\cwb\wavebuoy_aodn\output_path\final_testing"
FILES_PATH = r"C:\Users\00116827\cwb\wavebuoy_aodn\wavebuoy_nrt"
IRDS_PATH = r"\\drive.irds.uwa.edu.au\OGS-COD-001\CUTTLER_wawaves"

NC_FILE_NAME_TEMPLATE = "{operating_institution}_{monthly_datetime}_{site_id}_RT_WAVE-PARAMETERS_monthly.nc"

OPERATING_INSTITUTIONS = {"UWA":"",
                          "Deakin":"VIC-DEAKIN-UNI",
                          "NSW-DCCEEW" : "NSW-DPE",
                          "IMOS":"IMOS_COASTAL-WAVE-BUOYS"
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
    "degrees": 'SST', # Spotter SST
    "value": 'SST' # SM surface temperature
}