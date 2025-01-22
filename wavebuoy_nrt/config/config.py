FILES_OUTPUT_PATH = r"C:\Users\00116827\cwb\wavebuoy_aodn\output_path"
FILES_PATH = r"C:\Users\00116827\cwb\wavebuoy_aodn\wavebuoy_nrt"
NC_FILE_NAME_TEMPLATE = "{institution}_{monthly_datetime}_{site_id}_RT_WAVE-PARAMETERS_monthly.nc"

REGION_TO_INSTITUTION = {"VIC":"VIC-DEAKIN-UNI",
                         "WA":"IMOS_COASTAL-WAVE-BUOYS"}

# AODN_COLUMNS_TEMPLATE = {
#     'TIME': "timestamp",
#     'timeSeries': None,
#     'LATITUDE': "longitude",
#     'LONGITUDE': "latitude",
#     'SSWMD': "meanDirection",
#     'WAVE_quality_control': None,
#     'WMDS': "meanDirectionalSpread",
#     'WPDI': "peakDirection",
#     'WPDS': "peakDirectionalSpread",
#     'WPFM': "meanPeriod",
#     'WPPE': "peakPeriod",
#     'WSSH': "significantWaveHeight"
# }

AODN_COLUMNS_TEMPLATE = {
    "timestamp": 'TIME',
    # None: ['timeSeries', 'WAVE_quality_control'],
    "longitude": 'LONGITUDE',
    "latitude": 'LATITUDE',
    "meanDirection": 'SSWMD',
    "meanDirectionalSpread": 'WMDS',
    "peakDirection": 'WPDI',
    "peakDirectionalSpread": 'WPDS',
    "meanPeriod": 'WPFM',
    "peakPeriod": 'WPPE',
    "significantWaveHeight": 'WSSH',
    "degrees": 'SST',
    "value": 'SST'
}