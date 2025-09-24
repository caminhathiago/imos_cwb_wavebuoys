# Sofar Spotter SD Card Processor

This repository provides tools to process data exported from SD cards used by **Sofar Spotter** wave buoys. The main script extracts, filters, and calculates spectral data from raw log files.

---

## Quick Start

### 1. Install package

With your preferred python (>=3.12,<3.13) virtual environment activated, from the repo's root:

<pre> pip install . </pre>

### 2. Processing script

The script `wavebuoy_dm/dm_processing.py` can be used to process data either in standalone or programmatic approaches, both described in the sections 2.1 and 2.2.

#### 2.1 Standalone approach

The script cab be directly executed from the command line by providing the following arguments:

##### Arguments

| Argument               | Description                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `-l`, `--log-path`     | **(Required)** Directory where the Spotter SD card files are stored.                                            |
| `-d`, `--deploy-dates` | *(Optional)* Start and end datetime of the deployment period in the format `YYYYmmddTHHMMSS YYYYmmddTHHMMSS`. |
| `-u`, `--utc-offset` | *(Optional)* UTC offset of defined deploy-dates. Defaults to 0 (UTC).|
| `-ed`, `--enable-dask` | *(Optional)* Enables spectra calculation with Dask for faster processing using parallelism                                 |
| `-ot`, `--output-type` | *(Optional)* Choose output format: `csv` or `netcdf` (default: `netcdf`).                                       |


##### Execution example

<pre> python scripts/cwb-spotter-dm.py \
  --log-path /path/to/sdcard/files \
  --deploy-dates 20240101T000000 20240131T235959 \
  --utc-offset 8 \
  --output-type netcdf \
  --enable-dask  </pre>

#### 2.2 Programmatic usage

The processing tool can also be imported within an external python script or jupyter notebook. Please note for Windows machines the following must be wraped in a top-level code environment (i.e. inside a if __name__ == "__main__" block) to prevent conflicts between dask and Windows multiprocessing.

In a Jupyter notebook:
```python
from wavebuoy_dm.dm_processor import DMSpotterProcessor

# config arguments follow the same structure and requirements as the standalone approach
config = {
    "log_path": "/path/to/sdcard/files", # if not provided, the processor will try to fing SD card files in the working directory
    "utc_offset": 0,
    "deploy_dates": ["20240101T000000", "20240131T235959"], # pass start and end dates as a list formatted as ISO 8601 (YYYY-mm-ddTHH:MM:SS)
    "enable_dask": True,
    "output_type": "netcdf"
}

dm = DMSpotterProcessor(config, to_process=["displacements", "gps", "surface_temp", "atmospheric_pressure"])
dm.run(save_outputs=True)
```

In an executable python script:
```python
def main():
    from wavebuoy_dm.dm_processor import DMSpotterProcessor

    config = {
        "log_path": "/path/to/sdcard/files", # if not provided, the processor will try to fing SD card files in the working directory
        "enable_dask": True,
        "output_type": "netcdf",
        }

    dm = DMSpotterProcessor(config, to_process=["displacements", "gps", "surface_temp", "atmospheric_pressure"])
    dm.run(save_outputs=True)

if __name__ == "__main__":
    main()
```

At the end of the processing execution, the defined instance (e.g. `dm`) offers the attributes below. Dataset types are determined pending on what `output_type` was passed in the config dictionary (`csv` -> `polars.DataFrame`, `netcdf` -> `xarray.Dataset`).

| Attribute               | Type| Description                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `bulk`  | Data | results for waves integral (bulk) parameters, may contain surface temperature data if present and processed (`polars.DataFrame` or `xarray.Dataset`)   |
| `spectra` | Data | results for waves spectra (`polars.DataFrame` or `xarray.Dataset`) |
| `disp` | Data | results for raw displacements (`polars.DataFrame` or `xarray.Dataset`) |
| `gps` | Data | results for raw displacements (`polars.DataFrame`) |
| `surface_temp` | Data | results for raw displacements (`polars.DataFrame`) |
| `barometer` | Data| results for raw displacements (`polars.DataFrame`) |
| `suffixes_to_process` | Config | SD card Files suffixes to be processed (e.g. 'FLT', 'LOC') |
| `spectra_variables` | Config | variables present in bulk dataset |
| `bulk_variables` | Config | variables present in spectra dataset  |
| `client` | Config | dask client if provided |
| `spectra_info` | Config | configuration parameters to spectra calculations (to be documented) |
| `vargs` | Config | various general configuration parameters (extracted from config or vargs passed when using standalone approach) |


#### Outputs

Processing results are saved in the output directory `processed` created in the specified in the configuration (see below). For programmatic usage, outputs are saved only if save_outputs is set to True in the run method.

 <pre> /path/to/sdcard/files/processed/ </pre>

Files created are listed below:

 | File name              | Description  |  Extension |
| ---------------------- | -------------------| -----------------|
| `"logs/DM_spotter_processing"`     | Processing log for debugging purposes | `.log` | 
| `raw-displacements`     | Contains raw displacements as `Z` (heave), `Y` (east), `X` (north) and `TIME` (datetime) | `.nc` or `.csv` | 
| `gps` | Contains gps data as `LATITUDE`, `LONGITUDE` and `TIME` (datetime) | `.csv` | 
| `bulk` |  Contains calculated bulk wave parameters |  `.nc` or `.csv` | 
| `spectra` | Contains calculated spectra wave parameters | `.nc` or `.csv` | 
