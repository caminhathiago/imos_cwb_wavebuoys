# Sofar Spotter DS Card Processor

This repository provides tools to process data exported from SD cards used by **Sofar Spotter** wave buoys. The main script extracts, filters, and calculates spectral data from raw log files within a specified deployment window.

---

## Quick Start

### 1. Install Dependencies

<pre> ```pip install -r requirements.txt``` </pre>

### 2. Run the script

#### Script arguments

| Argument               | Description                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `-l`, `--log-path`     | **(Required)** Directory where the Spotter SD card files are stored.                                            |
| `-d`, `--deploy-dates` | **(Required)** Start and end datetime of the deployment period in the format `YYYYmmddTHHMMSS YYYYmmddTHHMMSS`. |
| `-ed`, `--enable-dask` | *(Optional)* Enables parallel spectra calculation using Dask.                                                   |
| `-ot`, `--output-type` | *(Optional)* Choose output format: `csv` or `netcdf` (default: `netcdf`).                                       |


### Execution example

<pre> ```bash python scripts/cwb-spotter-dm.py \ --log-path /path/to/sdcard/files \ --deploy-dates 20240101T000000 20240131T235959 \ --output-type netcdf \ --enable-dask ``` </pre>
