import os

import xarray as xr
import numpy as np
import glob as glob
from dotenv import load_dotenv

from wavebuoy_nrt.qc.qcTests import WaveBuoyQC

load_dotenv()

files = glob.glob(os.path.join(os.getenv("REQC_PATH"), "**", "*WAVE-PARAMETERS_monthly.nc"), recursive=True)

if not files:
    raise FileNotFoundError(f"No files were found in desired path. Check .env file. path: {os.getenv("REQC_PATH")}")

files_qualified = []
for file in files:
    print("\n")
    print(os.path.basename(file))
    ds = xr.open_dataset(file)

    if ds.TIME.size in (0,1):
        print(f"File has {ds.TIME.size} datapoints, skipping")
        continue

    qc = WaveBuoyQC(config_id=1)

    df = ds.to_dataframe().reset_index()
    df['processing_source'] = np.nan

    qc.load_data(data=df)
    parameters_to_qc = qc.get_parameters_to_qc(data=df, qc_config=qc.qc_config)
    qualified_data_embedded, waves_subflags = qc.qualify(data=df,
                                            parameter_type="waves",
                                        parameters=parameters_to_qc,
                                        window = "all",
                                        gross_range_test=True,
                                        rate_of_change_test=True)

    ds_requalified = ds.copy()
    del ds

    new_values = qualified_data_embedded["WAVE_quality_control"].to_numpy()

    if new_values.size != ds_requalified["WAVE_quality_control"].size:
        print("QC data length does not match the dataset variable length.")
        continue

    ds_requalified['WAVE_quality_control'].data = new_values
    
    ds_requalified.to_netcdf(file)
    del ds_requalified
    
    files_qualified.append(file)

