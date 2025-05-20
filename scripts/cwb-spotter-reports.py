import os
from datetime import datetime, timedelta
import re

from dotenv import load_dotenv
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt

from wavebuoy_nrt.wavebuoy import WaveBuoy
from wavebuoy_nrt.sofar.api import SofarAPI
from wavebuoy_nrt.qc.qcTests import WaveBuoyQC
from wavebuoy_nrt.netcdf.writer import ncWriter, ncAttrsComposer, ncAttrsExtractor, ncProcessor, ncMetaDataLoader
from wavebuoy_nrt.utils import args_processing, IMOSLogging, generalTesting, csvOutput
from wavebuoy_nrt.alerts.email import Email


load_dotenv()


def extract():
            
    meta_data_loader = ncMetaDataLoader(buoys_metadata=wb.buoys_metadata)
    deployment_metadata = meta_data_loader.load_latest_deployment_metadata(site_name=site.name)
    regional_metadata = meta_data_loader.load_regional_metadata()
    
    nc_files_available = wb.get_available_nc_files(site_id=site.name,
                                                    files_path=vargs.incoming_path,
                                                    deployment_metadata=deployment_metadata,
                                                    parameters_type="bulk")
    
    csv_files_available = glob.glob(os.path.join(site_path, "csv/*subflags.csv"))

    if nc_files_available:
        nc_files_available = sorted(nc_files_available, key=lambda x: re.search(r'\d{8}', x).group(0))
        csv_files_available = sorted(csv_files_available, key=lambda x: re.search(r'\d{8}', x).group(0))
    else:
        raise FileExistsError("no NC files available for this site yet.")

    return nc_files_available, csv_files_available


def run_tests(nc_files_available, csv_files_available, site_path):
    reports_path = os.path.join(site_path, "reports")
    if not os.path.exists(reports_path):
        os.mkdir(reports_path)

    for nc, csv in zip(nc_files_available, csv_files_available):
        
        datetime_prefix = os.path.basename(csv)[0:8]
        ds = xr.open_dataset(nc)
        df = pd.read_csv(csv)
        
        # flags report
        qc_report_subflags = df.filter(regex="_test").apply(pd.Series.value_counts).fillna(0).astype(int).T
        qc_report_subflags.index = qc_report_subflags.index.str.replace("WAVE_QC_", "").str.replace("TEMP_QC_", "")
        qc_report_primary_flag = df.filter(regex="_quality_control").astype(int).apply(pd.Series.value_counts).fillna(0).astype(int).T

        qc_report = pd.concat([qc_report_subflags, qc_report_primary_flag])
        qc_report.to_csv(os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_qc_report.csv"))

        # plots
        (ds.to_dataframe()
            .plot(subplots=True,
                figsize=(15, len(ds.to_dataframe().columns)*1.5),
                marker='o', grid=True, color="k")
        )
        plt.tight_layout()
        plt.savefig(os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_nc_subplots.png"))
        plt.close()
        
        (df
            .plot(subplots=True,
                figsize=(15, len(df.columns)*1.5),
                marker='o', grid=True, color="k")
        )
        plt.tight_layout()
        plt.savefig(os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_csv_subplots.png"))
        plt.close()

    print(f"Sucssesful reporting for {site_name.upper()}")
       
if __name__ == "__main__":
    # Args handling
    vargs = args_processing()

    # Start general logging
    general_log_file = os.path.join(vargs.incoming_path, "logs", f"general_{os.path.basename(__file__).removesuffix(".py")}.log") # f"{runtime}_general_process.log"
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name="general_logger",
                                                logging_filepath=general_log_file)
    
    wb = WaveBuoy(buoy_type="sofar")
    sofar_api = SofarAPI(buoys_metadata=wb.buoys_metadata)    
    imos_logging = IMOSLogging() 
    
    if vargs.site_to_process:
        wb.buoys_metadata = wb.buoys_metadata.loc[wb.buoys_metadata.index.isin(vargs.site_to_process)].copy()
    
    for idx, site in wb.buoys_metadata.iterrows():
        
        site_name = site.name.replace("_", "")
        site_path = os.path.join(vargs.incoming_path, "sites", site_name)

        if not os.path.exists(os.path.join(vargs.incoming_path, "sites", site_name)):
                continue
        
        try:
            nc_files_available, csv_files_available = extract()
            run_tests(nc_files_available, csv_files_available, site_path)

            print(f"")
        except Exception as e:
            print(f"Error processing {site_name.upper()}: {str(e)}")
            continue