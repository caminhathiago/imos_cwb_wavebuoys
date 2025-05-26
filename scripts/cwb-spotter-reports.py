import os
from datetime import datetime, timedelta
import re
import traceback

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
    
    nc_bulk = wb.get_available_nc_files(site_id=site.name,
                                                    files_path=vargs.incoming_path,
                                                    deployment_metadata=deployment_metadata,
                                                    parameters_type="bulk")
    
    nc_spectra = wb.get_available_nc_files(site_id=site.name,
                                                    files_path=vargs.incoming_path,
                                                    deployment_metadata=deployment_metadata,
                                                    parameters_type="spectral")
    
    csv_bulk = glob.glob(os.path.join(site_path, "csv/*subflags.csv"))

    if nc_bulk:
        nc_bulk = sorted(nc_bulk, key=lambda x: re.search(r'\d{8}', x).group(0))
        nc_spectra = sorted(nc_spectra, key=lambda x: re.search(r'\d{8}', x).group(0))
        csv_bulk = sorted(csv_bulk, key=lambda x: re.search(r'\d{8}', x).group(0))
    else:
        raise FileExistsError("no NC files available for this site yet.")

    return nc_bulk, nc_spectra, csv_bulk, deployment_metadata, regional_metadata

def regional_metadata_validation(reports_path, nc_path, ds, regional_metadata, datetime_prefix):
 
    file_prefix = os.path.basename(nc_path[0]).split("_")[0]

    # Regional Metadata
    match_regional = regional_metadata.loc[
        regional_metadata["operating_institution_nc_preffix"] == file_prefix
    ].squeeze() 

    checks = [
        ("project", "project"),
        ("operating_institution_long_name", "institution"),
        ("principal_investigator", "principal_investigator"),
        ("principal_investigator_email", "principal_investigator_email"),
    ]

    # Compare and store results: (field_regional, field_bulk, match_boolean)
    check_results = [
        (reg, bulk, match_regional[reg] == ds.attrs.get(bulk))
        for reg, bulk in checks
    ]
    
    all_match = all(match for _, _, match in check_results)

    log_lines = []

    if not all_match:
        log_lines.append(f"Mismatch detected for file: {os.path.basename(nc_path[0])}")
        for reg, bulk, match in check_results:
            if not match:
                expected = match_regional[reg]
                got = ds.attrs.get(bulk)
                log_lines.append(f" • {bulk}: expected {expected!r}, got {got!r}")
    else:
        log_lines.append(f"All fields match for file: {os.path.basename(nc_path[0])}")

    reg_validation_txt_path = os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_regional_metadata_validation_log.txt") 
    with open(reg_validation_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n\n")

# def deployment_metadata_validation(reports_path, nc_path, ds, deployment_metadata, datetime_prefix):
 
#     file_prefix = os.path.basename(nc_path[0]).split("_")[0]

#     # Regional Metadata
#     match_regional = deployment_metadata.loc[
#         deployment_metadata["Operating institution"] == file_prefix
#     ].squeeze() 

#     checks = [
#         ("project", "project"),
#         ("operating_institution_long_name", "institution"),
#         ("principal_investigator", "principal_investigator"),
#         ("principal_investigator_email", "principal_investigator_email"),
#     ]

#     # Compare and store results: (field_regional, field_bulk, match_boolean)
#     check_results = [
#         (reg, bulk, match_regional[reg] == ds.attrs.get(bulk))
#         for reg, bulk in checks
#     ]
    
#     all_match = all(match for _, _, match in check_results)

#     log_lines = []

#     if not all_match:
#         log_lines.append(f"Mismatch detected for file: {os.path.basename(nc_path[0])}")
#         for reg, bulk, match in check_results:
#             if not match:
#                 expected = match_regional[reg]
#                 got = ds.attrs.get(bulk)
#                 log_lines.append(f" • {reg}: expected {expected!r}, got {got!r}")
#     else:
#         log_lines.append(f"All fields match for file: {os.path.basename(nc_path[0])}")

#     # At the end of the process (outside your loop), write the log to file:
#     reg_validation_txt_path = os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_deployment_metadata_validation_log.txt") 
#     with open(reg_validation_txt_path, "a", encoding="utf-8") as f:
#         f.write("\n".join(log_lines) + "\n\n")

def qc_flags_report(reports_path, ds, df, datetime_prefix):
    # flags report
    qc_report_subflags = df.filter(regex="_test").apply(pd.Series.value_counts).fillna(0).astype(int).T
    qc_report_subflags.index = qc_report_subflags.index.str.replace("WAVE_QC_", "").str.replace("TEMP_QC_", "")
    qc_report_primary_flag = df.filter(regex="_quality_control").astype(int).apply(pd.Series.value_counts).fillna(0).astype(int).T

    qc_report = pd.concat([qc_report_subflags, qc_report_primary_flag])
    qc_report.to_csv(os.path.join(reports_path, f"{datetime_prefix}_{site_name.upper()}_qc_report.csv"))

def generate_plots(reports_path, ds, df, datetime_prefix):
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

def run_tests(site_path, nc_bulk, nc_spectra, csv_bulk, deployment_metadata, regional_metadata):
    reports_path = os.path.join(site_path, "reports")
    if not os.path.exists(reports_path):
        os.mkdir(reports_path)

    if not nc_spectra:
        nc_spectra = nc_spectra or [None] * len(nc_bulk)

    for nc_b, nc_s, csv_b in zip(nc_bulk, nc_spectra, csv_bulk):
        
        datetime_prefix = os.path.basename(csv_b)[0:8]
        ds_bulk = xr.open_dataset(nc_b)
        df = pd.read_csv(csv_b)
        if nc_s:
            ds_spectra = xr.open_dataset(nc_s)
        
        # regional_metadata_validation(reports_path, nc_bulk, ds_bulk, regional_metadata, datetime_prefix)
        qc_flags_report(reports_path, ds_bulk, df, datetime_prefix)
        generate_plots(reports_path, ds_bulk, df, datetime_prefix)

    print(f"Reporting successfull for {site_name.upper()}")

       
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
            payload_tests = extract()
            run_tests(site_path, *payload_tests)

            print(f"")
        except Exception as e:
            print(f"Error processing {site_name.upper()}:")
            traceback.print_exc()
            continue