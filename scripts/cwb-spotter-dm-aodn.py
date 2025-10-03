import os
from dataclasses import dataclass
from datetime import datetime

from multiprocessing import cpu_count
import time
import polars as pl
import pandas as pd
from dask.distributed import Client
import dask
import numpy as np
from dotenv import load_dotenv

from wavebuoy_dm.spectra import Spectra
from wavebuoy_dm.processing.concat import csvConcat
from wavebuoy_dm.processing.process import csvProcess
from wavebuoy_dm.netcdf.process import ncSpectra, ncDisp, ncBulk, Process
from wavebuoy_dm.netcdf.writer import ncWriter, ncAttrsComposer
from wavebuoy_dm.wavebuoy import WaveBuoy
from wavebuoy_dm.utils import IMOSLogging, args_aodn_processing, Plots
from wavebuoy_dm.qc.qc import WaveBuoyQC


@dataclass
class MetadataArgs:
    site_name: str
    deployment_metadata: pd.DataFrame
    buoys_metadata: pd.DataFrame
    regional_metadata: pd.DataFrame
    site_buoys_to_process: pd.DataFrame
    deploy_start: datetime
    deploy_end: datetime
    raw_data_path: str
    output_path: str

def process_paths(site_buoys_to_process):

    dm_deployment_path = site_buoys_to_process.loc["datapath"]
    # dm_deployment_path = os.path.dirname(dm_deployment_path
    #                                     .replace("Y:", "\\\\drive.irds.uwa.edu.au\\OGS-COD-001")
    #                                     .replace("X:", "\\\\drive.irds.uwa.edu.au\\OGS-COD-001")
    #                                     )
    output_path = os.path.join(dm_deployment_path, "processed_py")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    return dm_deployment_path, output_path

def load_metadata(site_buoys_to_process:pd.DataFrame, dm_deployment_path) -> list[pd.DataFrame]:

    raw_data_path = dm_deployment_path
    if os.path.basename(dm_deployment_path) != "log":                         
        raw_data_path = os.path.join(raw_data_path, 'log')
    
    if not os.path.exists(dm_deployment_path):
        raise FileNotFoundError(f"path {raw_data_path} does not exist. Check if deployment path exists, and if SD card data is in the folder log.")

    deploy_start, deploy_end, spot_id = WaveBuoy().extract_deploy_dates_spotid_from_path(site_buoys_to_process.datapath)

    site_name, region = WaveBuoy().extract_region_site_name(path=raw_data_path)
    buoys_metadata = WaveBuoy()._get_buoys_metadata(buoy_type='sofar', buoys_metadata_file_name="buoys_metadata.csv")
    deployment_metadata = WaveBuoy().load_latest_deployment_metadata(site_name=site_name, region=region)
    deployment_metadata.loc["instrument_burst_duration", "metadata_wave_buoy"] = site_buoys_to_process.instrument_burst_duration
    deployment_metadata.loc["instrument_burst_interval", "metadata_wave_buoy"] = site_buoys_to_process.instrument_burst_interval
    regional_metadata = WaveBuoy().load_regional_metadata()

    return {
            "site_name": site_name,
            "region": region,
            "buoys_metadata": buoys_metadata,
            "deployment_metadata": deployment_metadata,
            "regional_metadata": regional_metadata,
            "deploy_start": deploy_start,
            "deploy_end": deploy_end,
            "spot_id": spot_id,
            "raw_data_path": raw_data_path,
        }

def process_from_SD(raw_data_path, suffixes_to_concat=["FLT","LOC","SST","BARO"]) -> list[pl.DataFrame]:

    # DEP_LOGGER.info(f"Lazy concatenating csv files for {suffixes_to_concat}")
    # cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=suffixes_to_concat)
    # lazy_concat_results = cc.lazy_concat_files()

    # cp = csvProcess()
    # DEP_LOGGER.info("Lazy processing cocatenated csv files")
    # lazy_processed_results = cp.process_concat_results(lazy_concat_results)

    # DEP_LOGGER.info("Collecting processed csv files")
    # collected_results = cp.collect_results(lazy_processed_results)   

    # return collected_results["displacements"], collected_results["gps"], collected_results["surface_temp"]
    DEP_LOGGER.info(f"Lazy concatenating csv files for suffixes_to_concat")
    cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=suffixes_to_concat) 
    lazy_concat_results = cc.lazy_concat_files()

    cp = csvProcess()
    DEP_LOGGER.info("Lazy processing cocatenated csv files")
    lazy_processed_results = cp.process_concat_results(lazy_concat_results)

    DEP_LOGGER.info("Collecting processed csv files")
    collected_results = cp.collect_results(lazy_processed_results)

    if isinstance(collected_results.get("surface_temp"), pl.DataFrame) and not collected_results["surface_temp"].is_empty():
        collected_results["surface_temp"] = collected_results["surface_temp"].rename(
            {"temperature": "TEMP", "datetime": "TIME_TEMP"}
        )

    if isinstance(collected_results.get("barometer"), pl.DataFrame) and not collected_results["barometer"].is_empty():
        collected_results["barometer"] = collected_results["barometer"].rename(
            {"baro_pressure": "ATM_PRESSURE", "datetime": "TIME_ATM_PRESSURE"}
        )

    return collected_results

def filter_dates(results, utc_offset, deploy_start, deploy_end, time_crop_start, time_crop_end) -> list[pl.DataFrame]:

    cp = csvProcess()
    # DEP_LOGGER.info(f"Filtering displacements data with passed deployment datetimes: {deploy_start} - {deploy_end}")
    # disp = cp.filter_deployment_dates(dataframe=disp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
    #                                   time_crop_start=time_crop_start, time_crop_end=time_crop_end)
    
    # DEP_LOGGER.info(f"Filtering gps data with passed deployment datetimes: {deploy_start} - {deploy_end}")    
    # temp = cp.filter_deployment_dates(dataframe=temp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
    #                                    time_crop_start=time_crop_start, time_crop_end=time_crop_end)

    # # gps = cp.filter_deployment_dates(dataframe=gps, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
    # #                                      time_crop_start=time_crop_start, time_crop_end=time_crop_end)
    
    for key, dataframe in results.items():
        if key != 'gps' and isinstance(dataframe, pl.DataFrame):
            DEP_LOGGER.info(f"Filtering {key} data with passed deployment datetimes: {deploy_start} - {deploy_end}")
            result = cp.filter_deployment_dates(dataframe=dataframe, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                            time_crop_start=time_crop_start, time_crop_end=time_crop_end)
            results[key] = result 
    
    time_minutes = 2
    DEP_LOGGER.info(f"Buffering gps times by {time_minutes} minutes")
    results['gps'] = cp.buffer_gps_times(disp=results['displacements'], gps=results['gps'], time_minutes=time_minutes)
    
    if results['gps'].is_empty():
        raise ValueError(f"Buffering GPS based on displacements didn't work as resulting gps dataset is empty.")

    return results

def align_gps(spectra_bulk_df, gps) -> pl.DataFrame:

    cp = csvProcess()

    DEP_LOGGER.info(f"Interpolating gps to match spectra_bulk data")
    spectra_bulk_df = cp.interpolate_lat_lon(spectra_bulk_df, gps)

    return spectra_bulk_df

def qc_watch_circle(spectra_bulk_df, site_buoys_to_process:pd.DataFrame, output_path:str, deployment_metadata:pd.DataFrame):
    
    s = site_buoys_to_process
    
    cp = csvProcess()

    DEP_LOGGER.info(f"Calculating mooring setting")
    mainline, catenary, watch_circle = cp.calculate_watch_circle(s)

    DEP_LOGGER.info(f"Mainline = {mainline} | Catenary = {catenary} | Watch Circle = {round(watch_circle,2)}")


    DEP_LOGGER.info(f"Qualifying with respect to watch circle")
    spectra_bulk_df, out_of_radious_pct = cp.qc_watch_circle(
        spectra_bulk_df,
        deploy_lat=s.DeployLat,
        deploy_lon=s.DeployLon,
        watch_circle=watch_circle,
        watch_circle_fail=s.watch_circle_fail
    )

    DEP_LOGGER.info(f"Saving watch circle qc results after reprocessing")
    spectra_bulk_csv_path = cp.save_qc_watch_circle_csv(spectra_bulk_df, output_path)

    DEP_LOGGER.info(f"Extracting drifting periods")
    cp.extract_drifting_periods(spectra_bulk_df, output_path)

    if out_of_radious_pct >= s.out_of_radius_tolerance:
        
        DEP_LOGGER.info(f"First watch circle QC not satisfactory, {out_of_radious_pct}% out of radius (> tolerance = {s.out_of_radius_tolerance})")
        DEP_LOGGER.info(f"Reaplying strech factor over calculated mooring setting")
        
        mainline, catenary, watch_circle = cp.calculate_watch_circle(s, reprocess=True)

        DEP_LOGGER.info(f"Requalifying with respect to watch circle (={watch_circle})")
        spectra_bulk_df, out_of_radious_pct = cp.qc_watch_circle(
                                                    spectra_bulk_df,
                                                    deploy_lat=s.DeployLat,
                                                    deploy_lon=s.DeployLon,
                                                    watch_circle=watch_circle,
                                                    watch_circle_fail=s.watch_circle_fail
                                                )

        DEP_LOGGER.info(f"Saving watch circle qc results after reprocessing")
        spectra_bulk_csv_path = cp.save_qc_watch_circle_csv(spectra_bulk_df, output_path)
        
        DEP_LOGGER.info(f"Extracting drifting periods after reprocessing")
        cp.extract_drifting_periods(spectra_bulk_df, output_path, reprocess=True)

        DEP_LOGGER.info(f"Requalification successful, {out_of_radious_pct}% out of radius (< tolerance = {s.out_of_radius_tolerance}")
        

    p = Plots(site_name=site_buoys_to_process.loc['name'],
                  deployment_folder=site_buoys_to_process.loc["datapath"],
                  output_path=output_path)
    p.map_positions_shapefiles(data=spectra_bulk_df.to_pandas(),
                               deployment_center=(s.DeployLat,s.DeployLon),
                               watch_circle=watch_circle,
                               map_coverage=(800,40,10,5,1,.3),
                               figsize=(15,5))

    if out_of_radious_pct >= s.out_of_radius_tolerance: 
        raise ValueError(f"{out_of_radious_pct}% out of radius (watchcircle = {round(watch_circle,2)}), greater then tolerance ({s.out_of_radius_tolerance}%). CSV with qc_flag_watch saved as {spectra_bulk_csv_path}")

    DEP_LOGGER.info(f"Storing watch_circle in deployment metadata")
    deployment_metadata.loc["watch_circle", "metadata_wave_buoy"] = round(watch_circle,2)

    return spectra_bulk_df, deployment_metadata

def filter_watch_circle(disp, gps, site_buoys_to_process) -> list[pl.DataFrame]:
    
    """
    1. calculate distance between metadata centroid and mean centroid, create one column for each
    2. calculate watch circle from metadata: sqrt( buoy_info.mainline_length^2 - buoy_info.DeployDepth^2) + buoy_info.catenary_length;
    3. create watch_circle_multiplier variable = 1.25
    4. 
    
    """

    


    cp = csvProcess()
    deploy_lat, deploy_lon = site_buoys_to_process.loc['DeployLat'], site_buoys_to_process.loc['DeployLon']
    watch_circle = site_buoys_to_process.loc['watch_circle_manual']

    gps_renamed = gps.rename({'latitude':'LATITUDE', 'longitude':'LONGITUDE'})
    gps_filtered, percentage_cropped = cp.filter_watch_circle_geodesic(dataframe=gps_renamed, deploy_lat=deploy_lat, deploy_lon=deploy_lon, max_distance=watch_circle,
                                                   percentage_threshold=.97)
    DEP_LOGGER.info(f"watch circle crop = {round(percentage_cropped*100,2)}% of data")
    gps = gps_filtered.rename({'LATITUDE':'latitude', 'LONGITUDE':'longitude'})
    
    disp = cp.filter_deployment_dates(dataframe=disp, 
                                      deploy_start=gps['datetime'][0], 
                                      deploy_end=gps['datetime'][-1], hours_crop=3)

    return disp, gps

def calculate_spectra_from_displacements(disp: pl.DataFrame, enable_dask:bool):  

    s = Spectra()
    
    DEP_LOGGER.info(f"Setting spectra calculation parameters")
    info = {
        "hab": None,
        "fmaxSS": 1/8,
        "fmaxSea": 1/2,
        "bad_data_thresh": 2/3,
        "hs0_thresh": 3,
        "t0_thresh": 5
    }
    fs = 2.5
    min_samples = s.calculate_min_samples(fs=fs, spec_window=30)
    nfft = s.calculate_nfft(fs=fs, spec_window=30)
    merge = s.define_merging(nfft=nfft)
    nover = 0.5

    if enable_dask:
        DEP_LOGGER.info(f"Processing dask chunks for spectra calculation")
        disp_chunks_30m = s.generate_time_chunks(disp, time_chunk='30m')
        disp_chunks_30m = s.filter_insufficient_samples(disp_chunks_30m, min_samples)
        
        DEP_LOGGER.info(f"Splitting dask chunks by the number of workers (= {num_workers})")
        disp_dask_chunks = s.generate_dask_chunks(disp_chunks_30m, num_workers)

        DEP_LOGGER.info(f"Creating dask tasks")
        disp_tasks = [dask.delayed(s.process_dask_chunk)
                (dask_chunk, nfft, nover, fs, merge, 'xyz', info) 
                for dask_chunk in disp_dask_chunks]

        DEP_LOGGER.info(f"Computing dask tasks")
        spectra_bulk_results = dask.compute(*disp_tasks)

        DEP_LOGGER.info(f"Concatenating dask computed results")
        spectra_bulk_df = pl.concat(spectra_bulk_results)

        return spectra_bulk_df
    
    else:
        DEP_LOGGER.info(f"Calculating spectra with whole displacements data")
        return s.spectra_from_dataframe(disp, nfft, nover, fs, merge, 'xyz', min_samples, info)

def generate_spectra_NC_file(spectra_bulk_df,
                            gps,
                            site_name,
                            deployment_metadata,
                            buoys_metadata,
                            regional_metadata,
                            site_buoys_to_process,
                            deploy_start,
                            deploy_end,
                            raw_data_path,
                            output_path
                            ) -> None:
    
    
    # Process Spectra -------------------------------
    spectra = Spectra().select_parameters(spectra_bulk_df, dataset_type="spectra")    
    
    # cp = csvProcess()
    # deploy_lat, deploy_lon = site_buoys_to_process.loc['DeployLat'], site_buoys_to_process.loc['DeployLon']
    # watch_circle = site_buoys_to_process.loc['watch_circle']

    # spectra = cp.interpolate_lat_lon(dataframe=spectra, locations_dataframe=gps)
    # spectra = cp.filter_watch_circle_geodesic(dataframe=spectra, deploy_lat=deploy_lat, deploy_lon=deploy_lon, max_distance=watch_circle)
    
    # spectra = cp.filter_deployment_dates(dataframe=spectra, deploy_start=deploy_start, deploy_end=deploy_end)


    # Generate Spectra NC File ------------------------
    DEP_LOGGER.info(f"Composing dataset")
    spectra_ds = ncSpectra().compose_dataset(spectra)
    
    DEP_LOGGER.info(f"Loading attributes")
    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="spectral")
    
    DEP_LOGGER.info(f"Assigning general attributes")
    spectra_ds = nc_atts_comp.assign_general_attributes(dataset=spectra_ds, site_name=site_name)
    # DEP_LOGGER.info("general attributes assigned to embedded dataset")
    
    DEP_LOGGER.info(f"Creating timeseries variable")
    spectra_ds = ncSpectra().create_timeseries_variable(dataset=spectra_ds)
    # DEP_LOGGER.info("time series variable created")

    DEP_LOGGER.info(f"Extracting time periods")
    start_end_dates = ncWriter().extract_start_end_dates_list(dataset_objects=(spectra_ds,))

    DEP_LOGGER.info(f"Converting time to CF conventions")
    spectra_ds = ncSpectra().convert_time_to_CF_convention(spectra_ds)
     # DEP_LOGGER.info("dataset objects time dimension processed to conform to CF conventions")

    DEP_LOGGER.info(f"Assigning variables attributes")
    spectra_ds = nc_atts_comp.assign_variables_attributes(dataset=spectra_ds)
    # DEP_LOGGER.info("variables attributes assigned to datasets")
    
    DEP_LOGGER.info(f"Composing file names")
    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name,
                                periods=start_end_dates,
                                deployment_metadata=deployment_metadata,
                                parameters_type="spectral")
    
    DEP_LOGGER.info(f"Saving datasets as {nc_file_names}")
    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=(spectra_ds,)
                        )
    
    DEP_LOGGER.info(f"WAVE-SPECTRA netCDFs successfully generated")
    
def generate_bulk_NC_file(spectra_bulk_df,
                        gps,
                        temp,
                        site_name,
                        deployment_metadata,
                        buoys_metadata,
                        regional_metadata,
                        site_buoys_to_process,
                        deploy_start,
                        deploy_end,
                        raw_data_path,
                        output_path,
                        generate_plots:bool = True,
                        report_qc:bool = True) -> None:

    bulk = Spectra().select_parameters(spectra_bulk_df, dataset_type="bulk")  

    DEP_LOGGER.info(f"Starting Bulk parameters qualification")
    
    DEP_LOGGER.info(f"Converting to pandas dataframe")
    bulk_df = bulk.to_pandas()

    DEP_LOGGER.info(f"Loading QC (config_id = {site_buoys_to_process.qc_config})")
    qc = WaveBuoyQC(config_id=site_buoys_to_process.qc_config)
            
    DEP_LOGGER.info(f"Creating global qc column")
    bulk_df = qc.create_global_qc_columns(data=bulk_df)
    
    qc.load_data(data=bulk_df)
    DEP_LOGGER.info(f"Extracting parameters to QC")
    parameters_to_qc = qc.get_parameters_to_qc(data=bulk_df, qc_config=qc.qc_config)           
    
    DEP_LOGGER.info(f"Performing qualification")
    bulk_qualified = qc.qualify(data=bulk_df,
                                parameters=parameters_to_qc,
                                parameter_type="waves",
                                gross_range_test=True,
                                rate_of_change_test=True,
                                flat_line_test=True,
                                mean_std_test=True,
                                spike_test=False)
   
    DEP_LOGGER.info(f"Saving QC results as csvs")
    bulk_df.to_csv(os.path.join(output_path, "bulk_qc_subflags.csv"))
    bulk_qualified.to_csv(os.path.join(output_path, "bulk_qc.csv"))
    DEP_LOGGER.info("Bulk parameters qualification successfull")
    
    if isinstance(temp, pl.DataFrame) and not temp.is_empty():
        DEP_LOGGER.info(f"Starting temperature qualification")

        DEP_LOGGER.info(f"Converting to pandas dataframe")
        temp_df = temp.to_pandas()

                     
        DEP_LOGGER.info(f"Creating global qc column")
        temp_df = qc.create_global_qc_columns(data=temp_df)
        
        qc.load_data(data=temp_df)
        DEP_LOGGER.info(f"Extracting parameters to QC")
        parameters_to_qc_temp = qc.get_parameters_to_qc(data=temp_df, qc_config=qc.qc_config)
        
        DEP_LOGGER.info(f"Performing qualification")
        temp_qualified = qc.qualify(data=temp_df,
                                    parameters=parameters_to_qc_temp,
                                    parameter_type="temp",
                                    gross_range_test=True,
                                    rate_of_change_test=True,
                                    flat_line_test=True,
                                    mean_std_test=True,
                                    spike_test=True)
        
        DEP_LOGGER.info(f"Saving QC results as csvs")
        temp_qualified.to_csv(os.path.join(output_path, "temp_qc.csv"))
        temp_df.to_csv(os.path.join(output_path, "temp_qc_subflags.csv"))
        
        DEP_LOGGER.info("Temp qualification successfull")

    else:
        DEP_LOGGER.info(f"No temp data")
        temp_qualified = None
        temp_df = None
        

    DEP_LOGGER.info(f"Composing dataset")
    bulk_ds = ncBulk().compose_dataset(waves=bulk_qualified, temp=temp_qualified)
    
    DEP_LOGGER.info(f"Converting data types")
    bulk_ds = Process.convert_dtypes(dataset=bulk_ds, parameters_type="bulk")
    DEP_LOGGER.info("variables dtypes converted and now conforming to template")

    DEP_LOGGER.info(f"Loading attributes")
    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="bulk")
    
    DEP_LOGGER.info(f"Assigning general attributes")
    bulk_ds = nc_atts_comp.assign_general_attributes(dataset=bulk_ds, site_name=site_name)
    
    DEP_LOGGER.info(f"Creating timeseries variable")
    bulk_ds = ncBulk().create_timeseries_variable(dataset=bulk_ds)

    DEP_LOGGER.info(f"Extracting time periods")
    start_end_dates = ncWriter().extract_start_end_dates_list(dataset_objects=(bulk_ds,))

    DEP_LOGGER.info(f"Converting time to CF conventions")
    bulk_ds = ncBulk().convert_time_to_CF_convention(bulk_ds)

    DEP_LOGGER.info(f"Assigning variables attributes")
    bulk_ds = nc_atts_comp.assign_variables_attributes(dataset=bulk_ds)

    DEP_LOGGER.info(f"Composing file names")
    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name,
                                periods=start_end_dates,
                                deployment_metadata=deployment_metadata,
                                parameters_type="bulk")
    
    DEP_LOGGER.info(f"Saving datasets as {nc_file_names}")
    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=(bulk_ds,),
                            parameters_type="bulk"
                        )

    DEP_LOGGER.info(f"WAVE-PARAMETERS netCDFs successfully generated")

    if generate_plots:
        
        p = Plots(site_name=site_name,
                  deployment_folder=site_buoys_to_process.loc["datapath"],
                  output_path=output_path)
        # p.map_positions(data=bulk_df, map_coverage=(40,20,5,1,.3), figsize=(10,10))
        DEP_LOGGER.info(f"Generating subflags plots for each variable")
        p.qc_subflags_each_variable(dataset=bulk_ds,
                                    waves_subflags=bulk_df, 
                                    temp_subflags=temp_df,
                                            )

    if report_qc:
        DEP_LOGGER.info(f"Generating QC report for wave parameters")
        qc.report_qc([bulk_df, temp_df], parameters_to_qc, output_path)

def generate_raw_displacements_NC_files(disp,
                                        gps,
                                        site_name,
                                        deployment_metadata,
                                        buoys_metadata,
                                        regional_metadata,
                                        site_buoys_to_process,
                                        deploy_start,
                                        deploy_end,
                                        raw_data_path,
                                        output_path
                                        ) -> None:

    # cp = csvProcess()
    # disp = cp.filter_deployment_dates(dataframe=disp, deploy_start=deploy_start, deploy_end=deploy_end)

    # filter watchcircle
    DEP_LOGGER.info(f"Composing dataset")
    disp_ds = ncDisp().compose_dataset(data=disp, data_gps=gps)

    DEP_LOGGER.info(f"Loading attributes")
    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="displacements") 

    DEP_LOGGER.info(f"Assigning general attributes")
    disp_ds = nc_atts_comp.assign_general_attributes(dataset=disp_ds, site_name=site_name)
    
    DEP_LOGGER.info(f"Creating timeseries variable")
    disp_ds = ncSpectra().create_timeseries_variable(dataset=disp_ds)
    # # DEP_LOGGER.info("time series variable created")

    DEP_LOGGER.info(f"Extracting fortnightly periods")
    fortnight_periods = ncDisp().extract_fortnightly_periods_dataset(dataset=disp_ds)

    DEP_LOGGER.info(f"Splitting global dataset into fortnightly datasets")
    disp_ds_objects = ncDisp().split_dataset_fortnightly(dataset=disp_ds, periods=fortnight_periods)
    
    DEP_LOGGER.info(f"Converting time to CF conventions")
    disp_ds_objects = ncSpectra().convert_time_to_CF_convention_ds_list(disp_ds_objects)
   
    DEP_LOGGER.info(f"Assigning variables attributes")
    disp_ds_objects = nc_atts_comp.assign_variables_attributes_dataset_objects(dataset_objects=disp_ds_objects)

    DEP_LOGGER.info(f"Composing file names")
    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name,
                                periods=fortnight_periods,
                                deployment_metadata=deployment_metadata,
                                parameters_type="displacements")
    
    DEP_LOGGER.info(f"Saving datasets as {nc_file_names}")
    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=disp_ds_objects,
                            parameters_type='displacements'
                        )

if __name__ == "__main__":
    
    load_dotenv()
    vargs = args_aodn_processing()

    num_workers = max(cpu_count()//2, 1)
    client = Client(n_workers=num_workers)
  
    imos_logging = IMOSLogging()
    
    GENERAL_LOGGER = IMOSLogging().logging_start(logger_name=f"{datetime.now().strftime("%Y%m%dT%H%M%S")}_{vargs.region}_general_logger.log",
                                                logging_filepath=os.getenv("GENERAL_LOGGER_PATH"))

    buoys_to_process = WaveBuoy().load_buoys_to_process()

    for idx, site in buoys_to_process.iterrows():
        start_exec_time = time.time()
        try:

            dm_deployment_path, output_path = process_paths(site)


            DEP_LOGGER = imos_logging.logging_start(logging_filepath=output_path,
                                                    logger_name="DM_processing.log")
            
            GENERAL_LOGGER.info("="*10 + f" {os.path.basename(dm_deployment_path)} " + "="*50)
            GENERAL_LOGGER.info(f"Deployment file path: {dm_deployment_path}")
            
            DEP_LOGGER.info(f"DM Processing started - {site.name} ".upper())
            DEP_LOGGER.info(f"Deployment file path: {dm_deployment_path}")

            DEP_LOGGER.info(f"Metadata loading ".upper() + "="*50)
            metadata = load_metadata(site, dm_deployment_path)
            
            metadata_args = MetadataArgs(
                                site_name=metadata['site_name'],
                                deployment_metadata=metadata['deployment_metadata'],
                                buoys_metadata=metadata['buoys_metadata'],
                                regional_metadata=metadata['regional_metadata'],
                                site_buoys_to_process=site,
                                deploy_start=metadata['deploy_start'],
                                deploy_end=metadata['deploy_end'],
                                raw_data_path=metadata['raw_data_path'],
                                output_path=output_path
                            )

            DEP_LOGGER.info(f"SD card data processing ".upper() + "="*50)
            results = process_from_SD(metadata['raw_data_path'])
            
            results = filter_dates(results, 
                                    metadata_args.site_buoys_to_process.utc_offset, 
                                    metadata_args.deploy_start,
                                    metadata_args.deploy_end,
                                    metadata_args.site_buoys_to_process.time_crop_start,
                                    metadata_args.site_buoys_to_process.time_crop_end)
            
            DEP_LOGGER.info(f"Spectra Calculation ".upper() + "="*50)
            spectra_bulk_df = calculate_spectra_from_displacements(results['displacements'], vargs.enable_dask)

            DEP_LOGGER.info(f"Spectra results processing ".upper() + "="*50)
            spectra_bulk_df = align_gps(spectra_bulk_df, results['gps'])

            DEP_LOGGER.info(f"Spectra results watch circle qualification ".upper() + "="*50)
            spectra_bulk_df, metadata['deployment_metadata'] = qc_watch_circle(spectra_bulk_df, site, output_path, metadata['deployment_metadata'])

            DEP_LOGGER.info(f"WAVE-SPECTRA AODN compliant file generation step ".upper() + "="*50)
            generate_spectra_NC_file(spectra_bulk_df, results['gps'], **vars(metadata_args))

            DEP_LOGGER.info(f"WAVE-PARAMETERS AODN compliant file generation step ".upper() + "="*50)
            generate_bulk_NC_file(spectra_bulk_df, results['gps'], results['surface_temp'], **vars(metadata_args))

            DEP_LOGGER.info(f"RAW-DISPLACEMENTS AODN compliant file generation step ".upper() + "="*50)
            generate_raw_displacements_NC_files(results['displacements'], results['gps'], **vars(metadata_args))

            GENERAL_LOGGER.info(f"Processsing finished in {round((time.time() - start_exec_time)/60, 2)} min")
            DEP_LOGGER.info(f"Processsing finished in {round((time.time() - start_exec_time)/60, 2)} min")
            imos_logging.logging_stop(logger=DEP_LOGGER)

        except Exception as e:
            DEP_LOGGER.error(str(e), exc_info=True)
            GENERAL_LOGGER.error(str(e), exc_info=True)
            
            DEP_LOGGER_file_path = imos_logging.get_log_file_path(logger=DEP_LOGGER)
            DEP_LOGGER_file_path = DEP_LOGGER.handlers[0].baseFilename
            imos_logging.logging_stop(logger=DEP_LOGGER)
            error_DEP_LOGGER_file_path = imos_logging.rename_log_file_if_error(site_name=site.loc['name'],
                                                                        file_path=DEP_LOGGER_file_path,
                                                                        script_name=os.path.basename(__file__).removesuffix(".py"),
                                                                        add_runtime=False)
            continue
        
