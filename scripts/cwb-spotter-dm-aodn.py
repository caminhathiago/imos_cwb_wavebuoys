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

from wavebuoy_dm.spectra import Spectra
from wavebuoy_dm.processing.concat import csvConcat
from wavebuoy_dm.processing.process import csvProcess
from wavebuoy_dm.netcdf.process import ncSpectra, ncDisp, ncBulk
from wavebuoy_dm.netcdf.writer import ncWriter, ncAttrsComposer
from wavebuoy_dm.wavebuoy import WaveBuoy
from wavebuoy_dm.utils import IMOSLogging, args_processing
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
    dm_deployment_path = os.path.dirname(dm_deployment_path
                                        .replace("Y:", "\\\\drive.irds.uwa.edu.au\\OGS-COD-001")
                                        .replace("X:", "\\\\drive.irds.uwa.edu.au\\OGS-COD-001")
                                        )
    output_path = os.path.join(dm_deployment_path, "processed_py")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    return dm_deployment_path, output_path

def load_metadata(site_buoys_to_process:pd.DataFrame, dm_deployment_path) -> list[pd.DataFrame]:

    if os.path.basename(dm_deployment_path) != "log":                         
        raw_data_path = os.path.join(dm_deployment_path, 'log')
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"path {raw_data_path} does not exist. Check if deployment path exists, and if SD card data is in the folder log.")

    deploy_start, deploy_end, spot_id = WaveBuoy().extract_deploy_dates_spotid_from_path(site_buoys_to_process.datapath)

    site_name, region = WaveBuoy().extract_region_site_name(path=dm_deployment_path)
    buoys_metadata = WaveBuoy()._get_buoys_metadata(buoy_type='sofar', buoys_metadata_file_name="buoys_metadata.csv")
    deployment_metadata = WaveBuoy().load_latest_deployment_metadata(site_name=site_name, region=region)
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

def process_from_SD(raw_data_path) -> list[pl.DataFrame]:

    cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=["FLT","LOC","SST","BARO"]) #,"SENS_IND","SENS_AGG"])#, suffixes_to_concat=["HDR","SST","LOC","BARO","SPC"])
    lazy_concat_results = cc.lazy_concat_files()

    cp = csvProcess()
    lazy_processed_results = cp.process_concat_results(lazy_concat_results)

    collected_results = cp.collect_results_threadpool4(lazy_processed_results)
    

    # disp = cp.filter_absurd_datetimes(data=collected_results["displacements"])
    # gps = cp.filter_absurd_datetimes(data=collected_results["gps"])

    # temp = None
    # if "SST" in cc.suffixes_to_concat and "surface_temp" in collected_results.keys():
    #     temp = cp.filter_absurd_datetimes(data=collected_results["surface_temp"])
   

    return collected_results["displacements"], collected_results["gps"], collected_results["surface_temp"]

def filter_dates(disp, gps, temp, utc_offset, deploy_start, deploy_end, time_crop_start, time_crop_end) -> list[pl.DataFrame]:


    cp = csvProcess()
    disp = cp.filter_deployment_dates(dataframe=disp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                      time_crop_start=time_crop_start, time_crop_end=time_crop_end)
    gps = cp.filter_deployment_dates(dataframe=gps, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                      time_crop_start=time_crop_start, time_crop_end=time_crop_end)
    temp = cp.filter_deployment_dates(dataframe=temp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                       time_crop_start=time_crop_start, time_crop_end=time_crop_end)

    return disp, gps, temp

def align_gps(spectra_bulk_df, gps) -> pl.DataFrame:

    cp = csvProcess()
    spectra_bulk_df = cp.interpolate_lat_lon(spectra_bulk_df, gps)

    return spectra_bulk_df

def qc_watch_circle(spectra_bulk_df, site_buoys_to_process:pd.DataFrame, output_path:str):
    
    s = site_buoys_to_process
    mainline = s.mainline_length + (s.mainline_length*s.mooring_stretch_factor)
    catenary = s.catenary_length + (s.catenary_length*s.mooring_stretch_factor)
    watch_circle = np.sqrt(mainline**2 - s.DeployDepth**2) + catenary + s.watch_circle_gps_error

    cp = csvProcess()

    spectra_bulk_df, out_of_radious_pct = cp.qc_watch_circle(
        spectra_bulk_df,
        deploy_lat=s.DeployLat,
        deploy_lon=s.DeployLon,
        watch_circle=watch_circle,
        watch_circle_fail=s.watch_circle_fail
    )

    if out_of_radious_pct >= s.out_of_radius_tolerance:
        mainline = mainline + s.mainline_length_error
        catenary = catenary + s.catenary_length_error

        mainline *= (1 + s.mooring_stretch_factor)
        catenary *= (1 + s.mooring_stretch_factor)

        watch_circle = np.sqrt(mainline**2 - s.DeployDepth**2) + catenary + s.watch_circle_gps_error
        
        spectra_bulk_df, out_of_radious_pct = cp.qc_watch_circle(
                                                    spectra_bulk_df,
                                                    deploy_lat=s.DeployLat,
                                                    deploy_lon=s.DeployLon,
                                                    watch_circle=watch_circle,
                                                    watch_circle_fail=s.watch_circle_fail
                                                )

        spectra_bulk_df_file_name = os.path.join(output_path, "spectra_bulk_df_qc_watch.csv")
        (spectra_bulk_df[['TIME', 'LATITUDE', 'LONGITUDE', 'distance', 'WATCH_quality_control_primary', 'WATCH_quality_control_secondary']]
         .to_pandas()
         .to_csv(spectra_bulk_df_file_name)
            )
        
        if out_of_radious_pct >= s.out_of_radius_tolerance:
            
            raise ValueError(f"{out_of_radious_pct}% out of radius (watchcircle = {round(watch_circle,2)}), greater then tolerance ({s.out_of_radius_tolerance}%). CSV with qc_flag_watch saved as {spectra_bulk_df_file_name}")

    return spectra_bulk_df

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
    SITE_LOGGER.info(f"watch circle crop = {round(percentage_cropped*100,2)}% of data")
    gps = gps_filtered.rename({'LATITUDE':'latitude', 'LONGITUDE':'longitude'})
    
    disp = cp.filter_deployment_dates(dataframe=disp, 
                                      deploy_start=gps['datetime'][0], 
                                      deploy_end=gps['datetime'][-1], hours_crop=3)

    return disp, gps

def calculate_spectra_from_displacements(disp: pl.DataFrame):
    # SPECTRA ---------------------------------------------------------------
    
    
    
    info = {
        "hab": None,
        "fmaxSS": 1/8,
        "fmaxSea": 1/2,
        "bad_data_thresh": 2/3,
        "hs0_thresh": 3,
        "t0_thresh": 5
    }
    fs = 2.5
    min_samples = Spectra().calculate_min_samples(fs=fs, spec_window=30)
    nfft = Spectra().calculate_nfft(fs=fs, spec_window=30)
    merge = Spectra().define_merging(nfft=nfft)
    nover = 0.5

    disp_chunks_30m = Spectra().generate_time_chunks(disp, time_chunk='30m')
    disp_chunks_30m = Spectra().filter_insufficient_samples(disp_chunks_30m, min_samples)
    
    disp_dask_chunks = Spectra().generate_dask_chunks(disp_chunks_30m, num_workers)

    # Calculate Spectra -------------------------
    disp_tasks = [dask.delayed(Spectra().process_dask_chunk)
             (dask_chunk, nfft, nover, fs, merge, 'xyz', info) 
             for dask_chunk in disp_dask_chunks]

    spectra_bulk_results = dask.compute(*disp_tasks)
    print('concatenating...')
    spectra_bulk_df = pl.concat(spectra_bulk_results)

    return spectra_bulk_df

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

    spectra_ds = ncSpectra().compose_dataset(spectra)
    
    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="spectral")
    
    spectra_ds = nc_atts_comp.assign_general_attributes(dataset=spectra_ds, site_name=site_name)
    SITE_LOGGER.info("general attributes assigned to embedded dataset")
    
    spectra_ds = ncSpectra().create_timeseries_variable(dataset=spectra_ds)
    SITE_LOGGER.info("time series variable created")

    start_end_dates = ncWriter().extract_start_end_dates_list(dataset_objects=(spectra_ds,))

    spectra_ds = ncSpectra().convert_time_to_CF_convention(spectra_ds)
     # SITE_LOGGER.info("dataset objects time dimension processed to conform to CF conventions")

    spectra_ds = nc_atts_comp.assign_variables_attributes(dataset=spectra_ds)
    # SITE_LOGGER.info("variables attributes assigned to datasets")

    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name.upper(),
                                periods=start_end_dates,
                                deployment_metadata=deployment_metadata,
                                parameters_type="spectral")
    
    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=(spectra_ds,)
                        )

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
                        output_path) -> None:

    # BULK ------------------------------------------------------------

    # Select Bulk ------------------------
    bulk = Spectra().select_parameters(spectra_bulk_df, dataset_type="bulk")  


    # Process Bulk ------------------------
    # cp = csvProcess()
    # deploy_lat, deploy_lon = site_buoys_to_process.loc['DeployLat'], site_buoys_to_process.loc['DeployLon']
    # watch_circle = site_buoys_to_process.loc['watch_circle']

    # bulk = cp.interpolate_lat_lon(dataframe=bulk, locations_dataframe=gps)
    # bulk = cp.filter_watch_circle_geodesic(dataframe=bulk, deploy_lat=deploy_lat, deploy_lon=deploy_lon, max_distance=watch_circle)
    # bulk = cp.filter_deployment_dates(dataframe=bulk, deploy_start=deploy_start, deploy_end=deploy_end)

    if temp is not None:
        temp = temp.rename({"temperature":"TEMP", "datetime":"TIME_TEMP"})
    #     temp = cp.filter_deployment_dates(dataframe=temp, deploy_start=deploy_start, deploy_end=deploy_end)

    # Qualify Bulk ------------------------------------------
    bulk_df = bulk.to_pandas()
    
    qc = WaveBuoyQC(config_id=1)
            
    # Waves ----
    bulk_df = qc.create_global_qc_columns(data=bulk_df)
    
    qc.load_data(data=bulk_df)
    parameters_to_qc = qc.get_parameters_to_qc(data=bulk_df, qc_config=qc.qc_config)           
    bulk_qualified = qc.qualify(data=bulk_df,
                                parameters=parameters_to_qc,
                                parameter_type="waves",
                                gross_range_test=True,
                                rate_of_change_test=True,
                                flat_line_test=False,
                                mean_std_test=True,
                                spike_test=True)
   
    bulk_df.to_csv(os.path.join(output_path, "bulk_qc_subflags.csv"))
    bulk_qualified.to_csv(os.path.join(output_path, "bulk_qc.csv"))
    SITE_LOGGER.info("waves qualification successfull")
    
    # Temp ----
    # # TEMPRARY SETUP
    # temp = None
    # # TEMPRARY SETUP
    
    if temp is not None:
        temp_df = temp.to_pandas()
        
        qc = WaveBuoyQC(config_id=1)
                
        # Waves ----
        temp_df = qc.create_global_qc_columns(data=temp_df)
        
        qc.load_data(data=bulk_df)
        parameters_to_qc = qc.get_parameters_to_qc(data=temp_df, qc_config=qc.qc_config)
        temp_qualified = qc.qualify(data=temp_df,
                                    parameters=parameters_to_qc,
                                    parameter_type="temp",
                                    gross_range_test=True,
                                    rate_of_change_test=True,
                                    flat_line_test=True,
                                    mean_std_test=True,
                                    spike_test=True)
        
        temp_qualified.to_csv(os.path.join(output_path, "temp_qc.csv"))
        temp_df.to_csv(os.path.join(output_path, "temp_qc_subflags.csv"))
        SITE_LOGGER.info("temp qualification successfull")

    else:
        temp_qualified = None
        

    # Generate Bulk NC file -------------------------------------------------------
    bulk_ds = ncBulk().compose_dataset(waves=bulk_qualified, temp=temp_qualified)
    
    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="bulk")
    
    bulk_ds = nc_atts_comp.assign_general_attributes(dataset=bulk_ds, site_name=site_name)
    SITE_LOGGER.info("general attributes assigned to embedded dataset")
    
    bulk_ds = ncBulk().create_timeseries_variable(dataset=bulk_ds)
    SITE_LOGGER.info("time series variable created")

    start_end_dates = ncWriter().extract_start_end_dates_list(dataset_objects=(bulk_ds,))

    bulk_ds = ncBulk().convert_time_to_CF_convention(bulk_ds)
     # SITE_LOGGER.info("dataset objects time dimension processed to conform to CF conventions")

    bulk_ds = nc_atts_comp.assign_variables_attributes(dataset=bulk_ds)
    # SITE_LOGGER.info("variables attributes assigned to datasets")


    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name.upper(),
                                periods=start_end_dates,
                                deployment_metadata=deployment_metadata,
                                parameters_type="bulk")
    
    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=(bulk_ds,),
                            parameters_type="bulk"
                        )

    # RAW DISPLACEMENTS -------------------------------------------------------------
    # Generate Raw Displacements NC Files -------------------------------------------------

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

    disp_ds = ncDisp().compose_dataset(data=disp, data_gps=gps)

    nc_atts_comp = ncAttrsComposer(buoys_metadata=buoys_metadata,
                                    deployment_metadata=deployment_metadata,
                                   regional_metadata=regional_metadata,
                                   parameters_type="displacements") 

    disp_ds = nc_atts_comp.assign_general_attributes(dataset=disp_ds, site_name=site_name)
    SITE_LOGGER.info("general attributes assigned to embedded dataset")
    
    disp_ds = ncSpectra().create_timeseries_variable(dataset=disp_ds)
    # # SITE_LOGGER.info("time series variable created")

    fortnight_periods = ncDisp().extract_fortnightly_periods_dataset(dataset=disp_ds)
    disp_ds_objects = ncDisp().split_dataset_fortnightly(dataset=disp_ds, periods=fortnight_periods)
    
    disp_ds_objects = ncSpectra().convert_time_to_CF_convention_ds_list(disp_ds_objects)
   
    disp_ds_objects = nc_atts_comp.assign_variables_attributes_dataset_objects(dataset_objects=disp_ds_objects)

    nc_file_names = ncWriter().compose_file_names(
                                site_id=site_name.upper(),
                                periods=fortnight_periods,
                                deployment_metadata=deployment_metadata,
                                parameters_type="displacements")
    

    ncWriter().save_nc_file(output_path=output_path,
                            file_names=nc_file_names,
                            dataset_objects=disp_ds_objects,
                            parameters_type='displacements'
                        )


if __name__ == "__main__":
    
    vargs = args_processing()

    num_workers = max(cpu_count()//2, 1)
    client = Client(n_workers=num_workers)
  
    imos_logging = IMOSLogging()
    
    buoys_to_process = WaveBuoy().load_buoys_to_process(vargs.region)

    for idx, site in buoys_to_process.iterrows():
        start_exec_time = time.time()
        try:
            # METADATA ------------------
                
            dm_deployment_path, output_path = process_paths(site)

            # # TEMPORARY SETUP
            # output_path = r"\\drive.irds.uwa.edu.au\OGS-COD-001\CUTTLER_wawaves\Data\vicwaves\CapeBridgewater\delayedmode\cape-bridgewater_deploy20240618_retrieve20241027_SPOT-31670C\processed_py_QCtests"
            # # TEMPORARY SETUP

            SITE_LOGGER = imos_logging.logging_start(logging_filepath=output_path,
                                                    logger_name="site_logger")
            
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

            # PRE-PROCESSING AND CALCULATIONS -----------------     
            # disp, gps, temp = process_from_SD(metadata['raw_data_path'])

            import pickle
            # with open("pickle_files/gps.pkl", "wb") as f:
            #     pickle.dump(gps, f)
            # with open("pickle_files/temp.pkl", "wb") as f:
            #     pickle.dump(temp, f)
            # with open("pickle_files/disp.pkl", "wb") as f:
            #     pickle.dump(disp, f)
            

            with open("pickle_files/disp.pkl", "rb") as f:
                disp = pickle.load(f)
            with open("pickle_files/gps.pkl", "rb") as f:
                gps = pickle.load(f)
            with open("pickle_files/temp.pkl", "rb") as f:
                temp = pickle.load(f)
            

            disp, gps, temp = filter_dates(disp, gps, temp, 
                                           metadata_args.site_buoys_to_process.utc_offset, 
                                           metadata_args.deploy_start,
                                           metadata_args.deploy_end,
                                           metadata_args.site_buoys_to_process.time_crop_start,
                                           metadata_args.site_buoys_to_process.time_crop_end)
            
            spectra_bulk_df = calculate_spectra_from_displacements(disp)

            

            spectra_bulk_df = align_gps(spectra_bulk_df, gps)
            
            with open("pickle_files/spectra_bulk_df.pkl", "wb") as f:
                pickle.dump(spectra_bulk_df, f)
            # with open("pickle_files/spectra_bulk_df.pkl", "rb") as f:
            #     spectra_bulk_df = pickle.load(f)

            spectra_bulk_df = qc_watch_circle(spectra_bulk_df, site, output_path)

            
            
            
            # from datetime import datetime, timedelta
            # start = datetime(2024,7,1,2,30)
            # end = start + timedelta(hours=72*4)
            # temp = temp.filter(
            #     (pl.col("datetime") >= start) & (pl.col("datetime") <= end)
            # )
            # NC FILES GENERATION ---------------------
            # generate_spectra_NC_file(spectra_bulk_df, gps, **vars(metadata_args))
            # SITE_LOGGER.info("spectra NC generated")

            generate_bulk_NC_file(spectra_bulk_df, gps, temp, **vars(metadata_args))
            SITE_LOGGER.info("integral parameters NC generated")

            # generate_raw_displacements_NC_files(disp, gps, **vars(metadata_args))
            # SITE_LOGGER.info("raw displacements NCs generated")

            SITE_LOGGER.info(f"Processsing finished for {metadata['site_name']} in {(time.time() - start_exec_time)/60} min")
            imos_logging.logging_stop(logger=SITE_LOGGER)

        except Exception as e:
            # error_message = IMOSLogging().unexpected_error_message.format(site_name=metadata['site_name'].upper())
            SITE_LOGGER.error(str(e), exc_info=True)
            
            # Closing current site logging
            site_logger_file_path = imos_logging.get_log_file_path(logger=SITE_LOGGER)
            site_logger_file_path = SITE_LOGGER.handlers[0].baseFilename
            imos_logging.logging_stop(logger=SITE_LOGGER)
            error_logger_file_path = imos_logging.rename_log_file_if_error(site_name=site.loc['name'],
                                                                        file_path=site_logger_file_path,
                                                                        script_name=os.path.basename(__file__).removesuffix(".py"),
                                                                        add_runtime=False)
            continue
        
        print(f"EXEC TIME: {time.time() - start_exec_time} s")
