import os

import time
from xarray import Dataset
import polars as pl

from wavebuoy_dm.spectra import Spectra
from wavebuoy_dm.processing.concat import csvConcat
from wavebuoy_dm.processing.process import csvProcess
from wavebuoy_dm.netcdf.process import ncSpectra, ncDisp, ncBulk
from wavebuoy_dm.utils import IMOSLogging, args_processing_dm


def process_from_SD(raw_data_path, suffixes_to_concat=["FLT", "LOC"]) -> list[pl.DataFrame]:

    LOGGER.info(f"Lazy concatenating csv files for {suffixes_to_concat}")
    cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=suffixes_to_concat) 
    lazy_concat_results = cc.lazy_concat_files()

    cp = csvProcess()
    LOGGER.info("Lazy processing cocatenated csv files")
    lazy_processed_results = cp.process_concat_results(lazy_concat_results)

    LOGGER.info("Collecting processed csv files")
    collected_results = cp.collect_results(lazy_processed_results)

    return collected_results

def filter_dates(disp, gps, utc_offset, deploy_start, deploy_end) -> list[pl.DataFrame]:


    cp = csvProcess()
    LOGGER.info(f"Filtering displacements data with passed deployment datetimes: {deploy_start} - {deploy_end}")
    disp = cp.filter_deployment_dates(dataframe=disp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                      time_crop_start=0, time_crop_end=0)
    
    LOGGER.info(f"Filtering gps data with passed deployment datetimes: {deploy_start} - {deploy_end}")
    gps = cp.filter_deployment_dates(dataframe=gps, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                      time_crop_start=0, time_crop_end=0)

    return disp, gps

def calculate_spectra_from_displacements(disp: pl.DataFrame, enable_dask:bool):
    
    s = Spectra()
    
    LOGGER.info(f"Setting spectra calculation parameters")
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
        LOGGER.info(f"Processing dask chunks for spectra calculation")
        disp_chunks_30m = s.generate_time_chunks(disp, time_chunk='30m')
        disp_chunks_30m = s.filter_insufficient_samples(disp_chunks_30m, min_samples)
        
        LOGGER.info(f"Splitting dask chunks by the number of workers (= {num_workers})")
        disp_dask_chunks = s.generate_dask_chunks(disp_chunks_30m, num_workers)

        LOGGER.info(f"Creating dask tasks")
        disp_tasks = [dask.delayed(s.process_dask_chunk)
                (dask_chunk, nfft, nover, fs, merge, 'xyz', info) 
                for dask_chunk in disp_dask_chunks]

        LOGGER.info(f"Computing dask tasks")
        spectra_bulk_results = dask.compute(*disp_tasks)

        LOGGER.info(f"Concatenating dask computed results")
        spectra_bulk_df = pl.concat(spectra_bulk_results)

        return spectra_bulk_df
    
    else:
        LOGGER.info(f"Calculating spectra with whole displacements data")
        return s.spectra_from_dataframe(disp, nfft, nover, fs, merge, 'xyz', min_samples, info)

def align_gps(spectra_bulk_df, gps) -> pl.DataFrame:

    cp = csvProcess()

    LOGGER.info(f"Interpolating gps to match spectra_bulk data")
    spectra_bulk_df = cp.interpolate_lat_lon(spectra_bulk_df, gps)

    return spectra_bulk_df

def split_spectra_bulk(spectra_bulk) -> list[pl.DataFrame]:
    
    LOGGER.info(f"Splitting spectra_bulk into spectra and bulk individual polars dataframes")
    return (
        spectra_bulk[['TIME', 'LONGITUDE', 'LATITUDE', 'FREQUENCY', 'A1', 'B1', 'A2', 'B2', 'ENERGY']],
        spectra_bulk[['TIME', 'LONGITUDE', 'LATITUDE', 'WSSH', 'WPFM', 'WPPE', 'SSWMD', 'WPDI', 'WMDS', 'WPDS']],
                )

def convert_to_dataset(disp, gps, spectra, bulk) -> list[Dataset]:
    
    LOGGER.info(f"Converting raw displacements dataframe to dataset")
    disp = ncDisp().compose_dataset(disp, gps)        

    LOGGER.info(f"Converting spectra dataframe to dataset")
    spectra = ncSpectra().compose_dataset(spectra)
    
    LOGGER.info(f"Converting bulk dataframe to dataset")
    bulk = ncBulk().compose_dataset(bulk.to_pandas())

    return disp, spectra, bulk

def save_outputs(disp, gps, spectra, bulk, output_path):

    outputs = {
        "raw_displacements": disp,
        "gps": gps,
        "spectra": spectra,
        "bulk": bulk
    }

    LOGGER.info(f"Iterating over datasets")
    for name, data in outputs.items():

        if isinstance(data, pl.DataFrame):
            filepath = os.path.join(output_path, f"{name}.csv")
            LOGGER.info(f"Saving {name} data as {filepath}")
            if name == "spectra":
                # Assuming flattening is needed
                data.to_pandas().to_csv(filepath, index=False)
            else:
                data.write_csv(filepath)

        elif isinstance(data, Dataset):
            filepath = os.path.join(output_path, f"{name}.nc")
            LOGGER.info(f"Saving {name} data as {filepath}")
            data.to_netcdf(filepath, engine="netcdf4")

        else:
            LOGGER.warning(f"Unsupported data type for {name}: {type(data)}")


if __name__ == "__main__":
    
    start_exec_time = time.time()
    
    vargs = args_processing_dm()

    if vargs.enable_dask == True:
        from multiprocessing import cpu_count
        from dask.distributed import Client
        import dask

        num_workers = max(cpu_count()//2, 1)
        client = Client(n_workers=num_workers)
    
    imos_logging = IMOSLogging()
    LOGGER = imos_logging.logging_start(logging_filepath=vargs.output_path, logger_name="DM_spotter_processing.log")

    try:
    
        LOGGER.info("DM Processing started ".upper())
        LOGGER.info(f"SD card data processing ".upper() + "="*50)
        results = process_from_SD(vargs.log_path)
        
        disp, gps = filter_dates(disp=results['displacements'],
                                        gps=results['gps'],
                                        deploy_start=vargs.deploy_dates_start,
                                        deploy_end=vargs.deploy_dates_end,
                                        utc_offset=8)
        
        LOGGER.info(f"Spectra Calculation ".upper() + "="*50)
        spectra_bulk = calculate_spectra_from_displacements(disp, vargs.enable_dask)

        LOGGER.info(f"Spectra results processing ".upper() + "="*50)
        spectra_bulk = align_gps(spectra_bulk, gps)
        
        # import pickle
        # with open(os.path.join(vargs.output_path, "spectra_bulk.pkl"), "wb") as f:
        #     pickle.dump(spectra_bulk, f)
        # with open(os.path.join(vargs.output_path, "spectra_bulk.pkl"), "rb") as f:
        #     spectra_bulk = pickle.load(f)

        spectra, bulk = split_spectra_bulk(spectra_bulk)
        if vargs.output_type == 'netcdf':
            disp, spectra, bulk = convert_to_dataset(disp, gps, spectra, bulk)

        LOGGER.info(f"Ouputs saving as {vargs.output_type} ".upper() + "="*50)
        save_outputs(disp, gps, spectra, bulk, vargs.output_path)

        LOGGER.info("DM Processing finished ".upper() + "="*50)

    except Exception as e:
        LOGGER.error(str(e), exc_info=True)  
    
    LOGGER.info(f"EXECUTION TIME: {round(time.time() - start_exec_time, 2)} s")
    imos_logging.logging_stop(logger=LOGGER)
    
