import os
import time

from xarray import Dataset
import polars as pl
from multiprocessing import cpu_count
from dask.distributed import Client
import dask

from wavebuoy_dm.spectra import Spectra
from wavebuoy_dm.processing.concat import csvConcat
from wavebuoy_dm.processing.process import csvProcess
from wavebuoy_dm.netcdf.process import ncSpectra, ncDisp, ncBulk
from wavebuoy_dm.utils import IMOSLogging, args_processing_dm


class DMSpotterProcessor:
    def __init__(self, config=None):
        self.vargs = self.load_config(config)
        self.client = None
        self.imos_logging = IMOSLogging()
        self.LOGGER = self.imos_logging.logging_start(
            logging_filepath=self.vargs.output_path,
            logger_name="DM_spotter_processing.log"
        )

    def process_config(self, config:dict):
        import argparse
        vargs = argparse.ArgumentParser()

        if not os.path.exists(config['log_path']):
            raise ValueError('{path} not a valid path'.format(path=config.log_path))
        
        else:
            vargs.log_path = config['log_path']
            
            output_path = os.path.join(config['log_path'], "processed")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            vargs.output_path = output_path

        if config.get('deploy_dates'):
            from datetime import datetime
            vargs.deploy_dates_start = datetime.strptime(vargs.deploy_dates[0],"%Y%m%dT%H%M%S")
            vargs.deploy_dates_end = datetime.strptime(vargs.deploy_dates[1],"%Y%m%dT%H%M%S")
        else:
            vargs.deploy_dates = None

        if config.get('enable_dask'):
            vargs.enable_dask = config['enable_dask']

        output_type = config.get('output_type')
        if output_type:
            if output_type in ("netcdf, csv"):
                vargs.output_type = config['output_type']
            else:
                raise ValueError(f'{output_type} not a valid path')
        else:
            vargs.output_type = 'netcdf'

        return vargs

    def load_config(self, config: dict = None):
        
        if config is None:
            return args_processing_dm()
        else:
            import argparse
            vargs = argparse.ArgumentParser()

            if not os.path.exists(config['log_path']):
                raise ValueError('{path} not a valid path'.format(path=config.log_path))
            
            else:
                vargs.log_path = config['log_path']
                
                output_path = os.path.join(config['log_path'], "processed")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                vargs.output_path = output_path

            if config.get('deploy_dates'):
                from datetime import datetime
                vargs.deploy_dates_start = datetime.strptime(config['deploy_dates'][0],"%Y-%m-%dT%H:%M:%S")
                vargs.deploy_dates_end = datetime.strptime(config['deploy_dates'][1],"%Y-%m-%dT%H:%M:%S")
                vargs.deploy_dates = config['deploy_dates']
            else:
                vargs.deploy_dates = None

            if config.get('utc_offset'):
                vargs.utc_offset = config['utc_offset']
            else:
                vargs.utc_offset = 0 # UTC

            if config.get('enable_dask'):
                vargs.enable_dask = config['enable_dask']

            output_type = config.get('output_type')
            if output_type:
                if output_type in ("netcdf, csv"):
                    vargs.output_type = config['output_type']
                else:
                    raise ValueError(f'{output_type} not a valid path')
            else:
                vargs.output_type = 'netcdf'

            return vargs      

    def setup_dask(self):
        if self.vargs.enable_dask:
            self.num_workers = max(cpu_count() // 2, 1)
            self.client = Client(n_workers=self.num_workers)
            self.LOGGER.info(f"Dask client started with {self.num_workers} workers")

    def close_dask(self):
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            self.LOGGER.info("Dask client closed")

    def process_from_SD(self, raw_data_path, suffixes_to_concat=["FLT", "LOC"]) -> list[pl.DataFrame]:

        self.LOGGER.info(f"Lazy concatenating csv files for {suffixes_to_concat}")
        cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=suffixes_to_concat) 
        lazy_concat_results = cc.lazy_concat_files()

        cp = csvProcess()
        self.LOGGER.info("Lazy processing cocatenated csv files")
        lazy_processed_results = cp.process_concat_results(lazy_concat_results)

        self.LOGGER.info("Collecting processed csv files")
        collected_results = cp.collect_results(lazy_processed_results)

        return collected_results

    def filter_dates(self, disp, gps, utc_offset, deploy_start, deploy_end) -> list[pl.DataFrame]:


        cp = csvProcess()
        self.LOGGER.info(f"Filtering displacements data with passed deployment datetimes: {deploy_start} - {deploy_end}")
        disp = cp.filter_deployment_dates(dataframe=disp, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                        time_crop_start=0, time_crop_end=0)
        
        self.LOGGER.info(f"Filtering gps data with passed deployment datetimes: {deploy_start} - {deploy_end}")
        gps = cp.filter_deployment_dates(dataframe=gps, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                        time_crop_start=0, time_crop_end=0)

        return disp, gps

    def calculate_spectra_from_displacements(self, disp: pl.DataFrame, enable_dask:bool):
        
        s = Spectra()
        
        self.LOGGER.info(f"Setting spectra calculation parameters")
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
            self.LOGGER.info(f"Processing dask chunks for spectra calculation")
            disp_chunks_30m = s.generate_time_chunks(disp, time_chunk='30m')
            disp_chunks_30m = s.filter_insufficient_samples(disp_chunks_30m, min_samples)
            
            self.LOGGER.info(f"Splitting dask chunks by the number of workers (= {self.num_workers})")
            disp_dask_chunks = s.generate_dask_chunks(disp_chunks_30m, self.num_workers)

            self.LOGGER.info(f"Creating dask tasks")
            disp_tasks = [dask.delayed(s.process_dask_chunk)
                    (dask_chunk, nfft, nover, fs, merge, 'xyz', info) 
                    for dask_chunk in disp_dask_chunks]

            self.LOGGER.info(f"Computing dask tasks")
            spectra_bulk_results = dask.compute(*disp_tasks)

            self.LOGGER.info(f"Concatenating dask computed results")
            spectra_bulk_df = pl.concat(spectra_bulk_results)

            return spectra_bulk_df
        
        else:
            self.LOGGER.info(f"Calculating spectra with whole displacements data")
            return s.spectra_from_dataframe(disp, nfft, nover, fs, merge, 'xyz', min_samples, info)

    def align_gps(self, spectra_bulk_df, gps) -> pl.DataFrame:

        cp = csvProcess()

        self.LOGGER.info(f"Interpolating gps to match spectra_bulk data")
        spectra_bulk_df = cp.interpolate_lat_lon(spectra_bulk_df, gps)

        return spectra_bulk_df

    def split_spectra_bulk(self, spectra_bulk) -> list[pl.DataFrame]:
        
        self.LOGGER.info(f"Splitting spectra_bulk into spectra and bulk individual polars dataframes")
        return (
            spectra_bulk[['TIME', 'LONGITUDE', 'LATITUDE', 'FREQUENCY', 'A1', 'B1', 'A2', 'B2', 'ENERGY']],
            spectra_bulk[['TIME', 'LONGITUDE', 'LATITUDE', 'WSSH', 'WPFM', 'WPPE', 'SSWMD', 'WPDI', 'WMDS', 'WPDS']],
                    )

    def convert_to_dataset(self, disp, gps, spectra, bulk) -> list[Dataset]:
        
        self.LOGGER.info(f"Converting raw displacements dataframe to dataset")
        disp = ncDisp().compose_dataset(disp, gps)        

        self.LOGGER.info(f"Converting spectra dataframe to dataset")
        spectra = ncSpectra().compose_dataset(spectra)
        
        self.LOGGER.info(f"Converting bulk dataframe to dataset")
        bulk = ncBulk().compose_dataset(bulk.to_pandas())

        return disp, spectra, bulk

    def save_outputs(self, disp, gps, spectra, bulk, output_path):

        outputs = {
            "raw_displacements": disp,
            "gps": gps,
            "spectra": spectra,
            "bulk": bulk
        }

        self.LOGGER.info(f"Iterating over datasets")
        for name, data in outputs.items():

            if isinstance(data, pl.DataFrame):
                filepath = os.path.join(output_path, f"{name}.csv")
                self.LOGGER.info(f"Saving {name} data as {filepath}")
                if name == "spectra":
                    data.to_pandas().to_csv(filepath, index=False)
                else:
                    data.write_csv(filepath)

            elif isinstance(data, Dataset):
                filepath = os.path.join(output_path, f"{name}.nc")
                self.LOGGER.info(f"Saving {name} data as {filepath}")
                data.to_netcdf(filepath, engine="netcdf4")

            else:
                self.LOGGER.warning(f"Unsupported data type for {name}: {type(data)}")

    def run(self, save_outputs=True):
        start_exec_time = time.time()
        
        try:
            self.LOGGER.info("DM Processing started ".upper())
            
            self.setup_dask()
            
            self.LOGGER.info(f"SD card data processing ".upper() + "="*50)
            
            results = self.process_from_SD(self.vargs.log_path)

            if self.vargs.deploy_dates:
                disp, gps = self.filter_dates(
                    disp=results['displacements'],
                    gps=results['gps'],
                    deploy_start=self.vargs.deploy_dates_start,
                    deploy_end=self.vargs.deploy_dates_end,
                    utc_offset=self.vargs.utc_offset
                )
            else:
                disp, gps = results['displacements'], results['gps']

            self.LOGGER.info(f"Spectra Calculation ".upper() + "="*50)
            spectra_bulk = self.calculate_spectra_from_displacements(disp, self.vargs.enable_dask)

            self.LOGGER.info(f"Spectra results processing ".upper() + "="*50)
            spectra_bulk = self.align_gps(spectra_bulk, gps)

            spectra, bulk = self.split_spectra_bulk(spectra_bulk)
            if self.vargs.output_type == 'netcdf':
                disp, spectra, bulk = self.convert_to_dataset(disp, gps, spectra, bulk)

            self.disp, self.spectra, self.bulk, self.gps = disp, spectra, bulk, gps

            if save_outputs:
                self.LOGGER.info(f"Outputs saving as {self.vargs.output_type} ".upper() + "="*50)
                self.save_outputs(disp, gps, spectra, bulk, self.vargs.output_path)

            self.LOGGER.info("DM Processing finished ".upper() + "="*50)

            self.close_dask()

        except Exception as e:
            self.LOGGER.error(str(e), exc_info=True)
            self.close_dask()
        
        finally:
            exec_time = round(time.time() - start_exec_time, 2)
            self.LOGGER.info(f"EXECUTION TIME: {exec_time} s")
            self.imos_logging.logging_stop(logger=self.LOGGER)


if __name__ == "__main__":

    from wavebuoy_dm.dm_processor import DMSpotterProcessor
    
    

    vargs = args_processing_dm()
    processor = DMSpotterProcessor()
    processor.run()

