import os
import time
from datetime import datetime

from xarray import Dataset
import polars as pl
import multiprocessing
from multiprocessing import cpu_count
from dask.distributed import Client
import dask

from wavebuoy_dm.spectra import Spectra
from wavebuoy_dm.processing.concat import csvConcat
from wavebuoy_dm.processing.process import csvProcess
from wavebuoy_dm.netcdf.process import ncSpectra, ncDisp, ncBulk
from wavebuoy_dm.utils import IMOSLogging, args_processing_dm, Plots


class DMSpotterProcessor:
    def __init__(self, config=None, client=None, spectra_info=None, to_process=["displacements", "gps"]):
    
        # Templates
        self.output_types = ("netcdf", "csv")
        self.spectra_variables = ['TIME', 'LONGITUDE', 'LATITUDE', 'FREQUENCY', 'A1', 'B1', 'A2', 'B2', 'ENERGY']
        self.bulk_variables = ['TIME', 'LONGITUDE', 'LATITUDE', 'WSSH', 'WPFM', 'WPPE', 'SSWMD', 'WPDI', 'WMDS', 'WPDS']

        # Files to process
        self.map_variablestype_suffix = {
            "displacements": "FLT",
            "gps": "LOC",
            "surface_temp": "SST",
            "atmospheric_pressure": "BARO",
            "smart_mooring": "XXX"
        }
        self.suffixes_to_process = self.to_process(to_process)

        # Config
        self.vargs = self.load_config(config)
        
        # Dask client
        self.client = client if client else None

        # Logger
        self.imos_logging = IMOSLogging()
        self.LOGGER = self.imos_logging.logging_start(
            logging_filepath=self.vargs.output_path,
            logger_name="DM_spotter_processing.log"
        )

        # Spectra calculation config
        if spectra_info:
            self.spectra_info
        else:
            self.spectra_info = {
                "hab": None,
                "fmaxSS": 1/8,
                "fmaxSea": 1/2,
                "bad_data_thresh": 2/3,
                "hs0_thresh": 3,
                "t0_thresh": 5,
                "fs": 2.5,
                "spec_window": 30,
                "nover": 0.5,
            }
        
        self.calculate_spectra_info(self.spectra_info)
            
    def to_process(self, to_process):
        
        suffixes = []
        suffixes_unmatched = []
        for variables_type in to_process:
            if variables_type in self.map_variablestype_suffix.keys():
                suffixes.append(self.map_variablestype_suffix[variables_type])
            else:
                suffixes_unmatched.append(variables_type)
        
        if suffixes_unmatched:
            raise ValueError(f"""The following suffixes are not available {suffixes_unmatched}. Please check for typos. List of available variables types: {tuple(self.map_variablestype_suffix.keys())}""")

        return [self.map_variablestype_suffix[variables_type] for variables_type in to_process]

    def load_config(self, config: dict = None):
        
        if config is None:
            return args_processing_dm()
        else:
            import argparse
            vargs = argparse.ArgumentParser()

           # Paths
            log_path = config.get("log_path", os.getcwd())
            if not os.path.exists(log_path):
                raise ValueError(f"{log_path} is not a valid path")

            vargs.log_path = log_path

            output_path = os.path.join(log_path, "processed")
            os.makedirs(output_path, exist_ok=True)

            vargs.output_path = output_path

            # Deployment dates and UTC offset
            deploy_dates = config.get('deploy_dates', None)
            if deploy_dates:
                vargs.deploy_dates_start = datetime.strptime(deploy_dates[0],"%Y-%m-%dT%H:%M:%S")
                vargs.deploy_dates_end = datetime.strptime(deploy_dates[1],"%Y-%m-%dT%H:%M:%S")
            
            vargs.deploy_dates = deploy_dates

            if config.get('utc_offset', 0): # UTC if nothing is provided
                vargs.utc_offset = config['utc_offset']

            # Dask enabling
            enable_dask = config.get("enable_dask", False)
            if isinstance(enable_dask, str):
                enable_dask = enable_dask.lower() in ("true")
            vargs.enable_dask = enable_dask

            # Final output types
            output_type = config.get("output_type", "netcdf")
            if output_type not in self.output_types:
                raise ValueError(f"{output_type} is not a valid output type")

            vargs.output_type = output_type

            return vargs      

    def setup_dask(self):
        if self.vargs.enable_dask and self.client is None:
            if multiprocessing.current_process().name == "MainProcess":
                self.num_workers = max(cpu_count() // 2, 1)
                self.client = Client(n_workers=self.num_workers)
                self.LOGGER.info(f"Dask client started with {self.num_workers} workers")

    def close_dask(self):
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            self.LOGGER.info("Dask client closed")

    def process_from_SD(self, raw_data_path) -> list[pl.DataFrame]:

        self.LOGGER.info(f"Lazy concatenating csv files for {self.suffixes_to_process}")
        cc = csvConcat(files_path=raw_data_path, suffixes_to_concat=self.suffixes_to_process) 
        lazy_concat_results = cc.lazy_concat_files()

        cp = csvProcess()
        self.LOGGER.info("Lazy processing cocatenated csv files")
        lazy_processed_results = cp.process_concat_results(lazy_concat_results)

        self.LOGGER.info("Collecting processed csv files")
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

    def filter_dates(self, results, utc_offset, deploy_start, deploy_end) -> list[pl.DataFrame]:

        cp = csvProcess()

        results_filtered = []
        for key, result in results.items():
            self.LOGGER.info(f"Filtering displacements data with passed deployment datetimes: {deploy_start} - {deploy_end}")
            result = cp.filter_deployment_dates(dataframe=result, utc_offset=utc_offset, deploy_start=deploy_start, deploy_end=deploy_end,
                                            time_crop_start=0, time_crop_end=0)
            results_filtered.append(result)

        return results_filtered

    def calculate_spectra_info(self, spectra_info) -> dict:
        
        s = Spectra()
        
        min_samples = s.calculate_min_samples(fs=spectra_info["fs"], spec_window=spectra_info["spec_window"])
        nfft = s.calculate_nfft(fs=spectra_info["fs"], spec_window=spectra_info["spec_window"])
        merge = s.define_merging(nfft=nfft)

        self.spectra_info.update({
            "min_samples": min_samples,
            "nfft": nfft,
            "merge": merge
        })

    def check_dataset_empty(self, results:dict, variables_type:str) -> None:
        
        results_keys = list(results.keys())

        if variables_type not in results_keys:
            raise ValueError(
                f"'{variables_type}' key is missing in results. Please check input files or processing configurations."
            )

        result = results[variables_type]
        check = (isinstance(result, pl.DataFrame) and result.is_empty()) or (isinstance(result, list) and len(result) == 0)

        if check:
            raise ValueError(
                f"'{variables_type}' dataset is empty. Please check if files are corrupted or processing configurations are correct."
            )
        
    def calculate_spectra_from_displacements(self, results:dict, enable_dask:bool):
        
        self.check_dataset_empty(results, 'displacements')
        disp = results['displacements']

        s = Spectra()
        
        self.LOGGER.info(f"Setting spectra calculation parameters")
        fs = self.spectra_info['fs']
        min_samples = self.spectra_info['min_samples']
        nfft = self.spectra_info['nfft']
        merge = self.spectra_info['merge']
        nover = self.spectra_info['nover']

        if enable_dask:
            self.LOGGER.info(f"Processing dask chunks for spectra calculation")
            disp_chunks_30m = s.generate_time_chunks(disp, time_chunk='30m')
            disp_chunks_30m = s.filter_insufficient_samples(disp_chunks_30m, min_samples)
            
            self.LOGGER.info(f"Splitting dask chunks by the number of workers (= {self.num_workers})")
            disp_dask_chunks = s.generate_dask_chunks(disp_chunks_30m, self.num_workers)

            self.LOGGER.info(f"Creating dask tasks")
            disp_tasks = [dask.delayed(s.process_dask_chunk)
                    (dask_chunk, nfft, nover, fs, merge, 'xyz', self.spectra_info) 
                    for dask_chunk in disp_dask_chunks]

            self.LOGGER.info(f"Computing dask tasks")
            spectra_bulk_results = dask.compute(*disp_tasks)

            self.LOGGER.info(f"Concatenating dask computed results")
            spectra_bulk_df = pl.concat(spectra_bulk_results)

            return spectra_bulk_df
        
        else:
            self.LOGGER.info(f"Calculating spectra with whole displacements data")
            return s.spectra_from_dataframe(disp, nfft, nover, fs, merge, 'xyz', min_samples, self.spectra_info)

    def align_gps(self, spectra_bulk_df, results:dict) -> pl.DataFrame:

        self.check_dataset_empty(results, 'gps')
        gps = results['gps']

        cp = csvProcess()

        self.LOGGER.info(f"Interpolating gps to match spectra_bulk data")
        spectra_bulk_df = cp.interpolate_lat_lon(spectra_bulk_df, gps)

        return spectra_bulk_df

    def split_spectra_bulk(self, spectra_bulk:pl.DataFrame, results:dict) -> list[pl.DataFrame]:
        
        self.LOGGER.info(f"Splitting spectra_bulk into spectra and bulk individual polars dataframes")
        results['bulk'] = spectra_bulk[self.bulk_variables]
        results['spectra'] = spectra_bulk[self.spectra_variables]
        
        return results

    def convert_to_dataset(self, results:dict) -> list:
        
        # Convert displacements to Dataset if GPS is available
        if 'displacements' in results and 'gps' in results:
            self.LOGGER.info("Converting displacements dataframe to dataset")
            results['displacements'] = ncDisp().compose_dataset(results['displacements'], results['gps'])

        # Convert spectra
        if 'spectra' in results:
            self.LOGGER.info("Converting spectra dataframe to dataset")
            results['spectra'] = ncSpectra().compose_dataset(results['spectra'])

        # Convert bulk
        if "bulk" in results:
            self.LOGGER.info("Converting bulk dataframe to dataset")
            
            args = {"waves":results["bulk"].to_pandas()}
            
            temperature_data = results.get("surface_temp")
            if isinstance(temperature_data, pl.DataFrame) and not temperature_data.is_empty():
                args["temp"] = temperature_data
                # No temperature data → call without it
            results["bulk"] = ncBulk().compose_dataset(**args)

        return results

    def save_outputs(self, results, output_path):

        self.LOGGER.info(f"Iterating over datasets")
        for name, data in results.items():

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

            elif isinstance(data, list):
                self.LOGGER.warning(f"Not saving '{name}' as it is an empty list. This spotter may not have this sensor or it must not be working.")

    def _set_dataset_attributes(self, results: dict):
        for key in results:
            if not isinstance(results[key], list):
                setattr(self, key, results[key])

    def run(self, save_outputs=True, map_plots=True):
        
        start_exec_time = time.time() 
        try:
            self.LOGGER.info("DM Processing started ".upper())
            
            self.setup_dask()
            
            self.LOGGER.info(f"SD card data processing ".upper() + "="*50)
            
            results = self.process_from_SD(self.vargs.log_path)

            if self.vargs.deploy_dates:
                results = self.filter_dates(
                    results=results,
                    deploy_start=self.vargs.deploy_dates_start,
                    deploy_end=self.vargs.deploy_dates_end,
                    utc_offset=self.vargs.utc_offset
                )
    

            self.LOGGER.info(f"Spectra Calculation ".upper() + "="*50)
            spectra_bulk = self.calculate_spectra_from_displacements(results, self.vargs.enable_dask)

            self.LOGGER.info(f"Spectra results processing ".upper() + "="*50)
            spectra_bulk = self.align_gps(spectra_bulk, results)

            results = self.split_spectra_bulk(spectra_bulk, results)
            if self.vargs.output_type == 'netcdf':
                results = self.convert_to_dataset(results)

            self._set_dataset_attributes(results)

            if save_outputs:
                self.LOGGER.info(f"Outputs saving as {self.vargs.output_type} ".upper() + "="*50)
                self.save_outputs(results, self.vargs.output_path)

            self.LOGGER.info("DM Processing finished ".upper() + "="*50)

            self.close_dask()
                

        except Exception as e:
            self.LOGGER.error(str(e), exc_info=True)
            self.close_dask()
        
        finally:
            exec_time = round(time.time() - start_exec_time, 2)
            self.LOGGER.info(f"EXECUTION TIME: {exec_time} s")
            self.imos_logging.logging_stop(logger=self.LOGGER)


def main():

    vargs = args_processing_dm()
    processor = DMSpotterProcessor()
    processor.run()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 
    main()    
