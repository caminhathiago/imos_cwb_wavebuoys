import os
import logging
from datetime import datetime, timedelta
import re
import argparse
import sys
import importlib.resources as resources

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

def args_aodn_processing():
 
    parser = argparse.ArgumentParser(description='Creates NetCDF files.\n '
                                     'Prints out the path of the new locally generated NetCDF file.')
    
    parser.add_argument('-o', '--output-path', dest='output_path', type=str, default=None,
                        help="output directory of netcdf file",
                        required=False)
    
    parser.add_argument('-l', '--log-path', dest='log_path', type=str, default=None,
                        help="directory where SD card files are stored",
                        required=False)

    parser.add_argument('-pp', '--deploy-dates', dest='deploy_dates', type=str, default=None, nargs=2,
                        help="deployment dates period to be processed. Please pass start and end dates as YYYYmmdd YYYYmmdd, separated by a blank space.",
                        required=False)

    parser.add_argument('-r', '--region', dest='region', type=str, default=None,
                        help="directory where SD card files are stored",
                        required=True)
    
    parser.add_argument('-ed', '--enable-dask', dest='enable_dask', action='store_true',
                    help="Whether to enable spectra calculation with dask threading")

    vargs = parser.parse_args()
    
    # if not os.path.exists(vargs.output_path):
    #     try:
    #         os.makedirs(vargs.output_path)
    #     except Exception:
    #         raise ValueError('{path} not a valid path'.format(path=vargs.output_path))
    #         sys.exit(1)

    # if not os.path.exists(vargs.log_path):
    #     raise ValueError('{path} not a valid path'.format(path=vargs.log_path))

    if vargs.deploy_dates:
        vargs.deploy_dates_start = datetime.strptime(vargs.deploy_dates[0],"%Y%m%d")
        vargs.deploy_dates_end = datetime.strptime(vargs.deploy_dates[1],"%Y%m%d")


    return vargs



def args_processing_dm():

    parser = argparse.ArgumentParser()
 
    parser.add_argument('-l', '--log-path', dest='log_path', type=str, default=None,
                        help="Directory where SD card files are stored",
                        required=True)
    
    parser.add_argument('-d', '--deploy-dates', dest='deploy_dates', type=str, default=None, nargs=2,
                        help="Deployment datetimes period to be processed. Please pass start and end dates as YYYYmmddTHHMMSS YYYYmmdd, separated by a blank space",
                        required=False)

    parser.add_argument('-u', '--utc-offset', dest='utc_offset', type=float, default=.0,
                        help="Desired offset from utc in hours. Defaults to 0 (UTC)",
                        required=False)

    parser.add_argument('-ed', '--enable-dask', dest='enable_dask', action='store_true',
                    help="Whether to enable spectra calculation with dask threading")

    parser.add_argument('-ot', '--output-type', dest='output_type', type=str, default="netcdf",
                    help="Whether to save outputs as csv or netcdf, when applicable.")

    vargs = parser.parse_args()

    if not os.path.exists(vargs.log_path):
        raise ValueError('{path} not a valid path'.format(path=vargs.log_path))
    else:
        vargs.output_path = os.path.join(vargs.log_path, "processed")
        if not os.path.exists(vargs.output_path):
            os.makedirs(vargs.output_path)

    if vargs.deploy_dates:
        vargs.deploy_dates_start = datetime.strptime(vargs.deploy_dates[0],"%Y%m%dT%H%M%S")
        vargs.deploy_dates_end = datetime.strptime(vargs.deploy_dates[1],"%Y%m%dT%H%M%S")
    
    return vargs


def args_processing_dm_test(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path")
    parser.add_argument("--output_path")
    parser.add_argument("--enable_dask", type=bool)
    return parser.parse_args(argv)

class IMOSLogging:

    unexpected_error_message = "An unexpected error occurred when processing {site_name}\n Please check the site log for details"

    def __init__(self):
        pass

    def logging_start(self, logging_filepath, logger_name="general_logger", level=logging.INFO):
        """
        Start logging using the Python logging library.
        Parameters:
            logger_name (str): Name of the logger to create or retrieve.
            level (int): Logging level (default: logging.INFO).
        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        self.logging_filepath = os.path.join(logging_filepath, "logs", logger_name)

        if not os.path.exists(os.path.dirname(self.logging_filepath)):
            os.makedirs(os.path.dirname(self.logging_filepath))

        self.logger = logging.getLogger(logger_name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(level)
        handler = logging.FileHandler(self.logging_filepath, mode="w")
        handler.setLevel(level)
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.addHandler(stream_handler)

        return self.logger

    def logging_stop(self, logger):
        """Close logging handlers for the current logger."""
        handlers = list(logger.handlers)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def get_log_file_path(self, logger):
        return logger.handlers[0].baseFilename
    
    def rename_log_file_if_error(self, site_name: str, file_path, script_name: str, add_runtime: bool = True):
        site_name = site_name.upper()
        runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        pattern = f"{site_name}_{script_name}"
        new_name = "ERROR_" + f"{site_name}_{script_name}"
        if add_runtime:
            new_name += f"_{runtime}"

        new_file_name = re.sub(pattern, new_name, file_path)
        if os.path.exists(new_file_name):
            os.replace(file_path, new_file_name)
        else:
            os.rename(file_path, new_file_name)

        return os.path.join(file_path, new_file_name)

    def rename_push_log_if_error(self, file_path: str, add_runtime: bool = True):
        runtime = datetime.now().strftime("%Y%m%dT%H%M%S")
        pattern = "aodn_ftp_push"
        new_name = f"ERROR_aodn_ftp_push"
        if add_runtime:
            new_name += f"_{runtime}"

        new_file_name = re.sub(pattern, new_name, file_path)
        if os.path.exists(new_file_name):
            os.replace(file_path, new_file_name)
        else:
            os.rename(file_path, new_file_name)

        return os.path.join(file_path, new_file_name)
    

class DebuggingHelpers:

    @staticmethod
    def pickle_files(data:tuple, file_name:str, output_path:str = "test/pickle_files", mode:str = "wb"):
        import pickle
        with open(os.path.join(output_path,f"{file_name}.pkl"), mode) as f:
            if mode == "wb":
                pickle.dump(data, f)
            elif mode == "rb":
                return pickle.load(f)
            else:
                raise ValueError(f"{mode} not accepted as mode. Options: ['wb', 'rb'] ")


class Plots:

    def __init__(self, output_path:str, deployment_folder:str, site_name:str):
        
        if not os.path.exists(output_path):
            raise NotADirectoryError(f"{output_path} is not a valid directory")
        else:
            self.output_path = os.path.join(output_path, "plots")
            os.makedirs(self.output_path, exist_ok=True)

        self.site_name = site_name
        self.deployment_folder_path = deployment_folder
        self.deployment_folder = os.path.basename(deployment_folder)

    def convert_CF_time_to_datetime(self, dataset) -> xr.Dataset:

        time_label = [lab for lab in dataset.variables if "TIME" in lab][0]

        datetimes = pd.to_datetime(dataset[time_label].values, origin='1950-01-01', unit='D')

        dataset[time_label] = datetimes
        
        return dataset

    def qc_subflags_each_variable(self,
                                dataset,
                                waves_subflags,
                                temp_subflags,
                                figsize=(15, 3),
                                variable=None):

        vars = ['LATITUDE', 'LONGITUDE', 'WSSH', 'WPFM', 'WPPE', 'SSWMD', 
                'WPDI', 'WMDS', 'WPDS']
        subflags_vars = waves_subflags.filter(regex="_test").columns.tolist()

        waves_subflags = waves_subflags.set_index("TIME")

        if "TEMP" in dataset.variables:
            vars.append('TEMP')
            temp_subflags = temp_subflags.set_index("TIME_TEMP")
            temp_subflag_vars = temp_subflags.filter(regex="_test").columns.tolist()
            subflags_vars.extend(temp_subflag_vars)


        n_vars = len(vars)
        if n_vars == 0:
            print("No variables to plot.")
            return
        
        if variable:
            vars = [variable]

        for var in vars:
            if var not in dataset.variables:
                continue
            
            data_subflags = waves_subflags
            if var == "TEMP":
                primary_flags_column = "TEMP_quality_control"
                data_subflags = temp_subflags
            else:
                primary_flags_column = "WAVE_quality_control"

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3,1, figsize=(figsize[0], figsize[1]*3), sharex=True)

            data_subflags[var].plot(marker='o', ax=ax[0], ms=2, label='python')
            data_subflags[primary_flags_column].plot(marker='o', ax=ax[2], ms=2)
            data_subflags[var].plot(marker='o', ax=ax[1], ms=2, label='python', alpha=0.4)

            if var not in ("LATITUDE", "LONGITUDE") and not var.endswith("_quality_control"):
                
                if "TEMP" in var:
                    subflags_data = temp_subflags
                    subflags_preffix = "TEMP_QC_"
                else:
                    subflags_data = waves_subflags
                    subflags_preffix = "WAVE_QC_"

                var_subflags = subflags_data.filter(regex=var).columns.tolist()
                var_subflags.remove(var)
                var_subflags = [var_subflag for var_subflag in var_subflags if not var_subflag.endswith("_quality_control")]
                # var_subflags = [var_subflag for var_subflag in subflags_vars if var in var_subflag]
                
                for subflag in var_subflags:
                
                    subflag_names_codes = {
                        f"{subflags_preffix}{var}_spike_test": 10,
                        f"{subflags_preffix}{var}_mean_std_test": 15,
                        f"{subflags_preffix}{var}_flat_line_test": 16,
                        f"{subflags_preffix}{var}_gross_range_test": 19,
                        f"{subflags_preffix}{var}_rate_of_change_test": 20,

                    }

                    base_flag = subflag.replace(f"{subflags_preffix}{var}_", "")
                    flag_code = subflag_names_codes.get(f"{subflags_preffix}{var}_{base_flag}", None)

                    plot_subflags_kwargs = dict(marker='o', linestyle="None", ms=4,
                                ax=ax[1], label=f"{subflag.replace(f"{subflags_preffix}{var}","")}")
                    
                    flags_2 = subflags_data.loc[subflags_data[subflag] == 2, var]
                    flags_3 = subflags_data.loc[subflags_data[subflag] == 3, var]
                    flags_4 = subflags_data.loc[subflags_data[subflag] == 4, var]

                    for flags, color in zip([flags_2, flags_3, flags_4], ["green", "orange", "red"]):
                        if not flags.empty:
                            flags.plot(**plot_subflags_kwargs, color=color)

                            for time, val in flags.items():
                                ax[1].annotate(
                                    str(flag_code),
                                    xy=(time, val),
                                    xytext=(5, 5),
                                    textcoords="offset points",
                                    fontsize=8,
                                    ha='left',
                                    va='bottom'
                                )
        
            ax[0].set_ylabel(var)
            ax[1].set_ylabel(var)
            ax[2].set_ylabel(primary_flags_column)


            ax[0].legend()
            ax[0].grid(True)
            ax[1].grid(True)
            ax[2].grid(True)

            from matplotlib.dates import DayLocator, DateFormatter
            ax[2].xaxis.set_major_locator(DayLocator(interval=10))
            
            ax[2].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            for label in ax[2].get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")

            import matplotlib.patches as mpatches
            import matplotlib.lines as mlines
            custom_legend = [
                mpatches.Patch(color='white', label='10: spike '),
                mpatches.Patch(color='white', label='15: mean std'),
                mpatches.Patch(color='white', label='16: flat line'),
                mpatches.Patch(color='white', label='19: max/min range'),
                mpatches.Patch(color='white', label='20: rate of change'),

                mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='not assessed'),
                mlines.Line2D([], [], color='orange', marker='o', linestyle='None', label='suspect'),
                mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='fail')
            ]

            ax[1].legend(
                handles=ax[1].get_legend_handles_labels()[0] + custom_legend,
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                frameon=True
            )

            plt.tight_layout()
            output_file_name = f"{self.site_name}_{var}_subflags.png"
            plt.savefig(os.path.join(self.output_path, output_file_name), dpi=300)

    
    def map_positions(self,
                    data:pd.DataFrame,
                    map_coverage:tuple=(10.,20.),
                    figsize:tuple=(8, 8)):

        lats = data["LATITUDE"].values 
        lons = data["LONGITUDE"].values 

        lat_center = np.mean(lats)
        lon_center = np.mean(lons)

        for i, coverage in enumerate(map_coverage):
            
            fig, ax = plt.subplots(
                subplot_kw={'projection': ccrs.PlateCarree()},
                figsize=figsize
            )

            km_to_deg_lat = coverage / 111.0
            km_to_deg_lon = coverage / 111.0 * np.cos(np.radians(lat_center))

            lat_min = lat_center - km_to_deg_lat
            lat_max = lat_center + km_to_deg_lat
            lon_min = lon_center - km_to_deg_lon
            lon_max = lon_center + km_to_deg_lon           

            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True, linestyle="--")

            ax.scatter(lons, lats, color='red', s=8, transform=ccrs.PlateCarree(), label="Position (Lat/Lon)")
            ax.scatter(lon_center, lat_center, color='blue', s=8, marker="x", transform=ccrs.PlateCarree(), label="Mean Lat/Lon")

            shapename = 'admin_0_countries'
            populated_places = shpreader.natural_earth(resolution='110m',
                                                    category='cultural',
                                                    name='populated_places')

            reader = shpreader.Reader(populated_places)
            for city in reader.records():
                lon, lat = city.geometry.x, city.geometry.y
                if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                    ax.plot(lon, lat, marker='o', color='black', markersize=4,
                            transform=ccrs.PlateCarree())
                    ax.text(lon, lat, city.attributes['NAME'],
                            fontsize=8, transform=ccrs.PlateCarree())


            ax.legend()
            ax.set_title(f"Map {coverage} km around average location")
            plt.tight_layout()
            output_file_name = f"{self.site_name}_positions-{coverage}km.png"
            plt.savefig(os.path.join(self.output_path, output_file_name), dpi=300)


    def map_positions_shapefiles(self,
                                data: pd.DataFrame,
                                deployment_center:tuple,
                                watch_circle:float,
                                map_coverage: tuple = (10., 20.),
                                figsize: tuple = (8, 8)):
        
        with resources.path("wavebuoy_dm.maps", "AUS_2021_AUST_GDA2020.shp") as shp_path:
            aus = gpd.read_file(shp_path)

        watch_circle_cols = [col for col in data.columns if "WATCH_quality_control" in col]

        lats = data["LATITUDE"].values
        lons = data["LONGITUDE"].values

        lat_center = deployment_center[0]
        lon_center = deployment_center[1]

        lat_mean = np.mean(lats)
        lon_mean = np.mean(lons)

        import matplotlib.gridspec as gridspec

        for coverage in map_coverage:

            fig = plt.figure(figsize=figsize)
            outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

            # Left: one big panel
            ax_left = fig.add_subplot(outer[0, 0])

            # Right: split vertically into 2
            right_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer[0, 1], height_ratios=[1, 1, 1],
            )
            ax_top_right = fig.add_subplot(right_gs[0, 0])
            ax_middle_right = fig.add_subplot(right_gs[1, 0])
            ax_bottom_right = fig.add_subplot(right_gs[2, 0])

            # Map
            km_to_deg_lat = coverage / 111.0
            km_to_deg_lon = coverage * 1.5 / 111.0 * np.cos(np.radians(lat_center))

            lat_min = lat_center - km_to_deg_lat
            lat_max = lat_center + km_to_deg_lat
            lon_min = lon_center - km_to_deg_lon
            lon_max = lon_center + km_to_deg_lon

            aus.plot(ax=ax_left, color="lightgray", edgecolor="black")

            # Map positions
            ax_left.scatter(lons, lats, color="blue", s=8, label="Positions")
            
            if watch_circle_cols:
                
                for watch_circle_col in watch_circle_cols:
                    level = watch_circle_col.split("_")[-1]
                    
                    mask = data[watch_circle_col] == 3
                    ax_left.scatter(data.loc[mask, "LONGITUDE"], data.loc[mask, "LATITUDE"], color="goldenrod", ls="None", s=8, label=f"Suspect {level} watch circle")
                    
                    mask = data[watch_circle_col] == 4
                    ax_left.scatter(data.loc[mask, "LONGITUDE"], data.loc[mask, "LATITUDE"], color="red", ls="None", s=8, label=f"Fail {level} watch circle")

            ax_left.scatter(lon_mean, lat_mean, color="green", s=40, marker="x", label=f"Mean Position ({round(lat_mean,5)}, {round(lon_mean,5)})")
            ax_left.scatter(lon_center, lat_center, color="grey", s=40, marker="^", label="Watch circle center")

            # Watch1 circle plots
            deg_lat = watch_circle/ (1000*111) 
            deg_lon = watch_circle / (1000*(111 * np.cos(np.radians(lat_center))))

            theta = np.linspace(0, 2*np.pi, 100)
            lat_circle = deployment_center[0] + deg_lat * np.sin(theta)
            lon_circle = deployment_center[1] + deg_lon * np.cos(theta)
            
            from shapely.geometry import Point
            circle_geom = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lon_circle, lat_circle)]).unary_union.convex_hull

            circle_gdf = gpd.GeoDataFrame(geometry=[circle_geom])
            circle_gdf.plot(ax=ax_left, edgecolor='grey', linestyle='--',facecolor='none', linewidth=2, alpha=0.6)
           
            watch_circle_label = f"Watch circle ({watch_circle:.1f} m)"
            from matplotlib.lines import Line2D
            circle_handle = Line2D(
                [0], [0],
                color="grey",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=watch_circle_label,
            )

            # LatLon timeseries
            ax_top_right.plot(data["TIME"], data['LATITUDE'], marker='.')
            ax_middle_right.plot(data["TIME"], data['LONGITUDE'], marker='.')
            ax_bottom_right.plot(data["TIME"], data['distance'], marker='.')
            ax_bottom_right.plot(data["TIME"], np.repeat(watch_circle, len(data["TIME"])), ls="dashed", lw=1, color="grey", label=f"Watch circle = {round(watch_circle,1)} m")
            
            watch_circle_cols = [col for col in data.columns if "WATCH_quality_control" in col]
            for col in watch_circle_cols:
                # if "primary" in col:
                #     mask = data[col] == 3
                #     color = "goldenrod"
                #     ax_top_right.plot(data.loc[mask,"TIME"], data.loc[mask,'LATITUDE'], marker='.', color=color, ls="None")
                #     ax_middle_right.plot(data.loc[mask,"TIME"], data.loc[mask,'LONGITUDE'], marker='.', color=color, ls="None")
                #     ax_bottom_right.plot(data.loc[mask,"TIME"], data.loc[mask,'distance'], marker='.', color=color, ls="None")
                # elif "secondary" in col:
                #     mask = data[col] == 4
                #     color = "red"
                #     ax_top_right.plot(data.loc[mask,"TIME"], data.loc[mask,'LATITUDE'], marker='.', color=color)
                #     ax_middle_right.plot(data.loc[mask,"TIME"], data.loc[mask,'LONGITUDE'], marker='.', color=color)
                #     ax_bottom_right.plot(data.loc[mask,"TIME"], data.loc[mask,'distance'], marker='.', color=color)
                
                data_suspect = data.loc[data[col] == 3]
                color = "goldenrod"
                ax_top_right.plot(data_suspect["TIME"], data_suspect['LATITUDE'], marker='.', color=color, ls="None")
                ax_middle_right.plot(data_suspect["TIME"], data_suspect['LONGITUDE'], marker='.', color=color, ls="None")
                ax_bottom_right.plot(data_suspect["TIME"], data_suspect['distance'], marker='.', color=color, ls="None")
                
                data_fail = data.loc[data[col] == 4]
                color = "red"
                ax_top_right.plot(data_fail["TIME"], data_fail['LATITUDE'], marker='.', color=color)
                ax_middle_right.plot(data_fail["TIME"], data_fail['LONGITUDE'], marker='.', color=color)
                ax_bottom_right.plot(data_fail["TIME"], data_fail['distance'], marker='.', color=color)


            # Plots configs
            ax_left.set_xlim(lon_min, lon_max)
            ax_left.set_ylim(lat_min, lat_max)

            ax_left.set_xlabel("Longitude")
            ax_left.set_ylabel("Latitude")
            
            handles, labels = ax_left.get_legend_handles_labels()
            handles.append(circle_handle)
            labels.append(watch_circle_label)
            ax_left.legend(handles, labels)
            ax_left.set_title(f"{self.deployment_folder} - {coverage} km")

            ax_left.grid()
            ax_top_right.grid()
            ax_middle_right.grid()
            ax_bottom_right.grid()

            ax_top_right.set_ylabel("LATITUDE")
            ax_middle_right.set_ylabel("LONGITUDE")
            ax_bottom_right.set_ylabel("Distance from\nWatchCircle Center\n(m)")
            ax_bottom_right.legend()

            ax_top_right.tick_params(labelbottom=False)
            ax_middle_right.tick_params(labelbottom=False)

            from matplotlib.dates import DayLocator, DateFormatter
            ax_top_right.xaxis.set_major_locator(DayLocator(interval=10))
            ax_middle_right.xaxis.set_major_locator(DayLocator(interval=10))
            ax_bottom_right.xaxis.set_major_locator(DayLocator(interval=10))
            
            ax_bottom_right.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            for label in ax_bottom_right.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")

            plt.tight_layout()
            output_file_name = f"{self.site_name}_positions-{coverage}km.png"
            plt.savefig(os.path.join(self.output_path, output_file_name), dpi=100, bbox_inches="tight")
