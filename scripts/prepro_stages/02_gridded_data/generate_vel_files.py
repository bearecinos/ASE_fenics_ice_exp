"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the ase glacier experiment.

Options for the composite velocity mosaic:
- MEaSUREs
- ITSLIVE

Options for the cloud point velocity:
- Measures without filling gaps 2014
- ITSLIVE without filling gaps 2014
- It is possible to enhance the STD velocity error by a factor X,
    by specifying -error_factor as default this is equal to one.

The code generates a .h5 file, with the corresponding velocity
file suffix, depending on what has been chosen as data:
e.g. `*_itslive-comp_itslive-cloud_error-factor-1E+0`

The file contain the following variables stored as tuples
and containing no np.nan's:
- 'u_cloud', 'u_cloud_std', 'v_cloud', 'v_cloud_std', 'x_cloud', 'y_cloud'
- 'u_obs', 'u_std', 'v_obs', 'v_std', 'x', 'y', 'mask_vel' -> default composite

each variable with shape: (#values, )

@authors: Fenics_ice contributors
"""
import os
import sys
from configobj import ConfigObj
import numpy as np
import argparse
import xarray as xr
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
parser.add_argument("-composite",
                    type=str,
                    default='measures',
                    help="Data product for the composite velocities: "
                         "itslive or measures")
parser.add_argument("-add_cloud_data",
                    action="store_true",
                    help="If this is specify a year for the data is selected "
                         "we dont interpolate nans and add the data as it is "
                         " as cloud point velocities to the .h5 file")
parser.add_argument("-year",
                      type=int,
                      default= 2014,
                      help="if specify gives back vel "
                           "files corresponding that year")
parser.add_argument("-error_factor",
                    type=float, default=1.0,
                    help="Enlarge error in observation by a factor")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))


# Define main repository path
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

from ficetools import velocity as vel_tools
from ficetools import utils_funcs

# Define the ase Glacier extent to crop all velocity data to this region
# IMPORTANT .. verify that the extent is always bigger than the mesh!
ase_bbox = {}
for key in config['data_input_extent'].keys():
    ase_bbox[key] = np.float64(config['data_input_extent'][key])

ef = args.error_factor
year = args.year

# 1) Generate first composite velocities and uncertainties
if args.composite == 'itslive':
    print('The velocity product for the composite solution will be ITSlive')
    print('This choice is slightly slower '
          'as the files are in a very different format than MEaSUREs')

    # First load and process ITSLive data for storing
    # a composite mean of all velocity components and uncertainty
    path_itslive_main = os.path.join(MAIN_PATH,
                                config['input_files']['itslive'])
    file_names = os.listdir(path_itslive_main)

    paths_itslive = []
    for f in file_names:
        paths_itslive.append(os.path.join(path_itslive_main, f))

    paths_itslive = sorted(paths_itslive)
    print(paths_itslive)

    mosaic_file_path = paths_itslive[0]
    assert '_0000.nc' in mosaic_file_path

    cloud_file_path = utils_funcs.find_itslive_file(year, path_itslive_main)
    assert '_'+str(year)+'.nc' in cloud_file_path, 'File does not exist check main itslive data folder'

    dv = xr.open_dataset(mosaic_file_path)

    vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv,
                                                              error_factor=ef)

    vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx,
                                                            ase_bbox,
                                                            return_coords=True)

    vy_s = vel_tools.crop_velocity_data_to_extend(vy, ase_bbox)
    vx_std_s = vel_tools.crop_velocity_data_to_extend(std_vx, ase_bbox)
    vy_std_s = vel_tools.crop_velocity_data_to_extend(std_vy, ase_bbox)

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    array_ma = np.ma.masked_invalid(vx_s)

    # get only the valid values
    x_nona = x_grid[~array_ma.mask].ravel()
    y_nona = y_grid[~array_ma.mask].ravel()
    vx_nona = vx_s[~array_ma.mask].ravel()
    vy_nona = vy_s[~array_ma.mask].ravel()
    stdvx_nona = vx_std_s[~array_ma.mask].ravel()
    stdvy_nona = vy_std_s[~array_ma.mask].ravel()

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    composite_dict = {'x_comp': x_nona.ravel(),
                      'y_comp': y_nona.ravel(),
                      'vx_comp': vx_nona,
                      'vy_comp': vy_nona,
                      'std_vx_comp': stdvx_nona,
                      'std_vy_comp': stdvy_nona}

    cloud_dict = {'x_cloud': None,
                  'y_cloud': None,
                  'vx_cloud': None,
                  'vy_cloud': None,
                  'std_vx_cloud': None,
                  'std_vy_cloud': None}

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its ITSlive '+str(year))
        # Opening files with salem slower than rasterio
        # but they end up more organised in xr.DataArrays

        dv = xr.open_dataset(cloud_file_path)

        vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv,
                                                                  error_factor=ef)

        vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx,
                                                                ase_bbox,
                                                                return_coords=True)

        vy_s = vel_tools.crop_velocity_data_to_extend(vy, ase_bbox)
        vx_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, ase_bbox)
        vy_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, ase_bbox)

        # Mask arrays
        x_grid, y_grid = np.meshgrid(x_s, y_s)

        # array to mask ... a dot product of component and std
        mask_array = vx_s * vx_err_s

        array_ma = np.ma.masked_invalid(mask_array)

        # get only the valid values
        x_nona = x_grid[~array_ma.mask].ravel()
        y_nona = y_grid[~array_ma.mask].ravel()
        vx_nona = vx_s[~array_ma.mask].ravel()
        vy_nona = vy_s[~array_ma.mask].ravel()
        stdvx_nona = vx_err_s[~array_ma.mask].ravel()
        stdvy_nona = vy_err_s[~array_ma.mask].ravel()

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        cloud_dict = {'x_cloud': x_nona,
                      'y_cloud': y_nona,
                      'vx_cloud': vx_nona,
                      'vy_cloud': vy_nona,
                      'std_vx_cloud': stdvx_nona,
                      'std_vy_cloud': stdvy_nona}
else:
    print('The velocity product for the composite solution will be MEaSUREs')

    # First load and process MEaSUREs data for storing a composite mean of
    # all velocity components and uncertainty
    path_measures = os.path.join(MAIN_PATH,
                                 config['input_files']['measures_comp_interp'])

    dm = xr.open_dataset(path_measures)

    vx = dm.vx
    vy = dm.vy
    std_vx = dm.std_vx * ef
    std_vy = dm.std_vy * ef

    # Crop velocity data to the ase Glacier extend
    vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, ase_bbox,
                                                            return_coords=True)
    vy_s = vel_tools.crop_velocity_data_to_extend(vy, ase_bbox)
    std_vx_s = vel_tools.crop_velocity_data_to_extend(std_vx, ase_bbox)
    std_vy_s = vel_tools.crop_velocity_data_to_extend(std_vy, ase_bbox)

    # Mask arrays and interpolate nan with nearest neighbor
    x_grid, y_grid = np.meshgrid(x_s, y_s)
    
    mask_array = vx_s*std_vx_s
    array_ma = np.ma.masked_invalid(mask_array)

    # get only the valid values
    x_nona = x_grid[~array_ma.mask].ravel()
    y_nona = y_grid[~array_ma.mask].ravel()
    vx_nona = vx_s[~array_ma.mask].ravel()
    vy_nona = vy_s[~array_ma.mask].ravel()
    stdvx_nona = std_vx_s[~array_ma.mask].ravel()
    stdvy_nona = std_vy_s[~array_ma.mask].ravel()

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    composite_dict = {'x_comp': x_nona.ravel(),
                      'y_comp': y_nona.ravel(),
                      'vx_comp': vx_nona,
                      'vy_comp': vy_nona,
                      'std_vx_comp': stdvx_nona,
                      'std_vy_comp': stdvy_nona}

    cloud_dict = {'x_cloud': None,
                  'y_cloud': None,
                  'vx_cloud': None,
                  'vy_cloud': None,
                  'std_vx_cloud': None,
                  'std_vy_cloud': None}

    if args.add_cloud_data:
        print('The velocity product for the cloud '
              'point data its Measures from '+
              str(year-1) + 'to' + str(year))

        path_measures = utils_funcs.find_measures_file(year,
                                                       config['input_files']['measures_cloud'])
        print(path_measures)
        assert '_' + str(year-1) + '_' + str(year) + \
               '_1km_v01.nc' in path_measures, "File does not exist check main measures data folder"

        dm = xr.open_dataset(path_measures)

        vx = dm.VX
        vy = dm.VY
        std_vx = dm.STDX * ef
        std_vy = dm.STDY * ef

        # Crop velocity data to the ase Glacier extend
        vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx,
                                                                ase_bbox,
                                                                return_coords=True)
        vy_s = vel_tools.crop_velocity_data_to_extend(vy, ase_bbox)
        std_vx_s = vel_tools.crop_velocity_data_to_extend(std_vx, ase_bbox)
        std_vy_s = vel_tools.crop_velocity_data_to_extend(std_vy, ase_bbox)

        # Mask arrays and interpolate nan with nearest neighbor
        x_grid, y_grid = np.meshgrid(x_s, y_s)

        # array to mask ... a dot product of component and std
        mask_array = vx_s*std_vx_s

        array_ma = np.ma.masked_invalid(mask_array)

        # get only the valid values
        x_nona = x_grid[~array_ma.mask].ravel()
        y_nona = y_grid[~array_ma.mask].ravel()
        vx_nona = vx_s[~array_ma.mask].ravel()
        vy_nona = vy_s[~array_ma.mask].ravel()
        stdvx_nona = std_vx_s[~array_ma.mask].ravel()
        stdvy_nona = std_vy_s[~array_ma.mask].ravel()

        # Ravel all arrays so they can be stored with
        # a tuple shape (values, )
        cloud_dict = {'x_cloud': x_nona,
                      'y_cloud': y_nona,
                      'vx_cloud': vx_nona,
                      'vy_cloud': vy_nona,
                      'std_vx_cloud': stdvx_nona,
                      'std_vy_cloud': stdvy_nona}


composite = args.composite + '-comp_'
cloud = args.composite + '-cloud_'

if args.add_cloud_data:
    file_suffix = composite + \
                  cloud + \
                  str(year) + '-' +\
                  'error-factor-' + \
                  "{:.0E}".format(Decimal(ef)) + '.h5'
else:
    file_suffix = composite + \
                  'no-cloud_' + \
                  'error-factor-' + \
                  "{:.0E}".format(Decimal(ef)) + '.h5'

file_name = os.path.join(MAIN_PATH,
                         config['output_files']['ase_vel_obs'] +
                         file_suffix)


vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                      cloud_dict=cloud_dict,
                                      fpath=file_name)
