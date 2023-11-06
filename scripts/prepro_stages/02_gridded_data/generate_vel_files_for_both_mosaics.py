"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the ase glacier experiment.

Options for mosaic
- MEaSUREs 1996 to 31 December 2016
- ITSLIVE  1985-01-01 to 2018-12-31

Options for the cloud point velocity:
- Measures without filling gaps 2014
- ITSLIVE without filling gaps 2014
- It is possible to enhance the STD velocity error or
    value by a factor X, by specifying -error_factor as default
    this is equal to one.

The code generates a .h5 file for each data product (MEaSUREs and ITSLIVE),
 with the corresponding velocity file suffix, depending on the data type:
e.g. `*_itslive-comp_itslive-cloud_2014-error-factor-1E+0`

Each file contain the following variables stored as tuples
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
parser.add_argument("-year",
                      type=int,
                      default= 2014,
                      help="if specify gives back vel "
                           "files corresponding that year")
parser.add_argument("-error_factor",
                    type=float, default=1.0,
                    help="Enlarge error in observation by a factor")
parser.add_argument("-add_noise_to_data",
                    action="store_true",
                    help="If this is specify noise is added "
                         "to each vel_component .h5 file")

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
mu = 0.0
sigma = 30.0
year = args.year

############## Sort out paths for all data files ################
path_itslive_main = config['input_files']['itslive']
file_names = os.listdir(path_itslive_main)

paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive_main, f))

paths_itslive = sorted(paths_itslive)
print(paths_itslive)

mosaic_itslive_file_path = paths_itslive[0]
assert '_0000.nc' in mosaic_itslive_file_path

cloud_itslive_file_path = utils_funcs.find_itslive_file(year, path_itslive_main)
assert '_'+str(year)+'.nc' in cloud_itslive_file_path, \
    'File does not exist check main itslive data folder'

mosaic_measures_file_path = config['input_files']['measures_comp_interp']

print('The velocity product for the cloud '
      'point data its Measures from ' + str(year-1) + 'to' + str(year))

cloud_measures_file_path = utils_funcs.find_measures_file(year,
                                                          config['input_files']['measures_cloud'])
print(cloud_measures_file_path)

assert '_' + str(year-1) + '_' + str(year) + \
       '_1km_v01.nc' in cloud_measures_file_path, \
    "File does not exist check main measures data folder"

### Get ITS_LIVE mosaic data
# 1) data itslive mosaic (dim)
dim = xr.open_dataset(mosaic_itslive_file_path)

vxim, vyim, std_vxim, std_vyim = vel_tools.process_itslive_netcdf(dim,
                                                                      error_factor=ef)
# section of the data
vxim_s, xim_s, yim_s = vel_tools.crop_velocity_data_to_extend(vxim,
                                                              ase_bbox,
                                                              return_coords=True)

vyim_s = vel_tools.crop_velocity_data_to_extend(vyim, ase_bbox)
vxim_std_s = vel_tools.crop_velocity_data_to_extend(std_vxim, ase_bbox)
vyim_std_s = vel_tools.crop_velocity_data_to_extend(std_vyim, ase_bbox)

# 2) Get data MEaSUREs mosaic (dmm)
dmm = xr.open_dataset(mosaic_measures_file_path)

vxmm = dmm.vx
vymm = dmm.vy
std_vxmm = dmm.std_vx * ef
std_vymm = dmm.std_vy * ef

# Crop velocity data to the ase Glacier extend
vxmm_s, xmm_s, ymm_s = vel_tools.crop_velocity_data_to_extend(vxmm,
                                                              ase_bbox,
                                                              return_coords=True)

vymm_s = vel_tools.crop_velocity_data_to_extend(vymm, ase_bbox)
vxmm_std_s = vel_tools.crop_velocity_data_to_extend(std_vxmm, ase_bbox)
vymm_std_s = vel_tools.crop_velocity_data_to_extend(std_vymm, ase_bbox)

# Make sure coordinates are the same
assert len(xim_s) == len(xmm_s)
assert sorted(xim_s) == sorted(xmm_s)
assert sorted(yim_s) == sorted(ymm_s)

# Mask arrays and make sure same nans are drop in both
# Itslive and Measures
xim_grid, yim_grid = np.meshgrid(xim_s, yim_s)

mask_array = vxim_s * (vxmm_s * vxmm_std_s)

array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values for ITS_LIVE mosaic
xim_nona = xim_grid[~array_ma.mask].ravel()
yim_nona = yim_grid[~array_ma.mask].ravel()
vxim_nona = vxim_s[~array_ma.mask].ravel()
vyim_nona = vyim_s[~array_ma.mask].ravel()
stdvxim_nona = vxim_std_s[~array_ma.mask].ravel()
stdvyim_nona = vyim_std_s[~array_ma.mask].ravel()

vxmm_nona = vxmm_s[~array_ma.mask].ravel()
vymm_nona = vymm_s[~array_ma.mask].ravel()
stdvxmm_nona = vxmm_std_s[~array_ma.mask].ravel()
stdvymm_nona = vymm_std_s[~array_ma.mask].ravel()

if args.add_noise_to_data:
    noise = np.random.normal(loc=mu, scale=sigma, size=vxim_nona.shape)
    vxim_nona = vxim_nona + noise
    vyim_nona = vyim_nona + noise
    vxmm_nona = vxmm_nona + noise
    vymm_nona = vymm_nona + noise

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
composite_dict_i = {'x_comp': xim_nona,
                  'y_comp': yim_nona,
                  'vx_comp': vxim_nona,
                  'vy_comp': vyim_nona,
                  'std_vx_comp': stdvxim_nona,
                  'std_vy_comp': stdvyim_nona}

composite_dict_m = {'x_comp': xim_nona,
                  'y_comp': yim_nona,
                  'vx_comp': vxmm_nona,
                  'vy_comp': vymm_nona,
                  'std_vx_comp': stdvxmm_nona,
                  'std_vy_comp': stdvymm_nona}

# 3) For data cloud itslive (dci)
dci = xr.open_dataset(cloud_itslive_file_path)

vxci, vyci, std_vxci, std_vyci = vel_tools.process_itslive_netcdf(dci,
                                                                  error_factor=ef)

vxci_s, xci_s, yci_s = vel_tools.crop_velocity_data_to_extend(vxci,
                                                              ase_bbox,
                                                              return_coords=True)

vyci_s = vel_tools.crop_velocity_data_to_extend(vyci, ase_bbox)
vxci_err_s = vel_tools.crop_velocity_data_to_extend(std_vxci, ase_bbox)
vyci_err_s = vel_tools.crop_velocity_data_to_extend(std_vyci, ase_bbox)

# Mask arrays
xci_grid, yci_grid = np.meshgrid(xci_s, yci_s)

# array to mask ... a dot product of component and std
mask_array = vxci_s * vxci_err_s
array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values
xci_nona = xci_grid[~array_ma.mask].ravel()
yci_nona = yci_grid[~array_ma.mask].ravel()
vxci_nona = vxci_s[~array_ma.mask].ravel()
vyci_nona = vyci_s[~array_ma.mask].ravel()
stdvxci_nona = vxci_err_s[~array_ma.mask].ravel()
stdvyci_nona = vyci_err_s[~array_ma.mask].ravel()

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
cloud_dict_i = {'x_cloud': xci_nona,
              'y_cloud': yci_nona,
              'vx_cloud': vxci_nona,
              'vy_cloud': vyci_nona,
              'std_vx_cloud': stdvxci_nona,
              'std_vy_cloud': stdvyci_nona}

# 4) For data Cloud MEaSUREs (dcm)
dcm = xr.open_dataset(cloud_measures_file_path)

vxcm = dcm.VX
vycm = dcm.VY
std_vxcm = dcm.STDX * ef
std_vycm = dcm.STDY * ef

# Crop velocity data to the ase Glacier extend
vxcm_s, xcm_s, ycm_s = vel_tools.crop_velocity_data_to_extend(vxcm,
                                                              ase_bbox,
                                                              return_coords=True)
vycm_s = vel_tools.crop_velocity_data_to_extend(vycm, ase_bbox)
std_vxcm_s = vel_tools.crop_velocity_data_to_extend(std_vxcm, ase_bbox)
std_vycm_s = vel_tools.crop_velocity_data_to_extend(std_vycm, ase_bbox)

# Mask arrays and interpolate nan with nearest neighbor
xcm_grid, ycm_grid = np.meshgrid(xcm_s, ycm_s)

# array to mask ... a dot product of component and std
mask_array = vxcm_s*std_vxcm_s
array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values
xcm_nona = xcm_grid[~array_ma.mask].ravel()
ycm_nona = ycm_grid[~array_ma.mask].ravel()
vxcm_nona = vxcm_s[~array_ma.mask].ravel()
vycm_nona = vycm_s[~array_ma.mask].ravel()
stdvxcm_nona = std_vxcm_s[~array_ma.mask].ravel()
stdvycm_nona = std_vycm_s[~array_ma.mask].ravel()

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
cloud_dict_m = {'x_cloud': xcm_nona,
              'y_cloud': ycm_nona,
              'vx_cloud': vxcm_nona,
              'vy_cloud': vycm_nona,
              'std_vx_cloud': stdvxcm_nona,
              'std_vy_cloud': stdvycm_nona}

## Sorting names
composite_i = 'itslive' + '-comp_'
cloud_i = 'itslive' + '-cloud_'

composite_m = 'measures' + '-comp_'
cloud_m = 'measures' + '-cloud_'

file_suffix_i = composite_i + \
                cloud_i + \
                str(year) + '-' + \
                'error-factor-' + "{:.0E}".format(Decimal(ef))

file_suffix_m = composite_m + \
                cloud_m + \
                str(year) + '-' + \
                'error-factor-' + "{:.0E}".format(Decimal(ef))



if args.add_noise_to_data:
    file_suffix_i = file_suffix_i + '-added-noise'
    file_suffix_m = file_suffix_m + '-added-noise'

file_ext = '.h5'

file_name_i = os.path.join(MAIN_PATH,
                           config['output_files']['ase_vel_obs'] +
                           file_suffix_i +
                           file_ext)

file_name_m = os.path.join(MAIN_PATH,
                           config['output_files']['ase_vel_obs'] +
                           file_suffix_m +
                           file_ext)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict_i,
                                      cloud_dict=cloud_dict_i,
                                      fpath=file_name_i)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict_m,
                                      cloud_dict=cloud_dict_m,
                                      fpath=file_name_m)
