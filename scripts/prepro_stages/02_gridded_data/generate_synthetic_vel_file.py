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
import scipy.interpolate as interp
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
parser.add_argument("-vel_lambda",
                    type=float, default=0.01,
                    help="Lamda to interpolate between MEaSUREs and ITSLIVE")

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
vel_lambda = args.vel_lambda
print('This is the lambda value', vel_lambda)

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

### 3) generate synthetic data set for each component
#by:
#$\hat{p}_\lambda = (1 - \lambda) \hat{p}_M + \lambda \hat{p}_I$

vxss = (1-vel_lambda)*vxmm_s + vel_lambda*vxim_s
vyss = (1-vel_lambda)*vymm_s + vel_lambda*vyim_s

#STD should be interpolated to
# $\sigma_{lamda} = \sqrt (1-lamda)^{2} \sigma^{2}_{M} + lamda^{2} \sigma^{2}_{I}) $
vxss_std = np.sqrt((1 - vel_lambda)**2 * vxmm_std_s**2 + vel_lambda**2 * vxim_std_s**2)
vyss_std = np.sqrt((1 - vel_lambda)**2 * vymm_std_s**2 + vel_lambda**2 * vyim_std_s**2)

# Mask arrays and make sure nans are drop in both
# Itslive and Measures
xss_grid, yss_grid = np.meshgrid(xmm_s, ymm_s)

mask_array = vxss*vxss_std
array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values for both mosaics
xsm_nona = xss_grid[~array_ma.mask].ravel()
ysm_nona = yss_grid[~array_ma.mask].ravel()

vxsm_nona = vxss[~array_ma.mask].ravel()
vysm_nona = vyss[~array_ma.mask].ravel()

stdvxsm_nona = vxss_std[~array_ma.mask].ravel()
stdvysm_nona = vyss_std[~array_ma.mask].ravel()

composite_dict_s = {'x_comp': xsm_nona,
                    'y_comp': ysm_nona,
                    'vx_comp': vxsm_nona,
                    'vy_comp': vysm_nona,
                    'std_vx_comp': stdvxsm_nona,
                    'std_vy_comp': stdvysm_nona}

# 4) For data Cloud MEaSUREs (dcm) this is just so the file can be read by fenics_ice
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
composite_s = 'synthetic' + '-comp_'
cloud_s = 'synthetic' + '-cloud_'

file_suffix_s = composite_s + \
                cloud_s + \
                str(year) + '-' + \
                'lamda-' + "{:.0E}".format(Decimal(vel_lambda))

file_ext = '.h5'

file_name_s = os.path.join(MAIN_PATH,
                           config['output_files']['ase_vel_obs'] +
                           file_suffix_s +
                           file_ext)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict_s,
                                      cloud_dict=cloud_dict_m,
                                      fpath=file_name_s)