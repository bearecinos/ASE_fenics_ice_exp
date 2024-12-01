"""
Crops satellite velocities for individual years and composite means
 from different velocity data products to an area extent defined (for now)
  for the ase glacier experiment.

Options for mosaic
- MEaSUREs 1996 to 31 December 2016
- ENVEO  2014 to August - 2023

Options for the cloud point velocity:
- Measures without filling gaps 2014
- ENVEO without filling gaps 2014
- It is possible to enhance the STD velocity error or
    value by a factor X, by specifying -error_factor as default
    this is equal to one.

The code generates a .h5 file for each data product (MEaSUREs and ENVEO),
 with the corresponding velocity file suffix, depending on the data type:
e.g. `*_enveo-comp_enveo-cloud_2014-error-factor-1E+0`

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

ef = 1.0
mu = 0.0
sigma = 30.0
year = 2014

############## Sort out paths for all data files ################
path_enveo_main = config['input_files']['enveo_vel']
file_names = sorted(os.listdir(path_enveo_main))

paths_enveo = []
for f in file_names:
    paths_enveo.append(os.path.join(path_enveo_main, f))

print(paths_enveo)

mosaic_enveo_file_path = paths_enveo[0]
assert '20141001-20190831' in paths_enveo[0]

cloud_enveo_file_path = paths_enveo[1]
assert '20150401-20160331' in paths_enveo[1]

mosaic_measures_file_path = config['input_files']['measures_comp_interp']
print('The velocity product for the cloud '
      'point data its Measures from ' + str(year-1) + 'to' + str(year))
cloud_measures_file_path = utils_funcs.find_measures_file(year,
                                                          config['input_files']['measures_cloud'])
print(cloud_measures_file_path)

assert '_' + str(year-1) + '_' + str(year) + \
       '_1km_v01.nc' in cloud_measures_file_path, \
    "File does not exist check main measures data folder"

# 1) Get data MEaSUREs mosaic (dmm)
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

# Mask arrays and make sure nans are drop in both
# enveo and Measures
xmm_grid, ymm_grid = np.meshgrid(xmm_s, ymm_s)

mask_array = vxmm_s * vxmm_std_s
array_ma = np.ma.masked_invalid(mask_array)

xmm_nona = xmm_grid[~array_ma.mask].ravel()
ymm_nona = ymm_grid[~array_ma.mask].ravel()
vxmm_nona = vxmm_s[~array_ma.mask].ravel()
vymm_nona = vymm_s[~array_ma.mask].ravel()
stdvxmm_nona = vxmm_std_s[~array_ma.mask].ravel()
stdvymm_nona = vymm_std_s[~array_ma.mask].ravel()

### Get ENVEO mosaic data and interpolate to measures grid
# 2) data enveo mosaic (denv)
denv = xr.open_dataset(mosaic_enveo_file_path)

denv_int = vel_tools.interp_enveo_to_compatible_grid(denv, dmm, ase_bbox)

vx_e = denv_int.vx.data
vy_e = denv_int.vy.data
vxstd_e = denv_int.vx_std.data
vystd_e = denv_int.vy_std.data

x_enveo = denv_int.x
y_enveo = denv_int.y

assert sorted(x_enveo) == sorted(xmm_s)
assert sorted(y_enveo) == sorted(ymm_s)

xem_grid, yem_grid = np.meshgrid(x_enveo, y_enveo)

mask_array = vx_e*vystd_e
array_ma = np.ma.masked_invalid(mask_array)

xenveo_nona = xem_grid[~array_ma.mask].ravel()
yenveo_nona = yem_grid[~array_ma.mask].ravel()

vx_enveo_nona = vx_e[~array_ma.mask].ravel()
vy_enveo_nona = vy_e[~array_ma.mask].ravel()
stdvxem_nona = vxstd_e[~array_ma.mask].ravel()
stdvyem_nona = vystd_e[~array_ma.mask].ravel()

if args.add_noise_to_data:
    noise_ex = 0.01 * np.random.randn(vx_enveo_nona.shape[0])
    noise_mx = 0.01 * np.random.randn(vxmm_nona.shape[0])
    noise_ey = 0.01 * np.random.randn(vy_enveo_nona.shape[0])
    noise_my = 0.01 * np.random.randn(vymm_nona.shape[0])
    vx_enveo_nona = vx_enveo_nona + noise_ex
    vy_enveo_nona = vy_enveo_nona + noise_ey
    vxmm_nona = vxmm_nona + noise_mx
    vymm_nona = vymm_nona + noise_my

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
composite_dict_e = {'x_comp': xenveo_nona,
                    'y_comp': yenveo_nona,
                    'vx_comp': vx_enveo_nona,
                    'vy_comp': vy_enveo_nona,
                    'std_vx_comp': stdvxem_nona,
                    'std_vy_comp': stdvyem_nona}

composite_dict_m = {'x_comp': xmm_nona,
                    'y_comp': ymm_nona,
                    'vx_comp': vxmm_nona,
                    'vy_comp': vymm_nona,
                    'std_vx_comp': stdvxmm_nona,
                    'std_vy_comp': stdvymm_nona}

# 3) For data cloud enveo (dci)
dec = xr.open_mfdataset(cloud_enveo_file_path)

vxce = dec.land_ice_surface_easting_velocity.load()*365.25
vyce = dec.land_ice_surface_northing_velocity.load()*365.25
std_vxce = dec.land_ice_surface_easting_stddev.load()*365.25
std_vyce = dec.land_ice_surface_northing_stddev.load()*365.25

vxce_s, xce_s, yce_s = vel_tools.crop_velocity_data_to_extend(vxce,
                                                              ase_bbox,
                                                              return_coords=True)

vyce_s = vel_tools.crop_velocity_data_to_extend(vyce, ase_bbox)
vxce_err_s = vel_tools.crop_velocity_data_to_extend(std_vxce, ase_bbox)
vyce_err_s = vel_tools.crop_velocity_data_to_extend(std_vyce, ase_bbox)

# Mask arrays
xce_grid, yce_grid = np.meshgrid(xce_s, yce_s)

# array to mask ... a dot product of component and std
mask_array = vxce_s * vyce_err_s
array_ma = np.ma.masked_invalid(mask_array)

# get only the valid values
xce_nona = xce_grid[~array_ma.mask].ravel()
yce_nona = yce_grid[~array_ma.mask].ravel()
vxce_nona = vxce_s[~array_ma.mask].ravel()
vyce_nona = vyce_s[~array_ma.mask].ravel()
stdvxce_nona = vxce_err_s[~array_ma.mask].ravel()
stdvyce_nona = vyce_err_s[~array_ma.mask].ravel()

# Ravel all arrays so they can be stored with
# a tuple shape (values, )
cloud_dict_e = {'x_cloud': xce_nona,
                'y_cloud': yce_nona,
                'vx_cloud': vxce_nona,
                'vy_cloud': vyce_nona,
                'std_vx_cloud': stdvxce_nona,
                'std_vy_cloud': stdvyce_nona}

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
composite_e = 'enveo' + '-comp_'
cloud_e = 'enveo' + '-cloud_'

composite_m = 'measures' + '-comp_'
cloud_m = 'measures' + '-cloud_'

file_suffix_e = composite_e + \
                cloud_e + \
                str(year) + '-' + \
                'error-factor-' + "{:.0E}".format(Decimal(ef))

file_suffix_m = composite_m + \
                cloud_m + \
                str(year) + '-' + \
                'error-factor-' + "{:.0E}".format(Decimal(ef))

if args.add_noise_to_data:
    file_suffix_e = file_suffix_e + '-added-noise'
    file_suffix_m = file_suffix_m + '-added-noise'

file_ext = '.h5'

file_name_e = os.path.join(MAIN_PATH,
                           config['output_files']['ase_vel_obs'] +
                           file_suffix_e +
                           file_ext)

file_name_m = os.path.join(MAIN_PATH,
                           config['output_files']['ase_vel_obs'] +
                           file_suffix_m +
                           file_ext)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict_e,
                                      cloud_dict=cloud_dict_e,
                                      fpath=file_name_e)

vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict_m,
                                      cloud_dict=cloud_dict_m,
                                      fpath=file_name_m)
