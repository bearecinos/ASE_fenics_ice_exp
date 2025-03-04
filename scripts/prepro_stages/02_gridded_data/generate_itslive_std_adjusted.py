"""
Crops ITSLive velocity data to the ASE domain for a mosaic and cloud
point data distribution. By default this script will always use ITSLive data
and always provide cloud a cloud point data distribution.

Additionally, the STD of each vel component its adjusted with the absolute
difference between MEaSUREs 2014 and ITSLive 2014 data set.

Options for the composite velocity mosaic:
- ITSLIVE only

Options for the cloud point velocity:
- ITSLIVE 2014
- ITSLIVE 2014 with STDvx and STDvy adjusted according to the following:
    `np.maxima(vx_err_s, np.abs(vx_s-vx_mi_s))`
    where `vx_s-vx_mi_s` is the velocity difference between ITslive 2014
    and MEaSUREs 2014 (MEaSUREs was interpolated to the itslive grid).
- It is possible to subsample each cloud data set by selecting the top left
    value of every sub-grid box, which size is determined by the variable input: -step

The code generates one or two .h5 files; if a step is chosen, with the corresponding velocity
file suffix:
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-training_step-1E+1.h5`
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-test_step-1E+1.h5`

If there is no subsampling:
e.g. `*_itslive-comp_std-adjusted-cloud_subsample-none_step-0E+0.h5`

The files contain the following variables stored as tuples
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
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-compute_interpolation",
                    action="store_true",
                    help="If true computes the interpolation of MEaSUREs 2014 vel"
                         "file to the ITSLive grid and saves it as a netcdf file for re-use.")
parser.add_argument("-step",
                    type=int, default=10,
                    help="Sub-box size for the subsample")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

from ficetools import velocity as vel_tools

# Define the ASE extent to crop all velocity data to this region
# IMPORTANT .. verify that the extent is always bigger than the mesh!
ase_bbox = {}
for key in config['data_input_extent'].keys():
    ase_bbox[key] = np.float64(config['data_input_extent'][key])

print('The velocity product for the composite solution will be ITSlive')
print('This choice is slightly slower '
      'as the files are in a very different format than MEaSUREs')

# First load and process ITSLive data for storing
# a composite mean of all velocity components and uncertainty
path_itslive = os.path.join(MAIN_PATH,
                            config['input_files']['itslive'])
file_names = os.listdir(path_itslive)

paths_itslive = []
for f in file_names:
    paths_itslive.append(os.path.join(path_itslive, f))

paths_itslive = sorted(paths_itslive)
print(paths_itslive)

assert '_0000.nc' in paths_itslive[0]
assert '_2014.nc' in paths_itslive[4]

dv = xr.open_dataset(paths_itslive[0])

vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv)

vx_s, x_s, y_s = vel_tools.crop_velocity_data_to_extend(vx, ase_bbox,
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

print('The velocity product for the cloud '
      'point data its ITSlive 2014 with the STD adjusted')
# Opening files with salem slower than rasterio
# but they end up more organised in xr.DataArrays
path_measures = os.path.join(MAIN_PATH, config['input_files']['measures_cloud'])
dm = xr.open_dataset(path_measures)

vx_m = dm.VX
vy_m = dm.VY
std_vx_m = dm.STDX
std_vy_m = dm.STDY

dv = xr.open_dataset(paths_itslive[4])
vx, vy, std_vx, std_vy = vel_tools.process_itslive_netcdf(dv)

fpath = os.path.join(os.path.dirname(os.path.abspath(path_measures)),
                     'measures_in_itslive_grid.nc')
#
if args.compute_interpolation:
    vx_mi = vx_m.interp(y=dv.y.values, x=dv.x.values)
    vy_mi = vy_m.interp(y=dv.y.values, x=dv.x.values)
    std_vx_mi = std_vx_m.interp(y=dv.y.values, x=dv.x.values)
    std_vy_mi = std_vy_m.interp(y=dv.y.values, x=dv.x.values)

    with vel_tools.ncDataset(fpath, 'w', format='NETCDF4') as nc:

        nc.author = 'B.M Recinos'
        nc.author_info = 'The University of Edinburgh'

        x_dim = nc.createDimension('x', len(vx_mi.x.values)) # latitude axis
        y_dim = nc.createDimension('y', len(vx_mi.y.values))

        v = nc.createVariable('x', 'f4', ('x',))
        v.units = 'm'
        v.long_name = 'x coordinates'
        v[:] = vx_mi.x.values

        v = nc.createVariable('y', 'f4', ('y',))
        v.units = 'm'
        v.long_name = 'y coordinates'
        v[:] = vx_mi.y.values

        v = nc.createVariable('lat', 'f4', ('y','x'))
        v.units = 'degrees'
        v.long_name = 'latitude'
        v[:] = vx_mi.lat.values

        v = nc.createVariable('lon', 'f4', ('y','x'))
        v.units = 'degrees'
        v.long_name = 'longitude'
        v[:] = vx_mi.lon.values

        v = nc.createVariable('vx', 'f4', ('y','x'))
        v.units = 'm/yr'
        v.long_name = 'vx velocity component'
        v[:] = vx_mi.data

        v = nc.createVariable('vy', 'f4', ('y','x'))
        v.units = 'm/yr'
        v.long_name = 'vy velocity component'
        v[:] = vy_mi.data

        v = nc.createVariable('std_vx', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'std vx velocity component'
        v[:] = std_vx_mi.data

        v = nc.createVariable('std_vy', 'f4', ('y', 'x'))
        v.units = 'm/yr'
        v.long_name = 'std vy velocity component'
        v[:] = std_vy_mi.data

ds = xr.open_dataset(fpath)

# Crop data to the ase domain
# We start with measures interpolated
vx_mi_s = vel_tools.crop_velocity_data_to_extend(ds.vx, ase_bbox, return_xarray=True)
vy_mi_s = vel_tools.crop_velocity_data_to_extend(ds.vy, ase_bbox, return_xarray=True)

# now itslive
vxc_s = vel_tools.crop_velocity_data_to_extend(vx, ase_bbox,
                                                       return_xarray=True)
vyc_s = vel_tools.crop_velocity_data_to_extend(vy, ase_bbox, return_xarray=True)
vxc_err_s = vel_tools.crop_velocity_data_to_extend(std_vx, ase_bbox, return_xarray=True)
vyc_err_s = vel_tools.crop_velocity_data_to_extend(std_vy, ase_bbox, return_xarray=True)

# We adjust the STD
diff_vx = np.abs(vxc_s - vx_mi_s)
diff_vy = np.abs(vyc_s -vy_mi_s)

vxc_err_its_2014_adjust = vel_tools.create_adjusted_std_maxima(vxc_err_s, diff_vx)
vyc_err_its_2014_adjust = vel_tools.create_adjusted_std_maxima(vyc_err_s, diff_vy)

step = abs(args.step)

# Are we subsampling this one? yes only with step > 0.

if step != 0:
    print('This is happening')
    # Computing our training sets of cloud velocities
    # vx_trn_0 is the upper left
    # vx_trn_m is the middle point of the array
    (x_trn_0, y_trn_0, vx_trn_0), (x_trn_m, y_trn_m, vx_trn_m) = vel_tools.create_subsample(vxc_s, step)

    (_, _, vy_trn_0), (_, _, vy_trn_m) = vel_tools.create_subsample(vyc_s, step)
    (_, _, vx_std_trn_0), (_, _, vx_std_trn_m) = vel_tools.create_subsample(vxc_err_its_2014_adjust, step)
    (_, _, vy_std_trn_0), (_, _, vy_std_trn_m) = vel_tools.create_subsample(vyc_err_its_2014_adjust, step)

    # Computing our test set of cloud velocities
    for x_0, y_0 in zip(x_trn_0, y_trn_0):
        vxc_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vyc_s.loc[dict(x=x_0, y=y_0)] = np.nan
        vxc_err_its_2014_adjust.loc[dict(x=x_0, y=y_0)] = np.nan
        vyc_err_its_2014_adjust.loc[dict(x=x_0, y=y_0)] = np.nan

    for x_m, y_m in zip(x_trn_m, y_trn_m):
        vxc_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vyc_s.loc[dict(x=x_m, y=y_m)] = np.nan
        vxc_err_its_2014_adjust.loc[dict(x=x_m, y=y_m)] = np.nan
        vyc_err_its_2014_adjust.loc[dict(x=x_m, y=y_m)] = np.nan

    # Dropping the Nans from the training set
    out_cloud_0 = vel_tools.drop_nan_from_multiple_numpy(x_trn_0, y_trn_0,
                                                         vx_trn_0, vy_trn_0,
                                                         vx_std_trn_0, vy_std_trn_0)

    out_cloud_m = vel_tools.drop_nan_from_multiple_numpy(x_trn_m, y_trn_m,
                                                         vx_trn_m, vy_trn_m,
                                                         vx_std_trn_m, vy_std_trn_m)

    cloud_dict_training_0 = {f'{name:s}_cloud': getattr(out_cloud_0, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    cloud_dict_training_m = {f'{name:s}_cloud': getattr(out_cloud_m, name).values for name in ['x', 'y',
                                                                                               'vx', 'vy',
                                                                                               'std_vx', 'std_vy']}

    # Dropping the nans from the TEST set
    masked_array = np.ma.masked_invalid(vxc_s.data*vxc_err_its_2014_adjust.data)

    out_test = vel_tools.drop_invalid_data_from_several_arrays(vxc_s.x.values,
                                                               vxc_s.y.values,
                                                               vxc_s,
                                                               vyc_s,
                                                               vxc_err_its_2014_adjust,
                                                               vyc_err_its_2014_adjust,
                                                               masked_array)

    cloud_dict_test = {'x_cloud': out_test[0],
                       'y_cloud': out_test[1],
                       'vx_cloud': out_test[2],
                       'vy_cloud': out_test[3],
                       'std_vx_cloud': out_test[4],
                       'std_vy_cloud': out_test[5]}

    # We write the training file first
    file_suffix_0 = 'itslive-comp_std-adjusted-cloud_subsample-training-step-zero-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name_training_0 = os.path.join(MAIN_PATH, config['output_files']['ase_vel_obs'] + file_suffix_0)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict_training_0,
                                          fpath=file_name_training_0)

    file_suffix_m = 'itslive-comp_std-adjusted-cloud_subsample-training-step-middle-' + \
                    "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name_training_m = os.path.join(MAIN_PATH, config['output_files']['ase_vel_obs'] + file_suffix_m)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict_training_m,
                                          fpath=file_name_training_m)

    # We write the test file second
    file_suffix = 'itslive-comp_std-adjusted-cloud_subsample-test_step-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name_test = os.path.join(MAIN_PATH, config['output_files']['ase_vel_obs'] + file_suffix)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict_test,
                                          fpath=file_name_test)
else:
    # We write the complete cloud array!

    # Mask arrays
    x_s = vxc_s.x.data
    y_s = vxc_s.y.data

    x_grid, y_grid = np.meshgrid(x_s, y_s)

    # array to mask ... a dot product of component and std
    mask_array = vxc_s.data * vxc_err_its_2014_adjust.data

    # Remove nan from cloud data
    array_ma = np.ma.masked_invalid(mask_array)

    # get only the valid values
    x_nona = x_grid[~array_ma.mask].ravel()
    y_nona = y_grid[~array_ma.mask].ravel()
    vx_nona = vxc_s.data[~array_ma.mask].ravel()
    vy_nona = vyc_s.data[~array_ma.mask].ravel()
    stdvx_nona = vxc_err_its_2014_adjust.data[~array_ma.mask].ravel()
    stdvy_nona = vyc_err_its_2014_adjust.data[~array_ma.mask].ravel()

    # Ravel all arrays so they can be stored with
    # a tuple shape (values, )
    cloud_dict = {'x_cloud': x_nona,
                  'y_cloud': y_nona,
                  'vx_cloud': vx_nona,
                  'vy_cloud':  vy_nona,
                  'std_vx_cloud': stdvx_nona,
                  'std_vy_cloud': stdvy_nona}


    # We write the test file second
    file_suffix = 'itslive-comp_std-adjusted-cloud_subsample-none_step-' + \
                  "{:.0E}".format(Decimal(args.step)) + '.h5'

    file_name = os.path.join(MAIN_PATH, config['output_files']['ase_vel_obs'] + file_suffix)

    vel_tools.write_velocity_tuple_h5file(comp_dict=composite_dict,
                                          cloud_dict=cloud_dict,
                                          fpath=file_name)