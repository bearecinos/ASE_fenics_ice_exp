import sys
import salem
import os
import argparse
import glob
import numpy as np
from numpy import inf
from configobj import ConfigObj
import seaborn as sns
import geopandas as gpd
import xarray as xr
import pandas as pd
import pyproj
from pathlib import Path
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from fenics_ice import inout, model, solver

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-main_path_tomls",
                    type=str,
                    default="",
                    help="pass main path to gather .toml files")
parser.add_argument('-n_sens',
                    nargs="+",
                    type=int,
                    help="pass n_sens to plot (max 2)")
parser.add_argument("-sub_plot_dir",
                    type=str,
                    default="temp",
                    help="pass sub plot directory to store the plots")
parser.add_argument("-sub_plot_name", type=str,
                    default="temp", help="pass filename")
parser.add_argument("-save_regrid_output", action="store_true",
                    help="If this is specify we "
                         "regrid the model output and save a netcdf for the given num_sens.")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/' + sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import utils_funcs, graphics, velocity
from ficetools.backend import FunctionSpace, VectorFunctionSpace, Function, project

main_run_path = os.path.join(MAIN_PATH, str(args.main_path_tomls))
print(main_run_path)
assert os.path.exists(main_run_path), "Provide the right path to tomls for the runs"

run_path_measures = os.path.join(main_run_path, 'ase_itslive*')
path_tomls_folder_m = sorted(glob.glob(run_path_measures))
print('---- These are the tomls for ITSLIVE ----')
print(path_tomls_folder_m)

run_name = args.sub_plot_name
toml_m = ''

if run_name == 'ALL':
    toml_m = path_tomls_folder_m[0]
else:
    for path_m in path_tomls_folder_m:
        if run_name in path_m:
            toml_m = path_m
        else:
            continue

assert toml_m != ""

params_me = conf.ConfigParser(toml_m)

n_sens = args.n_sens

# Get the n_sens
num_sens = np.arange(0, params_me.time.num_sens)
print('Get data for Time')
n_zero = num_sens[n_sens[0]]
print(n_zero)

t_sens = np.flip(np.linspace(params_me.time.run_length, 0, params_me.time.num_sens))
t_zero = np.round(t_sens[n_sens[0]])
print(t_zero)

print('Get data for Time')
n_last = num_sens[n_sens[-1]]
print(n_last)
t_last = np.round(t_sens[n_sens[-1]])
print(t_last)

out_me = utils_funcs.get_vel_ob_sens_dict(params_me)

# Getting model space coordinates to interpolate to
# same for both itslive and measures
mesh_in = fice_mesh.get_mesh(params_me)
input_data = inout.InputData(params_me)
mdl = model.model(mesh_in, input_data, params_me)
M_coords = mdl.M.tabulate_dof_coordinates()

M = mdl.M

periodic_bc = params_me.mesh.periodic_bc

out_copy_me = utils_funcs.convert_vel_sens_output_into_functions(out_me,
                                                                 M_coords,
                                                                 M,
                                                                 periodic_bc,
                                                                 mesh_in)

dQ_dU_14_ME = out_copy_me['dObsU'][n_last]
dQ_dU_14_ME = np.nan_to_num(dQ_dU_14_ME, nan=0.0)
dQ_dU_14_ME[dQ_dU_14_ME == -inf] = 0

dQ_dV_14_ME = out_copy_me['dObsV'][n_last]
dQ_dV_14_ME = np.nan_to_num(dQ_dV_14_ME, nan=0.0)
dQ_dV_14_ME[dQ_dV_14_ME == -inf] = 0

dQ_dU_3_ME = out_copy_me['dObsU'][n_zero]
dQ_dU_3_ME = np.nan_to_num(dQ_dU_3_ME, nan=0.0)
dQ_dU_3_ME[dQ_dU_3_ME == -inf] = 0

dQ_dV_3_ME = out_copy_me['dObsV'][n_zero]
dQ_dV_3_ME = np.nan_to_num(dQ_dV_3_ME, nan=0.0)
dQ_dV_3_ME[dQ_dV_3_ME == -inf] = 0

assert not any(np.isinf(dQ_dU_14_ME))
assert not any(np.isinf(dQ_dV_14_ME))
assert not any(np.isinf(dQ_dU_3_ME))
assert not any(np.isinf(dQ_dV_3_ME))

epsilon = 1e-10  # Small value to prevent log10(0)
mag_vector_14 = np.log10(np.sqrt(dQ_dU_14_ME ** 2 + dQ_dV_14_ME ** 2) + epsilon)
mag_vector_3 = np.log10(np.sqrt(dQ_dU_3_ME ** 2 + dQ_dV_3_ME ** 2) + epsilon)

assert np.all(np.isfinite(mag_vector_14)), "mag_vector_14 contains NaN or Inf!"
assert np.all(np.isfinite(mag_vector_3)), "mag_vector_3 contains NaN or Inf!"

# Read velocity file used for the inversion
x = mesh_in.coordinates()[:, 0]
y = mesh_in.coordinates()[:, 1]
t = mesh_in.cells()

trim = tri.Triangulation(x, y, t)

vel_obs = utils_funcs.find_measures_file(2013,
                                         config['input_files']['measures_cloud'])
ase_bbox = {}
for key in config['mesh_extent'].keys():
    ase_bbox[key] = np.float64(config['mesh_extent'][key])

gv = velocity.define_salem_grid_from_measures(vel_obs, ase_bbox)
proj_gnd = pyproj.Proj('EPSG:3031')
gv = salem.Grid(nxny=(520, 710), dxdy=(gv.dx, gv.dy),
                x0y0=(-1702500.0, 500.0), proj=proj_gnd)

bedmachine = config['input_files']['bedmachine']
dm = xr.open_dataset(bedmachine)
mask_bm = dm.mask
mask_bm_ase = velocity.crop_velocity_data_to_extend(mask_bm,
                                                    ase_bbox,
                                                    return_xarray=True)
new_outputf = 'vel_obs_sens_regrid_' + run_name + str(n_zero) + '_' + str(n_last) + '.nc'

if args.save_regrid_output:
    mag_vector_3_regrid, grid_x, grid_y = utils_funcs.re_grid_model_output(x, y,
                                                                           mag_vector_3,
                                                                           resolution=240,
                                                                           mask_xarray=mask_bm_ase,
                                                                           return_points=True)

    mag_vector_14_regrid = utils_funcs.re_grid_model_output(x, y,
                                                            mag_vector_14,
                                                            resolution=240,
                                                            mask_xarray=mask_bm_ase)

    dQ_dU_3_ME_regrid = utils_funcs.re_grid_model_output(x, y,
                                                         dQ_dU_3_ME,
                                                         resolution=240,
                                                         mask_xarray=mask_bm_ase)
    dQ_dV_3_ME_regrid = utils_funcs.re_grid_model_output(x, y,
                                                         dQ_dV_3_ME,
                                                         resolution=240,
                                                         mask_xarray=mask_bm_ase)

    dQ_dU_14_ME_regrid = utils_funcs.re_grid_model_output(x, y,
                                                          dQ_dU_14_ME,
                                                          resolution=240,
                                                          mask_xarray=mask_bm_ase)

    dQ_dV_14_ME_regrid = utils_funcs.re_grid_model_output(x, y,
                                                          dQ_dV_14_ME,
                                                          resolution=240,
                                                          mask_xarray=mask_bm_ase)

    with velocity.ncDataset(os.path.join(plot_path, new_outputf), 'w', format="NETCDF4") as nc:
        nc.author = 'B.M Recinos'
        nc.author_info = 'The University of Edinburgh'
        nc.proj4 = proj_gnd.srs  # full Proj4 string
        nc.epsg = 'EPSG:3031'  # optional identifier

        x_dim = nc.createDimension('x', len(grid_x[0, :]))  # latitude axis
        y_dim = nc.createDimension('y', len(grid_y[:, 0]))

        v = nc.createVariable('x', 'f4', ('x',))
        v.units = 'm'
        v.long_name = 'x coordinates'
        v[:] = grid_x[0, :]

        v = nc.createVariable('y', 'f4', ('y',))
        v.units = 'm'
        v.long_name = 'y coordinates'
        v[:] = grid_y[:, 0]

        start = n_sens[0]
        end = n_sens[-1]

        v = nc.createVariable('dQ_dU_' + str(start), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dU ' + str(start)
        v[:] = dQ_dU_3_ME_regrid

        v = nc.createVariable('dQ_dV_' + str(start), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dV ' + str(start)
        v[:] = dQ_dV_3_ME_regrid

        v = nc.createVariable('dQ_dU_' + str(end), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dU ' + str(end)
        v[:] = dQ_dU_14_ME_regrid

        v = nc.createVariable('dQ_dV_' + str(end), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dV ' + str(end)
        v[:] = dQ_dV_14_ME_regrid

        v = nc.createVariable('dQ_dM_' + str(start), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dObs magnitude as log(10)' + str(start)
        v[:] = mag_vector_3_regrid

        v = nc.createVariable('dQ_dM_' + str(end), 'f4', ('y', 'x'))
        v.units = 'm'
        v.long_name = 'dQ/dObs magnitude as log(10) ' + str(end)
        v[:] = mag_vector_14_regrid
