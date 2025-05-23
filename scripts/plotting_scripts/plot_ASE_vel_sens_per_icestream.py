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
parser.add_argument("-csv_exp_name", type=str,
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

run_path_measures = os.path.join(main_run_path, 'ase_measures*')
path_tomls_folder_m = sorted(glob.glob(run_path_measures))
print('---- These are the tomls for MEaSUREs ----')
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


rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['axes.titlesize'] = 14

cmap_sen = sns.color_palette("magma", as_cmap=True)
r = 0.8
tick_options = {'axis': 'both', 'which': 'both', 'bottom': False,
                'top': False, 'left': False, 'right': False, 'labelleft': False, 'labelbottom': False}

minv = 3.0
maxv = 6.0
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)

label_math = r'$\frac{\partial Q}{\partial\hat{p}}$ (m$^2$ . yr)'
format_ticker = [r'3$\times 10^{10}$',
                 r'4.5$\times 10^{10}$',
                 r'6$\times 10^{10}$']

shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = None

if run_name == 'THW':
    shp_f = shp.loc[[64, 138]]
    shp_sel = shp_f.dissolve(by='Regions', as_index=False)
if run_name == 'PIG':
    shp_sel = shp.loc[[63]]
if run_name == 'SPK':
    shp_f = shp.loc[[62, 137, 139]]
    shp_sel = shp_f.dissolve(by='Regions', as_index=False)
if run_name == 'ALL':
    shp_sel = shp.loc[[62, 63, 64, 137, 138, 139]]

assert shp_sel is not None

shp_sel.crs = gv.proj.crs
proj = pyproj.Proj('EPSG:3031')

gdf = gpd.read_file(config['input_files']['grounding_line'])
ase_ground = gdf[220:235]
data = ase_ground.to_crs(proj.crs).reset_index()

model_gnd_10 = gpd.read_file(config['input_files']['model_gl_10'])
model_gnd_40 = gpd.read_file(config['input_files']['model_gl_40'])

# We add the lakes and Rignot 2024 grounding line indicating seawater intrusions
shp_lake = gpd.read_file(config['input_files']['thw_lake'])
gnd_line = gpd.read_file(config['input_files']['rignot_thw'])
gnd_rig = gnd_line.to_crs(proj.crs).reset_index()

data_frame = pd.read_csv(os.path.join(plot_path,
                                      args.csv_exp_name), index_col=0)

label_lin = [r'$\Delta$ $(Q^{M}_{T}$ - $Q^{I}_{T})$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + \n' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$']

y_label_lin = r'$\Delta$ $Q_{T}$ [$m^3$]'

color_palette = sns.color_palette("deep")

img = salem.open_xr_dataset(config['input_files']['mosaic'])

# Make salem grid for antarctica map
y_mos = img.y
x_mos = img.x

dy_mos = abs(y_mos[0] - y_mos[1])
dx_mos = abs(x_mos[0] - x_mos[1])

# Pixel corner
origin_y_mos = y_mos[0] + dy_mos * 0.5
origin_x_mos = x_mos[0] - dx_mos * 0.5

gmos = salem.Grid(nxny=(len(x_mos), len(y_mos)),
                  dxdy=(dx_mos, -1*dy_mos),
                  x0y0=(origin_x_mos, origin_y_mos), proj=proj)

##################### Plotting ################################################
r = 0.9

fig = plt.figure(figsize=(12.5*r, 6*r), constrained_layout=True)

gs = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.35, width_ratios=[4, 4, 3], height_ratios=[2, 1])
ax0 = fig.add_subplot(gs[0:2, 0])

ax0.set_aspect('equal')
#ax0.set_facecolor(sns.xkcd_rgb["ocean blue"])
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                               crs=gv.proj)
c = ax0.tricontourf(x_n, y_n, t, mag_vector_3,
                    levels=levels,
                    cmap=cmap_sen, extend="both")

smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.set_shapefile(shp_sel, linewidth=2, edgecolor=sns.xkcd_rgb["grey"])

for g, geo in enumerate(shp_lake.geometry):
    smap.set_geometry(shp_lake.loc[g].geometry,
                      linewidth=0.5,
                      alpha=0.1,
                      facecolor='white', edgecolor='white',
                      crs=gv.proj)
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2,
                      color=sns.xkcd_rgb["white"],
                      alpha=0.3, crs=gv.proj)

for g, geo in enumerate(model_gnd_10.geometry):
    smap.set_geometry(model_gnd_10.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["ocean blue"],
                      crs=gv.proj)

if run_name == 'THW':
    for g, geo in enumerate(gnd_rig.geometry):
        smap.set_geometry(gnd_rig.loc[g].geometry,
                          linewidth=1.0,
                          color=sns.xkcd_rgb["black"],
                          alpha=0.3, crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter(format_ticker))
cbar.set_label(label_math, fontsize=14)
n_text = AnchoredText('year ' + str(t_zero),
                      prop=dict(size=12),
                      frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='lower left')
ax0.add_artist(at)


ax1 = fig.add_subplot(gs[0:2, 1])
ax1.set_aspect('equal')
#ax1.set_facecolor(sns.xkcd_rgb["ocean blue"])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax1.tricontourf(x_n, y_n, t, mag_vector_14,
                    levels=levels,
                    cmap=cmap_sen,
                    extend="both")

smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.set_shapefile(shp_sel, linewidth=2, edgecolor=sns.xkcd_rgb["grey"])

for g, geo in enumerate(shp_lake.geometry):
    smap.set_geometry(shp_lake.loc[g].geometry,
                      linewidth=0.5,
                      alpha=0.1,
                      facecolor='white', edgecolor='white',
                      crs=gv.proj)
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2,
                      color=sns.xkcd_rgb["white"],
                      alpha=0.3, crs=gv.proj)
for g, geo in enumerate(model_gnd_40.geometry):
    smap.set_geometry(model_gnd_40.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["ocean blue"],
                      crs=gv.proj)


if run_name == 'THW':
    for g, geo in enumerate(gnd_rig.geometry):
        smap.set_geometry(gnd_rig.loc[g].geometry,
                          linewidth=1.0,
                          color=sns.xkcd_rgb["black"],
                          alpha=0.3, crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter(format_ticker))
cbar.set_label(label_math, fontsize=14)
n_text = AnchoredText('year ' + str(t_last),
                      prop=dict(size=12),
                      frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='lower left')
ax1.add_artist(at)


ax2 = fig.add_subplot(gs[2])
# Axis limits
xlim = (0, 40)
ylim = (0, 2e12)
# Axis ticks
xticks = np.arange(0, 41, 10)
yticks = np.arange(0, 2e12, 0.5e12)

# Plot manually on each axis (replace with your own data)
delta_qoi = data_frame['delta_VAF_measures_'+run_name] - data_frame['delta_VAF_itslive_'+run_name]
dot_product = data_frame['Dot_product_'+run_name]
p1, = ax2.plot(data_frame['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
p2, = ax2.plot(data_frame['time'], dot_product, color=color_palette[1])
ax2.set_ylim(ylim)
ax2.set_yticks(yticks)
ax2.set_xlabel('Time (yrs)')
ax2.set_ylabel(y_label_lin)
ax2.set_xticks(xticks)
ax2.grid(True)
ax2.legend(handles=[p1, p2],
               labels=label_lin,
               frameon=True, fontsize=12, loc='upper left')
at = AnchoredText('c', prop=dict(size=12), frameon=True, loc='lower right')
#ax2.add_artist(at)


ax3 = fig.add_subplot(gs[5])
smap = salem.Map(gmos, countries=False)

smap.set_rgb(natural_earth='lr')
smap.set_shapefile(shp_sel, linewidth=0.2, facecolor='red', edgecolor=sns.xkcd_rgb["black"])
smap.set_lonlat_contours(xinterval=0, yinterval=0)
smap.visualize(ax=ax3, addcbar=False)
at = AnchoredText('d', prop=dict(size=9), frameon=True, loc='lower left')
ax3.add_artist(at)

path_to_plot = os.path.join(str(plot_path), str(run_name) + '.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)
