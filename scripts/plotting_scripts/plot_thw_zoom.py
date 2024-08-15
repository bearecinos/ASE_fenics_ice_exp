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
import pyproj
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

main_run_path = os.path.join(MAIN_PATH, str(args.main_path_tomls))
assert os.path.exists(main_run_path), "Provide the right path to tomls for the runs"

run_path_measures = os.path.join(main_run_path, 'ase_measures*')
path_tomls_folder_m = sorted(glob.glob(run_path_measures))
print('---- These are the tomls for MEaSUREs ----')
print(path_tomls_folder_m)

run_name = 'THW'

toml_m = ''

for path_m in path_tomls_folder_m:
    if run_name in path_m:
        toml_m = path_m
    else:
        continue

assert toml_m != ""
assert 'THW' in toml_m

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
dQ_dU_14_ME[dQ_dU_14_ME == -inf] = 0

dQ_dV_14_ME = out_copy_me['dObsV'][n_last]
dQ_dV_14_ME[dQ_dV_14_ME == -inf] = 0

dQ_dU_3_ME = out_copy_me['dObsU'][n_zero]
dQ_dU_3_ME[dQ_dU_3_ME == -inf] = 0

dQ_dV_3_ME = out_copy_me['dObsV'][n_zero]
dQ_dV_3_ME[dQ_dV_3_ME == -inf] = 0

assert not any(np.isinf(dQ_dU_14_ME))
assert not any(np.isinf(dQ_dV_14_ME))
assert not any(np.isinf(dQ_dU_3_ME))
assert not any(np.isinf(dQ_dV_3_ME))

mag_vector_14 = np.log10(np.sqrt(dQ_dU_14_ME ** 2 + dQ_dV_14_ME ** 2))
mag_vector_3 = np.log10(np.sqrt(dQ_dU_3_ME ** 2 + dQ_dV_3_ME ** 2))

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

rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.titlesize'] = 16

cmap_sen = sns.color_palette("magma", as_cmap=True)
r = 0.8
tick_options = {'axis': 'both', 'which': 'both', 'bottom': False,
                'top': False, 'left': False, 'right': False, 'labelleft': False, 'labelbottom': False}

minv = 3.5
maxv = 6.5
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)
print(ticks)

label_math = r'$| \frac{\delta Q}{\delta V} |$'
format_ticker = [r'3.5$\times 10^{10}$',
                 r'5.0$\times 10^{10}$',
                 r'6.5$\times 10^{10}$']

shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = shp.loc[[64, 138]]
shp_sel.crs = gv.proj.crs

# Reading grounding line from Rignot et al 2024
gnd_line = gpd.read_file(config['input_files']['rignot_thw'])
proj = pyproj.Proj('EPSG:3031')

gnd_rig = gnd_line.to_crs(proj.crs).reset_index()

# Rignot et al 2024 Water pressure P, as fraction of ice overburden
# Pressure contours (only > 0.8) is plotted
df_water_press = gpd.read_file(config['input_files']['rignot_P'])
idx = df_water_press.index[df_water_press['ELEV'] > 0.80]
df_press = df_water_press.loc[idx]
df_press = df_press.to_crs(proj.crs).reset_index()

# Lakes!
shp_lake = gpd.read_file(config['input_files']['thw_lake'])

# Making a new grid for plot zooming on the previous Salem grid
# Based on MEaSUREs
gv = salem.Grid(nxny=(370, 300), dxdy=(gv.dx, gv.dy),
                x0y0=(-1559357, -299373), proj=proj)

########### Plotting ##############################################

r=0.9

fig = plt.figure(figsize=(12.5*r, 6*r), constrained_layout=True)
#fig.suptitle("Thwaites and Haynes")

gs = gridspec.GridSpec(1, 2)
ax0 = fig.add_subplot(gs[0])


ax0.set_aspect('equal')
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

for g, geo in enumerate(gnd_rig.geometry):
    smap.set_geometry(gnd_rig.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["black"],
                      alpha=0.3, crs=gv.proj)

for g, geo in enumerate(df_press.geometry):
    smap.set_geometry(df_press.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["grey"],
                      alpha=0.5, crs=gv.proj)

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


ax1 = fig.add_subplot(gs[1])
ax1.set_aspect('equal')
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

for g, geo in enumerate(gnd_rig.geometry):
    smap.set_geometry(gnd_rig.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["black"],
                      alpha=0.3, crs=gv.proj)

for g, geo in enumerate(df_press.geometry):
    smap.set_geometry(df_press.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["grey"],
                      alpha=0.5, crs=gv.proj)

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

plt.tight_layout()

path_to_plot = os.path.join(str(plot_path), 'THW_zoomed_sub_hydro' + '.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)
