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
parser.add_argument("-sub_plot_name", type=str,
                    default="temp", help="pass filename")

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

run_path_itslive = os.path.join(main_run_path, 'ase_itslive-*')
path_tomls_folder = sorted(glob.glob(run_path_itslive))
print('---- These are the tomls for itslive ----')
print(path_tomls_folder)

run_path_measures = os.path.join(main_run_path, 'ase_measures*')
path_tomls_folder_m = sorted(glob.glob(run_path_measures))
print('---- These are the tomls for MEaSUREs ----')
print(path_tomls_folder_m)

run_name = args.sub_plot_name

toml_i = ''
toml_m = ''

for path_i, path_m in zip(path_tomls_folder, path_tomls_folder_m):
    if run_name in path_i:
        toml_i = path_i
    if run_name in path_m:
        toml_m = path_m
    else:
        continue

assert toml_i != ""
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

minv = 3.0
maxv = 6.0
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)

label_math = r'$| \frac{\partial Q}{\partial V} |$'
format_ticker = [r'3$\times 10^{10}$',
                 r'4.5$\times 10^{10}$',
                 r'6$\times 10^{10}$']

shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = None

if run_name == 'THW':
    shp_sel = shp.loc[[64, 138]]
if run_name == 'PIG':
    shp_sel = shp.loc[[63]]
if run_name == 'SPK':
    shp_sel = shp.loc[[137, 138, 139]]

assert shp_sel is not None

shp_sel.crs = gv.proj.crs

gdf = gpd.read_file(config['input_files']['grounding_line'])
ase_ground = gdf[220:235]
proj = pyproj.Proj('EPSG:3031')

data = ase_ground.to_crs(proj.crs).reset_index()

if run_name == 'THW':
    # We add the lakes
    shp_lake = gpd.read_file(config['input_files']['thw_lake'])


##################### Plotting ################################################
fig1 = plt.figure(figsize=(10 * r, 14 * r))
spec = gridspec.GridSpec(1, 2, wspace=0.35)

### dQ/dU and dQ/dV magnitude for year zero

ax0 = plt.subplot(spec[0])
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
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2,
                      color=sns.xkcd_rgb["white"],
                      alpha=0.3, crs=gv.proj)

if run_name == 'THW':
    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=0.5,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)


smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter(format_ticker))
cbar.set_label(label_math, fontsize=16)
n_text = AnchoredText('year ' + str(t_zero),
                      prop=dict(size=14),
                      frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
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
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2,
                      color=sns.xkcd_rgb["white"],
                      alpha=0.3,
                      crs=gv.proj)

if run_name == 'THW':
    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=0.5,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)

smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter(format_ticker))
cbar.set_label(label_math, fontsize=16)
n_text = AnchoredText('year ' + str(t_last),
                      prop=dict(size=14),
                      frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='upper left')
ax1.add_artist(at)

plt.tight_layout()

path_to_plot = os.path.join(str(plot_path), str(run_name) + '.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)
