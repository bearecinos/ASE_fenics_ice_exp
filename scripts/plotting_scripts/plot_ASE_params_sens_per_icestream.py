import sys
import salem
import os
import argparse
import glob
import numpy as np
from configobj import ConfigObj
import seaborn as sns
import geopandas as gpd
import pyproj
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh

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

dq_dalpha_t0, dq_dbeta_t0 = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_me,
                                                                               n_sen=n_zero,
                                                                               mult_mmatrix=True)

dq_dalpha_t40, dq_dbeta_t40 = utils_funcs.compute_vertex_for_dQ_dalpha_component(params_me,
                                                                                 n_sen=n_last,
                                                                                 mult_mmatrix=True)

# We get rid of negative sensitivities
dq_dalpha_t0[dq_dalpha_t0 < 0] = 0
dq_dbeta_t0[dq_dbeta_t0 < 0] = 0

dq_dalpha_t40[dq_dalpha_t40 < 0] = 0
dq_dbeta_t40[dq_dbeta_t40 < 0] = 0

perc_dq_dalpha_t40 = np.percentile(dq_dalpha_t40, 90)

dq_dalpha_t40_norm = utils_funcs.normalize(dq_dalpha_t40, percentile=perc_dq_dalpha_t40)
dq_dalpha_t0_norm = utils_funcs.normalize(dq_dalpha_t0, percentile=perc_dq_dalpha_t40)


perc_dq_dbeta_t40 = np.percentile(dq_dbeta_t40, 90)

dq_dbeta_t40_norm = utils_funcs.normalize(dq_dbeta_t40, percentile=perc_dq_dbeta_t40)
dq_dbeta_t0_norm = utils_funcs.normalize(dq_dbeta_t0, percentile=perc_dq_dbeta_t40)

# Getting model space coordinates to interpolate to
# same for both itslive and measures
mesh_in = fice_mesh.get_mesh(params_me)

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

# Right Projection! TODO: We need to add this to every plot!
proj_gnd = pyproj.Proj('EPSG:3031')
gv = salem.Grid(nxny=(520, 710), dxdy=(gv.dx, gv.dy),
                x0y0=(-1702500.0, 500.0), proj=proj_gnd)

# Plot  details
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18
cmap_sen = sns.color_palette("magma", as_cmap=True)

r=1.2

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

label_alpha = r'$\frac{\delta Q_{VAF}}{\delta \alpha^{2}}$'

label_beta = r'$\frac{\delta Q}{\delta \beta^{2}}$'

minv = 0.0
maxv = 5.0
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)

# Gathering all the shapefiles
shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = None

if run_name == 'THW':
    shp_sel = shp.loc[[64, 138]]
if run_name == 'PIG':
    shp_sel = shp.loc[[63]]
if run_name == 'SPK':
    shp_sel = shp.loc[[137, 138, 139]]
if run_name == 'ALL':
    shp_sel = shp.loc[[63, 64, 138, 137, 138, 139]]

assert shp_sel is not None
shp_sel.crs = gv.proj.crs

# Grounding line for other icestreams
gdf = gpd.read_file(config['input_files']['grounding_line'])
ase_ground = gdf[220:235]
data = ase_ground.to_crs(proj_gnd.crs).reset_index()

if run_name == 'THW' or run_name == 'ALL':
    # We add the lakes
    shp_lake = gpd.read_file(config['input_files']['thw_lake'])
    gnd_line = gpd.read_file(config['input_files']['rignot_thw'])
    gnd_rig = gnd_line.to_crs(proj_gnd.crs).reset_index()

## ALPHA PLOT #####################################################
fig1 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.set_facecolor(sns.color_palette("Paired")[0])
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y, crs=gv.proj)
c = ax0.tricontourf(x_n, y_n, t, dq_dalpha_t0_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
#ax0.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
smap.set_shapefile(shp_sel, linewidth=2, edgecolor=sns.xkcd_rgb["grey"])
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2,
                      color=sns.xkcd_rgb["white"],
                      alpha=0.3, crs=gv.proj)

if run_name == 'THW':
    for g, geo in enumerate(gnd_rig.geometry):
        smap.set_geometry(gnd_rig.loc[g].geometry,
                          linewidth=1.0,
                          color=sns.xkcd_rgb["orange"],
                          alpha=0.3, crs=gv.proj)

    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=1.0,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)
if run_name == 'ALL':
    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=1.0,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)

cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(label_alpha, fontsize=22)
n_text = AnchoredText('year '+ str(t_zero), prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

### Plotting year 40

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.set_facecolor(sns.color_palette("Paired")[0])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
c = ax1.tricontourf(x_n, y_n, t, dq_dalpha_t40_norm, levels = levels, cmap=cmap_sen, extend="both")
#ax1.triplot(x_n, y_n, trim.triangles, '-', color='grey', lw=0.2, alpha=0.5)
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
    for g, geo in enumerate(gnd_rig.geometry):
        smap.set_geometry(gnd_rig.loc[g].geometry,
                          linewidth=1.0,
                          color=sns.xkcd_rgb["orange"],
                          alpha=0.3, crs=gv.proj)

    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=1.0,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)

if run_name == 'ALL':
    for g, geo in enumerate(shp_lake.geometry):
        smap.set_geometry(shp_lake.loc[g].geometry,
                          linewidth=1.0,
                          alpha=0.1,
                          facecolor='white', edgecolor='white',
                          crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)

cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(label_alpha, fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

plt.tight_layout()

path_to_plot = os.path.join(str(plot_path), str(run_name) + '_alpha_params_sens_.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)

######## Plotting Beta ###############################################

fig1 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
ax0.set_facecolor(sns.color_palette("Paired")[0])
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y, crs=gv.proj)
c = ax0.tricontourf(x_n, y_n, t, dq_dbeta_t0_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.set_shapefile(shp_sel, linewidth=1.0, edgecolor=sns.xkcd_rgb["grey"])
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)

cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(label_beta, fontsize=22)
n_text = AnchoredText('year '+ str(t_zero), prop=dict(size=18), frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
ax1.set_facecolor(sns.color_palette("Paired")[0])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
c = ax1.tricontourf(x_n, y_n, t, dq_dbeta_t40_norm, levels = levels, cmap=cmap_sen, extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.set_shapefile(shp_sel, linewidth=1.0, edgecolor=sns.xkcd_rgb["grey"])
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)

cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='')
cbar.set_label(label_beta, fontsize=22)
n_text = AnchoredText('year '+ str(t_last), prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

plt.tight_layout()
path_to_plot = os.path.join(str(plot_path), str(run_name) + '_beta_params_sens_.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)