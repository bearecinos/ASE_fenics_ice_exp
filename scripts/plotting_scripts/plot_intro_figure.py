import sys
import salem
import pyproj
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from configobj import ConfigObj
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
import numpy as np

from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps
import matplotlib.tri as tri
import argparse

def beta_to_bglen(x):
    return x*x

parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
parser.add_argument("-toml_path",
                    type=str,
                    default="",
                    help="pass .toml file")
parser.add_argument("-sub_plot_dir",
                    type=str,
                    default="temp",
                    help="pass sub plot directory to store the plots")
parser.add_argument("-plot_name",
                    type=str,
                    default="temp",
                    help="pass plot name e.g. measures")
parser.add_argument("-plot_domain",
                    action="store_true",
                    help="If this is specify "
                         "the we make a plot too.")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Define main repository path
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

# Paths to data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import utils_funcs, graphics, velocity
from ficetools.backend import FunctionSpace, VectorFunctionSpace, Function, project

toml_config1 = args.toml_path

params_il = conf.ConfigParser(toml_config1)

mesh_in = fice_mesh.get_mesh(params_il)

print('We are using this velocity data', params_il.config_dict['obs'])

# Compute the function spaces from the Mesh
Q = FunctionSpace(mesh_in, 'Lagrange',1)
Qh = FunctionSpace(mesh_in, 'Lagrange',3)
M = FunctionSpace(mesh_in, 'DG', 0)

if not params_il.mesh.periodic_bc:
   Qp = Q
   V = VectorFunctionSpace(mesh_in,'Lagrange', 1, dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params_il, mesh_in, dim=1)
    V =  fice_mesh.get_periodic_space(params_il, mesh_in, dim=2)

# Read output data to plot
diag_il = params_il.io.diagnostics_dir
phase_suffix_il = params_il.inversion.phase_suffix

# This will be the same for both runs
phase_name = params_il.inversion.phase_name
run_name = params_il.io.run_name

exp_outdir_il = Path(diag_il) / phase_name / phase_suffix_il

file_u_il = "_".join((params_il.io.run_name+phase_suffix_il, 'U.xml'))
file_uvobs_il = "_".join((params_il.io.run_name+phase_suffix_il, 'uv_cloud.xml'))
file_alpha_il = "_".join((params_il.io.run_name+phase_suffix_il, 'alpha.xml'))
file_bglen_il = "_".join((params_il.io.run_name+phase_suffix_il, 'beta.xml'))

U_il = exp_outdir_il / file_u_il
uv_obs_il = exp_outdir_il / file_uvobs_il
alpha_il = exp_outdir_il / file_alpha_il
bglen_il = exp_outdir_il / file_bglen_il

assert U_il.is_file(), "File not found"
assert uv_obs_il.is_file(), "File not found"
assert alpha_il.is_file(), "File not found"
assert bglen_il.is_file(), "File not found"

# Define function spaces for alpha only and uv_comp
alpha_live = Function(Qp, str(alpha_il))
C2_il = project(alpha_live*alpha_live, M)

# Beta space
beta_il = Function(Qp, str(bglen_il))
bglen_live = project(beta_to_bglen(beta_il), M)

uv_live = Function(M, str(uv_obs_il))
uv_obs_live = project(uv_live, Q)


# Get mesh triangulation
x, y, t = graphics.read_fenics_ice_mesh(mesh_in)
trim = tri.Triangulation(x, y, t)

# Compute vertex values for each parameter function
# in the mesh
# Model velocities
U_itlive =  utils_funcs.compute_vertex_for_velocity_field(str(U_il), V, Q, mesh_in)

# The recovered basal traction
C2_v_il = C2_il.compute_vertex_values(mesh_in)
# Bglen
bglen_v_il = bglen_live.compute_vertex_values(mesh_in)

# Velocity observations
uv_live = uv_obs_live.compute_vertex_values(mesh_in)

# 1. The inverted value of B2; It is explicitly assumed that B2 = alpha**2
alpha_ilp = project(alpha_live, M)
alpha_v_il = alpha_ilp.compute_vertex_values(mesh_in)

beta_ilp = project(beta_il, M)
beta_v_il = beta_ilp.compute_vertex_values(mesh_in)

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

# Gathering all the shapefiles
shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = shp.loc[[62, 63, 64, 138, 137, 138, 139]]
shp_sel.crs = gv.proj.crs

# Grounding line for other icestreams
gdf = gpd.read_file(config['input_files']['grounding_line'])
ase_ground = gdf[220:235]
data = ase_ground.to_crs(proj_gnd.crs).reset_index()

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18

cmap_vel=sns.color_palette("Spectral_r", as_cmap=True)
#cmap_params_alpha = colormaps.get_cmap('YlOrBr')
cmap_params_alpha = colormaps.get_cmap('RdYlBu_r')
#cmap_params_bglen = colormaps.get_cmap('cubehelix')
cmap_params_bglen = colormaps.get_cmap('YlGnBu')

# Now plotting
r=1.2

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

fig1 = plt.figure(figsize=(10*r, 12*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.3)

ax0 = plt.subplot(spec[0])
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)

x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 800
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax0.tricontourf(x_n, y_n, t, U_itlive, levels = levels, cmap=cmap_vel, extend="both")
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_vel)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='speed [m. $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)

c = ax1.tricontourf(x_n, y_n, t, uv_live, levels = levels, cmap=cmap_vel, extend="both")
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_vel)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='speed [m. $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper right')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 0
maxv = 40
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax2.tricontourf(x_n, y_n, t, alpha_v_il, levels=levels, cmap=cmap_params_alpha, extend="both")
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
smap.set_cmap(cmap_params_alpha)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
for g, geo in enumerate(data.geometry):
    smap.set_geometry(data.loc[g].geometry,
                      linewidth=2.0,
                      color=sns.xkcd_rgb["black"], crs=gv.proj)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='Sliding parameter \n [m$^{-1/6}$ yr$^{1/6}$ Pa$^{1/2}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper right')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
minv = 400
maxv = 800
levels = np.linspace(minv,maxv,200)
ticks = np.linspace(minv,maxv,3)
c = ax3.tricontourf(x_n, y_n, t, beta_v_il, levels=levels, cmap=cmap_params_bglen, extend="both")
smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
smap.set_cmap(cmap_params_bglen)
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='Ice stiffness parameter \n [Pa$^{1/2}$. yr$^{1/6}$]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper right')
ax3.add_artist(at)

ax0.title.set_text(r'Model speed')
ax1.title.set_text(r'Observed speed MEaSUREs')
ax2.title.set_text(r'$\alpha_{MEaSUREs}$' + '\n' + r'$\gamma_{\alpha} = 1.0 \times 10^{4}$')
ax3.title.set_text(r'$\beta_{MEaSUREs}$' + '\n' + r'$\gamma_{\beta} = 4.0 \times 10^{1}$')

fig1.set_constrained_layout(True)
path_to_plot = os.path.join(str(plot_path), 'ase_inversion' + args.plot_name +  '.png')
plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)

if args.plot_domain:
    basins = sns.color_palette("Spectral_r")
    cmap_vel=sns.color_palette("Spectral_r", as_cmap=True)
    # Now plotting next figure
    r=1.2

    fig1 = plt.figure(figsize=(8*r, 7*r))#, constrained_layout=True)
    spec = gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.05)

    ax0 = plt.subplot(spec[0])
    ax0.set_aspect('equal')
    smap = salem.Map(gv, countries=False)
    x_n, y_n = smap.grid.transform(x, y,
                                  crs=gv.proj)

    smap.set_shapefile(shp_sel.loc[[63]], linewidth=2, edgecolor=basins[0], facecolor=basins[0], alpha=0.5)
    smap.set_shapefile(shp_sel.loc[[64, 138]], linewidth=2, edgecolor=basins[1], facecolor=basins[1], alpha=0.5)
    smap.set_shapefile(shp_sel.loc[[62, 137, 139]], linewidth=2, edgecolor=basins[2], facecolor=basins[2], alpha=0.5)
    smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
    smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)

    ax0.triplot(x_n, y_n, trim.triangles, '-', color='black', lw=0.2)
    smap.visualize(ax=ax0, orientation='horizontal', addcbar=True)
    at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
    ax0.add_artist(at)


    ax1 = plt.subplot(spec[1])
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    smap = salem.Map(gv, countries=False)
    x_n, y_n = smap.grid.transform(x, y,
                                  crs=gv.proj)
    minv = 0
    maxv = 800
    levels = np.linspace(minv,maxv,200)
    ticks = np.linspace(minv,maxv,3)
    c = ax1.tricontourf(x_n, y_n, t, uv_live, levels = levels, cmap=cmap_vel, extend="both")
    smap.set_vmin(minv)
    smap.set_vmax(maxv)
    smap.set_extend('both')
    smap.set_cmap(cmap_vel)

    for g, geo in enumerate(data.geometry):
        smap.set_geometry(data.loc[g].geometry,
                          linewidth=2.0,
                          color=sns.xkcd_rgb["black"], crs=gv.proj)

    smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                              linestyles='-', colors='grey', add_tick_labels=False)
    smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
    cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                             label='Velocity [m/yr]')
    at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
    ax1.add_artist(at)

    ax0.title.set_text('Model domain')
    ax1.title.set_text('MEaSUREsv2 average 1996-2016')

    fig1.set_constrained_layout(True)
    path_to_plot = os.path.join(str(plot_path), 'ase_input_data_measures.png')
    plt.savefig(path_to_plot, bbox_inches='tight', dpi=150)
