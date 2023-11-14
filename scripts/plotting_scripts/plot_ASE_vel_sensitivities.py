import sys
import salem
import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from configobj import ConfigObj
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from fenics_ice import inout, model

import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import seaborn as sns
import argparse
from numpy import inf

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-toml_path_i",
                    type=str,
                    default="",
                    help="pass .toml file")
parser.add_argument("-toml_path_m", type=str,
                    default="",
                    help="pass .toml file")
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
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import utils_funcs, graphics, velocity

tomlf_i = args.toml_path_i
params_il = conf.ConfigParser(tomlf_i)

tomlf_m = args.toml_path_m
params_me = conf.ConfigParser(tomlf_m)

out_il = utils_funcs.get_vel_ob_sens_dict(params_il)
out_me = utils_funcs.get_vel_ob_sens_dict(params_me)

# Getting model space coordinates to interpolate to
# same for both itslive and measures
mesh_in = fice_mesh.get_mesh(params_il)
input_data = inout.InputData(params_il)
mdl = model.model(mesh_in, input_data, params_il)
M_coords = mdl.M.tabulate_dof_coordinates()

M = mdl.M

periodic_bc = params_il.mesh.periodic_bc

out_copy_il = utils_funcs.convert_vel_sens_output_into_functions(out_il,
                                                                 M_coords,
                                                                 M,
                                                                 periodic_bc,
                                                                 mesh_in)

out_copy_me = utils_funcs.convert_vel_sens_output_into_functions(out_me,
                                                                 M_coords,
                                                                 M,
                                                                 periodic_bc,
                                                                 mesh_in)

t_sens = np.flip(np.linspace(params_il.time.run_length, 0, params_il.time.num_sens))
n_sens_to_plot = args.n_sens

#Get the n_sens
num_sens = np.arange(15)
print('Get data for Time')
n_zero = num_sens[n_sens_to_plot[0]]
n_last = num_sens[n_sens_to_plot[1]]
print(n_zero)
print(n_last)

t_zero = int(np.round(t_sens[n_sens_to_plot[0]])) + 1
print(t_zero)
t_last = int(np.round(t_sens[-1]))

# ###################### Get sensitivities arrays to plot on a map ###################
dQ_dU_14_IL = np.log10(np.abs(out_copy_il['dObsU'][n_last]))
dQ_dU_14_IL[dQ_dU_14_IL == -inf] = 0

dQ_dV_14_IL = np.log10(np.abs(out_copy_il['dObsV'][n_last]))
dQ_dV_14_IL[dQ_dV_14_IL == -inf] = 0

dQ_dU_3_IL = np.log10(np.abs(out_copy_il['dObsU'][n_zero]))
dQ_dU_3_IL[dQ_dU_3_IL == -inf] = 0

dQ_dV_3_IL = np.log10(np.abs(out_copy_il['dObsV'][n_zero]))
dQ_dV_3_IL[dQ_dV_3_IL == -inf] = 0

assert not any(np.isinf(dQ_dU_14_IL))
assert not any(np.isinf(dQ_dV_14_IL))
assert not any(np.isinf(dQ_dU_3_IL))
assert not any(np.isinf(dQ_dU_3_IL))

dQ_dU_14_ME = np.log10(np.abs(out_copy_me['dObsU'][n_last]))
dQ_dU_14_ME[dQ_dU_14_ME == -inf] = 0

dQ_dV_14_ME = np.log10(np.abs(out_copy_me['dObsV'][n_last]))
dQ_dV_14_ME[dQ_dV_14_ME == -inf] = 0

dQ_dU_3_ME = np.log10(np.abs(out_copy_me['dObsU'][n_zero]))
dQ_dU_3_ME[dQ_dU_3_ME == -inf] = 0

dQ_dV_3_ME = np.log10(np.abs(out_copy_me['dObsV'][n_zero]))
dQ_dV_3_ME[dQ_dV_3_ME == -inf] = 0

assert not any(np.isinf(dQ_dU_14_ME))
assert not any(np.isinf(dQ_dV_14_ME))
assert not any(np.isinf(dQ_dU_3_ME))
assert not any(np.isinf(dQ_dV_3_ME))

x = mesh_in.coordinates()[:, 0]
y = mesh_in.coordinates()[:, 1]
t = mesh_in.cells()

trim = tri.Triangulation(x, y, t)

vel_obs = config['input_files']['measures_cloud']

ase_bbox = {}
for key in config['mesh_extent'].keys():
    ase_bbox[key] = np.float64(config['mesh_extent'][key])

gv = velocity.define_salem_grid_from_measures(vel_obs, ase_bbox)

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['axes.titlesize'] = 18

#####  Now plotting sensitivities on a map
# for a given number of n_sens     ###############################
r=0.8

tick_options = {'axis':'both','which':'both','bottom':False,
     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

#cmap_sen = sns.color_palette("viridis", as_cmap=True)
cmap_sen = sns.color_palette("magma", as_cmap=True)

minv = 0.0
maxv = 8.0
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)

fig1 = plt.figure(figsize=(10*r, 14*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.3)

### dQ/dU and dQ/dV for year zero

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
c = ax0.tricontourf(x_n, y_n, t, dQ_dU_3_IL,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen, extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax1.tricontourf(x_n, y_n, t, dQ_dV_3_IL,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='upper left')
ax1.add_artist(at)


### dQ/dU and dQ/dV for year last
ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax2.tricontourf(x_n, y_n, t, dQ_dU_14_IL,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='',
                         ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last),
                      prop=dict(size=18), frameon=True,
                      loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=12),
                  frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax3.tricontourf(x_n, y_n, t, dQ_dV_14_IL,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                        r'5e$^{10}$',
                                                        r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=12),
                  frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path,
                         'ase_vel_obs_sens_itslive_maps' +
                         str(t_zero) + '_' + str(t_last) +
                         '.png'),
            bbox_inches='tight', dpi=150)

fig2 = plt.figure(figsize=(10*r, 14*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.3)

### dQ/dU and dQ/dV for year zero

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(x, y,
                              crs=gv.proj)
c = ax0.tricontourf(x_n, y_n, t, dQ_dU_3_ME,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen, extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax1.tricontourf(x_n, y_n, t, dQ_dV_3_ME,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_zero),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='upper left')
ax1.add_artist(at)


### dQ/dU and dQ/dV for year last
ax2 = plt.subplot(spec[2])
ax2.set_aspect('equal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax2.tricontourf(x_n, y_n, t, dQ_dU_14_ME,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='',
                         ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                       r'5e$^{10}$',
                                                       r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last),
                      prop=dict(size=18), frameon=True,
                      loc='upper right')
ax2.add_artist(n_text)
at = AnchoredText('c', prop=dict(size=12),
                  frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
ax3.set_aspect('equal')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("bottom", size="5%", pad=0.5)
smap = salem.Map(gv, countries=False)
c = ax3.tricontourf(x_n, y_n, t, dQ_dV_14_ME,
                    levels = levels,
                    ticker=ticks,
                    cmap=cmap_sen,
                    extend="both")
smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)
smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks,
                         format=ticker.FixedFormatter([r'1e$^{10}$',
                                                        r'5e$^{10}$',
                                                        r'10e$^{10}$']))
cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
n_text = AnchoredText('year '+ str(t_last),
                      prop=dict(size=18),
                      frameon=True, loc='upper right')
ax3.add_artist(n_text)
at = AnchoredText('d', prop=dict(size=12),
                  frameon=True, loc='upper left')
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path,
                         'ase_vel_obs_sens_measures_maps' +
                         str(t_zero) + '_' + str(t_last) +
                         '.png'),
            bbox_inches='tight', dpi=150)