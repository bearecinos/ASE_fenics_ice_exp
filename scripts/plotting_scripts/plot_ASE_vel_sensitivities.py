import sys
import salem
import matplotlib.pyplot as plt
import os
from pathlib import Path
from configobj import ConfigObj
from fenics_ice import config as conf
from fenics_ice import mesh as fice_mesh
from fenics_ice import inout, model

import numpy as np
from numpy import inf
import seaborn as sns
from matplotlib import rcParams, ticker
import argparse


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

qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(tomlf_i,
                                                         main_dir_path=Path(MAIN_PATH))

qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(tomlf_m,
                                                         main_dir_path=Path(MAIN_PATH))

dot_product_U_il = []
dot_product_V_il = []

for n_sen in np.arange(15):
    dq_du_dot = np.dot(np.abs(out_copy_il['dObsU'][n_sen]),
                       np.abs(out_copy_il['u_obs'] - out_copy_me['u_obs']))
    dq_dv_dot = np.dot(np.abs(out_copy_il['dObsV'][n_sen]),
                       np.abs(out_copy_il['v_obs'] - out_copy_me['v_obs']))

    dot_product_U_il.append(dq_du_dot)
    dot_product_V_il.append(dq_dv_dot)

t_sens = np.flip(np.linspace(params_il.time.run_length, 0, params_il.time.num_sens))

dot_product_U_intrp = np.interp(qoi_dict_c1['x'], t_sens, dot_product_U_il)
dot_product_V_intrp = np.interp(qoi_dict_c1['x'], t_sens, dot_product_V_il)

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

######## Plotting the linearity behaviour of the velocity sensitivities ################
color_palette = sns.color_palette("deep")
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 5
sns.set_context('poster')

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)

p1, = ax.plot(qoi_dict_c1['x'],
              np.abs(qoi_dict_c1['y']-qoi_dict_c2['y']),
              linestyle='dashed',
              color=color_palette[3])

p2, = ax.plot(qoi_dict_c1['x'],
              dot_product_U_intrp,
              color=color_palette[0],
              label='',
              linewidth=3)

p3, = ax.plot(qoi_dict_c1['x'],
              dot_product_V_intrp,
              color=color_palette[1],
              label='',
              linewidth=3)

plt.legend(handles = [p1, p2, p3],
           labels = [r'$\Delta$ abs($Q^{I}_{T}$ - $Q^{M}_{T}$)',
                     r'$\delta Q_{I} / \delta u_{I}$ . abs($u_{I}$ - $u_{M}$)',
                     r'$\delta Q_{I} / \delta v_{I}$ . abs($v_{I}$ - $v_{M}$)'],
           frameon=True, fontsize=16)

ax.set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')
ax.set_xlabel('Time [yrs]')
ax.grid(True, which="both", ls="-")
ax.axhline(y=0, color='k', linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(plot_path,
                         'ase_linearity_sens_plot.png'),
            bbox_inches='tight', dpi=150)
#
# ###################### Get sensitivities arrays to plot on a map ###################
# dQ_dU_14_IL = np.log10(np.abs(out_copy_il['dObsU'][n_last]))
# dQ_dU_14_IL[dQ_dU_14_IL == -inf] = 0
#
# dQ_dV_14_IL = np.log10(np.abs(out_copy_il['dObsV'][n_last]))
# dQ_dV_14_IL[dQ_dV_14_IL == -inf] = 0
#
# dQ_dU_3_IL = np.log10(np.abs(out_copy_il['dObsU'][n_zero]))
# dQ_dU_3_IL[dQ_dU_3_IL == -inf] = 0
#
# dQ_dV_3_IL = np.log10(np.abs(out_copy_il['dObsV'][n_zero]))
# dQ_dV_3_IL[dQ_dV_3_IL == -inf] = 0
#
# assert not any(np.isinf(dQ_dU_14_IL))
# assert not any(np.isinf(dQ_dV_14_IL))
# assert not any(np.isinf(dQ_dU_3_IL))
# assert not any(np.isinf(dQ_dU_3_IL))
#
# x = mesh_in.coordinates()[:, 0]
# y = mesh_in.coordinates()[:, 1]
# t = mesh_in.cells()
#
# trim = tri.Triangulation(x, y, t)
#
# vel_obs = config['input_files']['measures_cloud']
#
# ase_bbox = {}
# for key in config['mesh_extent'].keys():
#     ase_bbox[key] = np.float64(config['mesh_extent'][key])
#
# gv = velocity.define_salem_grid_from_measures(vel_obs, ase_bbox)
#
# rcParams['axes.labelsize'] = 18
# rcParams['xtick.labelsize'] = 18
# rcParams['ytick.labelsize'] = 18
# rcParams['axes.titlesize'] = 18
#
# #####  Now plotting sensitivities on a map
# # for a given number of n_sens     ###############################
# r=0.8
#
# tick_options = {'axis':'both','which':'both','bottom':False,
#      'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
#
# cmap_sen = sns.color_palette("viridis", as_cmap=True)
# minv = 0.0
# maxv = 8.0
# levels = np.linspace(minv, maxv, 200)
# ticks = np.linspace(minv, maxv, 3)
#
# #Take the log of the absolute value to plot with a better colorscale
# fig1 = plt.figure(figsize=(10*r, 14*r))#, constrained_layout=True)
# spec = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.3)
#
# ### dQ/dU and dQ/dV for year zero
#
# ax0 = plt.subplot(spec[0])
# ax0.set_aspect('equal')
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("bottom", size="5%", pad=0.5)
# smap = salem.Map(gv, countries=False)
# x_n, y_n = smap.grid.transform(x, y,
#                               crs=gv.proj)
# c = ax0.tricontourf(x_n, y_n, t, dQ_dU_3_IL,
#                     levels = levels,
#                     ticker=ticks,
#                     cmap=cmap_sen, extend="both")
# smap.set_vmin(minv)
# smap.set_vmax(maxv)
# smap.set_extend('both')
# smap.set_cmap(cmap_sen)
# smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
# cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
#                          label='', ticks=ticks,
#                          format=ticker.FixedFormatter([r'1e$^{10}$',
#                                                        r'5e$^{10}$',
#                                                        r'10e$^{10}$']))
# cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
# n_text = AnchoredText('year '+ str(t_zero),
#                       prop=dict(size=18),
#                       frameon=True, loc='upper right')
# ax0.add_artist(n_text)
# at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='upper left')
# ax0.add_artist(at)
#
#
# ax1 = plt.subplot(spec[1])
# ax1.set_aspect('equal')
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("bottom", size="5%", pad=0.5)
# smap = salem.Map(gv, countries=False)
# c = ax1.tricontourf(x_n, y_n, t, dQ_dV_3_IL,
#                     levels = levels,
#                     ticker=ticks,
#                     cmap=cmap_sen,
#                     extend="both")
# smap.set_vmin(minv)
# smap.set_vmax(maxv)
# smap.set_extend('both')
# smap.set_cmap(cmap_sen)
# smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
# cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
#                          label='', ticks=ticks,
#                          format=ticker.FixedFormatter([r'1e$^{10}$',
#                                                        r'5e$^{10}$',
#                                                        r'10e$^{10}$']))
# cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
# n_text = AnchoredText('year '+ str(t_zero),
#                       prop=dict(size=18),
#                       frameon=True, loc='upper right')
# ax1.add_artist(n_text)
# at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='upper left')
# ax1.add_artist(at)
#
#
# ### dQ/dU and dQ/dV for year last
# ax2 = plt.subplot(spec[2])
# ax2.set_aspect('equal')
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("bottom", size="5%", pad=0.5)
# smap = salem.Map(gv, countries=False)
# c = ax2.tricontourf(x_n, y_n, t, dQ_dU_14_IL,
#                     levels = levels,
#                     ticker=ticks,
#                     cmap=cmap_sen,
#                     extend="both")
# smap.set_vmin(minv)
# smap.set_vmax(maxv)
# smap.set_extend('both')
# smap.set_cmap(cmap_sen)
# smap.visualize(ax=ax2, orientation='horizontal', addcbar=False)
# cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
#                          label='',
#                          ticks=ticks,
#                          format=ticker.FixedFormatter([r'1e$^{10}$',
#                                                        r'5e$^{10}$',
#                                                        r'10e$^{10}$']))
# cbar.set_label(r'$\frac{\delta Q}{\delta u}$', fontsize=22)
# n_text = AnchoredText('year '+ str(t_last),
#                       prop=dict(size=18), frameon=True,
#                       loc='upper right')
# ax2.add_artist(n_text)
# at = AnchoredText('c', prop=dict(size=12),
#                   frameon=True, loc='upper left')
# ax2.add_artist(at)
#
# ax3 = plt.subplot(spec[3])
# ax3.set_aspect('equal')
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("bottom", size="5%", pad=0.5)
# smap = salem.Map(gv, countries=False)
# c = ax3.tricontourf(x_n, y_n, t, dQ_dV_14_IL,
#                     levels = levels,
#                     ticker=ticks,
#                     cmap=cmap_sen,
#                     extend="both")
# smap.set_vmin(minv)
# smap.set_vmax(maxv)
# smap.set_extend('both')
# smap.set_cmap(cmap_sen)
# smap.visualize(ax=ax3, orientation='horizontal', addcbar=False)
# cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
#                          label='', ticks=ticks,
#                          format=ticker.FixedFormatter([r'1e$^{10}$',
#                                                         r'5e$^{10}$',
#                                                         r'10e$^{10}$']))
# cbar.set_label(r'$\frac{\delta Q}{\delta v}$', fontsize=22)
# n_text = AnchoredText('year '+ str(t_last),
#                       prop=dict(size=18),
#                       frameon=True, loc='upper right')
# ax3.add_artist(n_text)
# at = AnchoredText('d', prop=dict(size=12),
#                   frameon=True, loc='upper left')
# ax3.add_artist(at)
#
# plt.tight_layout()
# plt.savefig(os.path.join(plot_path,
#                          'ase_sensitivities_velobs_itslive.png'),
#             bbox_inches='tight', dpi=150)
