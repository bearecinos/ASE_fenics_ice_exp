import sys
import salem
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from pathlib import Path
from configobj import ConfigObj
from fenics_ice import config as conf

from matplotlib import rcParams
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

qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(tomlf_i,
                                                         main_dir_path=Path(MAIN_PATH))

qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(tomlf_m,
                                                         main_dir_path=Path(MAIN_PATH))

t_sens = np.flip(np.linspace(params_il.time.run_length,
                             0,
                             params_il.time.num_sens))

xm, ym = np.split(out_me['uv_obs_pts'], 2, axis=1)
xi, yi = np.split(out_il['uv_obs_pts'], 2, axis=1)

d_il = {'xi': xi.ravel(),
        'yi':yi.ravel(),
        'u_obs': out_il['u_obs'],
        'v_obs': out_il['v_obs']}

df_il = pd.DataFrame(data=d_il)

d_me = {'xm': xm.ravel(),
        'ym':ym.ravel(),
        'u_obs': out_me['u_obs'],
        'v_obs': out_me['v_obs']}

df_me = pd.DataFrame(data=d_me)

df_merge = pd.merge(df_il,
                    df_me,
                    how='inner',
                    left_on=['xi','yi'],
                    right_on = ['xm','ym'])

# we will save all dataframes on a dict
all_dfs = defaultdict(list)

for n_sen in np.arange(15):
    dict_me = {'xm': xm.ravel(),
               'ym': ym.ravel(),
               'dObsU': out_me['dObsU'][n_sen],
               'dObsV': out_me['dObsV'][n_sen]}

    df_measures = pd.DataFrame(data=dict_me)

    dict_il = {'xi': xi.ravel(),
               'yi': yi.ravel(),
               'dObsU': out_il['dObsU'][n_sen],
               'dObsV': out_il['dObsV'][n_sen]}

    df_itslive = pd.DataFrame(data=dict_il)

    df_merge_dqdo = pd.merge(df_itslive,
                             df_measures,
                             how='inner',
                             left_on=['xi', 'yi'],
                             right_on=['xm', 'ym'])

    assert df_merge_dqdo.shape == df_merge.shape

    all_dfs[n_sen] = df_merge_dqdo

dot_product_U_il = []
dot_product_V_il = []
dot_product_U_me = []
dot_product_V_me = []

for n_sen in np.arange(15):
    dq_du_il = all_dfs[n_sen]['dObsU_x']
    dq_dv_il = all_dfs[n_sen]['dObsV_x']

    dq_du_me = all_dfs[n_sen]['dObsU_y']
    dq_dv_me = all_dfs[n_sen]['dObsV_y']

    u_il = df_merge['u_obs_x']
    v_il = df_merge['v_obs_x']

    u_me = df_merge['u_obs_y']
    v_me = df_merge['v_obs_y']

    dq_du_dot_il = np.dot(dq_du_il, u_il - u_me)
    dq_dv_dot_il = np.dot(dq_dv_il, v_il - v_me)

    dq_du_dot_me = np.dot(dq_du_me, u_me - u_il)
    dq_dv_dot_me = np.dot(dq_dv_me, v_me - v_il)

    dot_product_U_il.append(dq_du_dot_il)
    dot_product_V_il.append(dq_dv_dot_il)

    dot_product_U_me.append(dq_du_dot_me)
    dot_product_V_me.append(dq_dv_dot_me)

dot_product_U_intrp = np.interp(qoi_dict_c1['x'],
                                t_sens,
                                dot_product_U_il)
dot_product_V_intrp = np.interp(qoi_dict_c1['x'],
                                t_sens,
                                dot_product_V_il)

dot_product_U_intrp_me = np.interp(qoi_dict_c1['x'],
                                   t_sens,
                                   dot_product_U_me)
dot_product_V_intrp_me = np.interp(qoi_dict_c1['x'],
                                   t_sens,
                                   dot_product_V_me)

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
              linestyle='dashed', color=color_palette[3])
p3, = ax.plot(qoi_dict_c1['x'],
              dot_product_V_intrp + dot_product_U_intrp,
              color=color_palette[2], label='', linewidth=3)

p2, = ax.plot(qoi_dict_c1['x'], dot_product_U_intrp,
              color=color_palette[0], label='', linewidth=3)
p4, = ax.plot(qoi_dict_c1['x'], dot_product_V_intrp,
              color=color_palette[1], label='', linewidth=3)

plt.legend(handles = [p1, p3, p2, p4],
           labels = [r'$\Delta$ abs($Q^{I}_{T}$ - $Q^{M}_{T}$)',
                     r'$\frac{\partial Q_{I}}{\partial U_{I}} \cdot (u_{I} - u_{M})$' + ' + '+
                     r'$\frac{\partial Q_{I}}{\partial V_{I}} \cdot (v_{I} - v_{M})$',
                     r'$\frac{\partial Q_{I}}{\partial U_{I}} \cdot (u_{I} - u_{M})$',
                     r'$\frac{\partial Q_{I}}{\partial V_{I}} \cdot (v_{I} - v_{M})$'],
           frameon=True, fontsize=16)

ax.set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')
ax.set_xlabel('Time [yrs]')

ax.grid(True, which="both", ls="-")
ax.axhline(y=0, color='k', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(plot_path,
                         'ase_vel_obs_linearity_itslive.png'),
            bbox_inches='tight', dpi=150)

### Plot the same but for MEaSUREs derivatives
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)


p1, = ax.plot(qoi_dict_c1['x'],
              np.abs(qoi_dict_c2['y'] - qoi_dict_c1['y']),
              linestyle='dashed', color=color_palette[3])
p3, = ax.plot(qoi_dict_c1['x'],
              dot_product_V_intrp_me + dot_product_U_intrp_me,
              color=color_palette[2], label='', linewidth=3)

p2, = ax.plot(qoi_dict_c1['x'], dot_product_U_intrp_me,
              color=color_palette[0], label='', linewidth=3)
p4, = ax.plot(qoi_dict_c1['x'], dot_product_V_intrp_me,
              color=color_palette[1], label='', linewidth=3)

plt.legend(handles = [p1, p3, p2, p4],
           labels = [r'$\Delta$ abs($Q^{M}_{T}$ - $Q^{I}_{T}$)',
                     r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + '+
                     r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$',
                     r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$',
                     r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$'],
           frameon=True, fontsize=16)

ax.set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')
ax.set_xlabel('Time [yrs]')

ax.grid(True, which="both", ls="-")
ax.axhline(y=0, color='k', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(plot_path,
                         'ase_vel_obs_linearity_mesures.png'),
            bbox_inches='tight', dpi=150)
