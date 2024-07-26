import sys
import os
import glob
import numpy as np
from configobj import ConfigObj
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from fenics_ice import config as conf
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
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
from ficetools.backend import FunctionSpace, VectorFunctionSpace, Function, project

run_path_prior = os.path.join(MAIN_PATH,
                              'scripts/run_experiments/run_paper_tomls/vaf_ice_stream_partition/ase_itslive-*')
path_tomls_folder = sorted(glob.glob(run_path_prior))

run_path_measures = os.path.join(MAIN_PATH,
                              'scripts/run_experiments/run_paper_tomls/vaf_ice_stream_partition/ase_measures*')
path_tomls_folder_m = sorted(glob.glob(run_path_measures))


tomlf_i = path_tomls_folder[0]
tomlf_m = path_tomls_folder_m[0]

toml_SPK_me = path_tomls_folder_m[2]
assert 'SPK' in toml_SPK_me
toml_PIG_me = path_tomls_folder_m[1]
assert 'PIG' in toml_PIG_me
toml_THW_me = path_tomls_folder_m[3]
assert 'THW' in toml_THW_me

toml_SPK_il = path_tomls_folder[2]
assert 'SPK' in toml_SPK_il
toml_PIG_il = path_tomls_folder[1]
assert 'PIG' in toml_PIG_il
toml_THW_il = path_tomls_folder[3]
assert 'THW' in toml_THW_il

# Reading all params configurations for Measures and Vel obs sens outputs

params_il = conf.ConfigParser(tomlf_i)
out_il = utils_funcs.get_vel_ob_sens_dict(params_il)

params_me = conf.ConfigParser(tomlf_m)
out_me = utils_funcs.get_vel_ob_sens_dict(params_me)

params_me_SPK = conf.ConfigParser(toml_SPK_me)
params_me_PIG = conf.ConfigParser(toml_PIG_me)
params_me_THW = conf.ConfigParser(toml_THW_me)

out_me_SPK = utils_funcs.get_vel_ob_sens_dict(params_me_SPK)
out_me_PIG = utils_funcs.get_vel_ob_sens_dict(params_me_PIG)
out_me_THW = utils_funcs.get_vel_ob_sens_dict(params_me_THW)

params_il_SPK = conf.ConfigParser(toml_SPK_il)
params_il_PIG = conf.ConfigParser(toml_PIG_il)
params_il_THW = conf.ConfigParser(toml_THW_il)

out_il_SPK = utils_funcs.get_vel_ob_sens_dict(params_il_SPK)
out_il_PIG = utils_funcs.get_vel_ob_sens_dict(params_il_PIG)
out_il_THW = utils_funcs.get_vel_ob_sens_dict(params_il_THW)

# Reading qoi pickle dictionaries for VAF trajectories
qoi_dict_c2 = graphics.get_data_for_sigma_path_from_toml(tomlf_m,
                                                         main_dir_path=Path(MAIN_PATH))
qoi_dict_c2_SPK = graphics.get_data_for_sigma_path_from_toml(toml_SPK_me,
                                                             main_dir_path=Path(MAIN_PATH))
qoi_dict_c2_PIG = graphics.get_data_for_sigma_path_from_toml(toml_PIG_me,
                                                             main_dir_path=Path(MAIN_PATH))
qoi_dict_c2_THW = graphics.get_data_for_sigma_path_from_toml(toml_THW_me,
                                                             main_dir_path=Path(MAIN_PATH))

# Now for Itslive
qoi_dict_c1 = graphics.get_data_for_sigma_path_from_toml(tomlf_i,
                                                         main_dir_path=Path(MAIN_PATH))
qoi_dict_c1_SPK = graphics.get_data_for_sigma_path_from_toml(toml_SPK_il,
                                                             main_dir_path=Path(MAIN_PATH))
qoi_dict_c1_PIG = graphics.get_data_for_sigma_path_from_toml(toml_PIG_il,
                                                             main_dir_path=Path(MAIN_PATH))
qoi_dict_c1_THW = graphics.get_data_for_sigma_path_from_toml(toml_THW_il,
                                                             main_dir_path=Path(MAIN_PATH))

# Get time series
t_sens = np.flip(np.linspace(params_il.time.run_length,
                             0,
                             params_il.time.num_sens))

all_dfs_full_mask, df_merge = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il,
                                                                               dic_me=out_me,
                                                                               return_df_merge=True)

all_dfs_measures_SPK = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_SPK, dic_me=out_me_SPK)
all_dfs_measures_PIG = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_PIG, dic_me=out_me_PIG)
all_dfs_measures_THW = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_THW, dic_me=out_me_THW)

dot_Ume_full_mask, dot_Vme_full_mask = velocity.dot_product_per_pair(all_dfs_full_mask, df_merge)
dot_Ume_SPK_mask, dot_Vme_SPK_mask = velocity.dot_product_per_pair(all_dfs_measures_SPK, df_merge)
dot_Ume_PIG_mask, dot_Vme_PIG_mask = velocity.dot_product_per_pair(all_dfs_measures_PIG, df_merge)
dot_Ume_THW_mask, dot_Vme_THW_mask = velocity.dot_product_per_pair(all_dfs_measures_THW, df_merge)

dot_Ume_full_mask_intrp = np.interp(qoi_dict_c2['x'],
                                    t_sens,
                                    dot_Ume_full_mask)
dot_Vme_full_mask_intrp = np.interp(qoi_dict_c2['x'],
                                    t_sens,
                                    dot_Vme_full_mask)

dot_Ume_SPK_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Ume_SPK_mask)
dot_Vme_SPK_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Vme_SPK_mask)

dot_Ume_PIG_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Ume_PIG_mask)
dot_Vme_PIG_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Vme_PIG_mask)

dot_Ume_THW_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Ume_THW_mask)
dot_Vme_THW_mask_intrp = np.interp(qoi_dict_c2['x'],
                                   t_sens,
                                   dot_Vme_THW_mask)

color_palette = sns.color_palette("deep")

rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 5
rcParams['axes.titlesize'] = 5
sns.set_context('poster')

label = [r'$\Delta$ abs($Q^{M}_{T}$ - $Q^{I}_{T}$)',
         r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + ' +
         r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$']

y_label = r'$\Delta$ $Q_{T}$ [$m^3$]'

r=1.4

fig1 = plt.figure(figsize=(20*r, 6*r), constrained_layout=True)
spec = gridspec.GridSpec(1, 4, wspace=0.25, hspace=0.05)

### dQ/dU and dQ/dV magnitude for year zero
ax0 = plt.subplot(spec[0])

p1, = ax0.plot(qoi_dict_c2['x'],
              np.abs(qoi_dict_c2['y'] - qoi_dict_c1['y']),
              linestyle='dashed', color=color_palette[3])

p3, = ax0.plot(qoi_dict_c2['x'],
              dot_Vme_full_mask_intrp + dot_Ume_full_mask_intrp,
              color=color_palette[2], label='', linewidth=3)

plt.legend(handles = [p1, p3],
           labels=label,
           frameon=True, fontsize=18)

ax0.set_ylabel(y_label)
ax0.set_xlabel('Time [yrs]')
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc='lower right')
ax0.add_artist(at)

### For SPK ####################################################################
ax1 = plt.subplot(spec[1])

p1, = ax1.plot(qoi_dict_c2_SPK['x'],
              np.abs(qoi_dict_c2_SPK['y'] - qoi_dict_c1_SPK['y']),
              linestyle='dashed', color=color_palette[3])

p3, = ax1.plot(qoi_dict_c2_SPK['x'],
               dot_Vme_SPK_mask_intrp + dot_Ume_SPK_mask_intrp,
              color=color_palette[2], label='', linewidth=3)

ax1.set_xlabel('Time [yrs]')

at = AnchoredText('b', prop=dict(size=16), frameon=True, loc='upper left')
ax1.add_artist(at)

### For PIG ####################################################################
ax2 = plt.subplot(spec[2])

p1, = ax2.plot(qoi_dict_c2_PIG['x'],
               np.abs(qoi_dict_c2_PIG['y'] - qoi_dict_c1_PIG['y']),
               linestyle='dashed', color=color_palette[3])
p3, = ax2.plot(qoi_dict_c2_PIG['x'],
               dot_Vme_PIG_mask_intrp + dot_Ume_PIG_mask_intrp,
               color=color_palette[2], label='', linewidth=3)

ax2.set_xlabel('Time [yrs]')

at = AnchoredText('c', prop=dict(size=16), frameon=True, loc='upper left')
ax2.add_artist(at)

### For THW ####################################################################
ax3 = plt.subplot(spec[3])

p1, = ax3.plot(qoi_dict_c2_THW['x'],
               np.abs(qoi_dict_c2_THW['y'] - qoi_dict_c1_THW['y']),
               linestyle='dashed', color=color_palette[3])
p3, = ax3.plot(qoi_dict_c2_THW['x'],
               dot_Vme_THW_mask_intrp + dot_Ume_THW_mask_intrp,
               color=color_palette[2], label='', linewidth=3)

ax3.set_xlabel('Time [yrs]')
at = AnchoredText('d', prop=dict(size=16), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.title.set_text('Full domain')
ax1.title.set_text('Smith, Pope, Kohler')
ax2.title.set_text('Pine Island')
ax3.title.set_text('Thwaites')

plt.tight_layout()

fig_save_path = os.path.join(plot_path, 'linearity_test_final.png')

plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)