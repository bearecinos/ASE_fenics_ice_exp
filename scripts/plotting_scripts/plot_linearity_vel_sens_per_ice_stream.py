import sys
import os
import glob
import numpy as np
from configobj import ConfigObj
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from fenics_ice import config as conf
from fenics_ice import inout
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-sub_plot_dir",
                    type=str,
                    default="temp",
                    help="pass sub plot directory to store the plots")
parser.add_argument("-add_std",
                    action="store_true",
                    help="If this is specify std is plot instead of velocity data "
                         "for each vel_component in the dot product.")
parser.add_argument("-add_enveo",
                    action="store_true",
                    help="If this is specify the difference "
                         "between measures and enveo velocities "
                         "is plot additionally to itslive.")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

add_std = args.add_std
print(f"add_std is you are plotting velocities: {add_std}")

add_enveo = args.add_enveo
print(f"add_enveo is you are plotting "
      f"also enveo velocity differences: {add_enveo}")

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

if add_enveo:
    # We read velocities
    file_enveo = os.path.join(MAIN_PATH,
                              'output/02_gridded_data/ase_obs_vel_enveo-comp_enveo-cloud_2014-error-factor-1E+0.h5')
    assert os.path.exists(file_enveo), (f"File '{file_enveo}' does not exist. "
                                        f"You must produce the file via "
                                        f"scripts/prepro_stages/02_gridded_data/.")
    vel_enveo = inout.read_vel_obs(Path(file_enveo), model=None, use_cloud_point=False)

    # We merge MEaSUREs and ENVEO common data points!
    all_dfs_full_mask_enveo, df_merge_enveo = velocity.merge_measures_and_enveo_vel_obs_sens(dic_env=vel_enveo,
                                                                                             dic_me=out_me,
                                                                                             return_df_merge=True)

    all_dfs_measures_SPK_ENVEO = velocity.merge_measures_and_enveo_vel_obs_sens(dic_env=vel_enveo, dic_me=out_me_SPK)
    all_dfs_measures_PIG_ENVEO = velocity.merge_measures_and_enveo_vel_obs_sens(dic_env=vel_enveo, dic_me=out_me_PIG)
    all_dfs_measures_THW_ENVEO = velocity.merge_measures_and_enveo_vel_obs_sens(dic_env=vel_enveo, dic_me=out_me_THW)

    # We do the dot product between the derivatives and velocity differences, do it for the STD if needed.
    dot_Ume_full_mask_ENV, dot_Vme_full_mask_ENV = velocity.dot_product_per_pair_enveo(all_dfs_full_mask_enveo,
                                                                                       df_merge_enveo,
                                                                                       add_std=add_std)
    dot_Ume_SPK_mask_ENV, dot_Vme_SPK_mask_ENV = velocity.dot_product_per_pair_enveo(all_dfs_measures_SPK_ENVEO,
                                                                                     df_merge_enveo,
                                                                                     add_std=add_std)
    dot_Ume_PIG_mask_ENV, dot_Vme_PIG_mask_ENV = velocity.dot_product_per_pair_enveo(all_dfs_measures_PIG_ENVEO,
                                                                                     df_merge_enveo,
                                                                                     add_std=add_std)
    dot_Ume_THW_mask_ENV, dot_Vme_THW_mask_ENV = velocity.dot_product_per_pair_enveo(all_dfs_measures_THW_ENVEO,
                                                                                     df_merge_enveo,
                                                                                     add_std=add_std)
    dot_Ume_full_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                            t_sens,
                                            dot_Ume_full_mask_ENV)
    dot_Vme_full_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                            t_sens,
                                            dot_Vme_full_mask_ENV)

    dot_Ume_SPK_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Ume_SPK_mask_ENV)
    dot_Vme_SPK_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Vme_SPK_mask_ENV)

    dot_Ume_PIG_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Ume_PIG_mask_ENV)
    dot_Vme_PIG_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Vme_PIG_mask_ENV)

    dot_Ume_THW_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Ume_THW_mask_ENV)
    dot_Vme_THW_mask_intrp_ENV = np.interp(qoi_dict_c2['x'],
                                           t_sens,
                                           dot_Vme_THW_mask_ENV)

    d_to_add = {'Dot_product_all_enveo': dot_Vme_full_mask_intrp_ENV + dot_Ume_full_mask_intrp_ENV,
                'Dot_product_PIG_enveo': dot_Vme_PIG_mask_intrp_ENV + dot_Ume_PIG_mask_intrp_ENV,
                'Dot_product_SPK_enveo': dot_Vme_SPK_mask_intrp_ENV + dot_Ume_SPK_mask_intrp_ENV,
                'Dot_product_THW_enveo': dot_Vme_THW_mask_intrp_ENV + dot_Ume_THW_mask_intrp_ENV}

    cols_to_add = pd.DataFrame(data=d_to_add)


## Now for ITSLIVE and MEaSUREs
all_dfs_full_mask, df_merge = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il,
                                                                               dic_me=out_me,
                                                                               return_df_merge=True)

all_dfs_measures_SPK = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_SPK, dic_me=out_me_SPK)
all_dfs_measures_PIG = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_PIG, dic_me=out_me_PIG)
all_dfs_measures_THW = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_THW, dic_me=out_me_THW)

dot_Ume_full_mask, dot_Vme_full_mask = velocity.dot_product_per_pair(all_dfs_full_mask, df_merge, add_std=add_std)
dot_Ume_SPK_mask, dot_Vme_SPK_mask = velocity.dot_product_per_pair(all_dfs_measures_SPK, df_merge,add_std=add_std)
dot_Ume_PIG_mask, dot_Vme_PIG_mask = velocity.dot_product_per_pair(all_dfs_measures_PIG, df_merge,add_std=add_std)
dot_Ume_THW_mask, dot_Vme_THW_mask = velocity.dot_product_per_pair(all_dfs_measures_THW, df_merge, add_std=add_std)

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

# We save all the data to plot it later
d = {'time': qoi_dict_c2['x'],
     'delta_VAF_measures_all': qoi_dict_c2['y'],
     'delta_VAF_itslive_all': qoi_dict_c1['y'],
     'Dot_product_all': dot_Vme_full_mask_intrp + dot_Ume_full_mask_intrp,
     'delta_VAF_measures_PIG': qoi_dict_c2_PIG['y'],
     'delta_VAF_itslive_PIG': qoi_dict_c1_PIG['y'],
     'Dot_product_PIG': dot_Vme_PIG_mask_intrp + dot_Ume_PIG_mask_intrp,
     'delta_VAF_measures_SPK': qoi_dict_c2_SPK['y'],
     'delta_VAF_itslive_SPK': qoi_dict_c1_SPK['y'],
     'Dot_product_SPK': dot_Vme_SPK_mask_intrp + dot_Ume_SPK_mask_intrp,
     'delta_VAF_measures_THW': qoi_dict_c2_THW['y'],
     'delta_VAF_itslive_THW': qoi_dict_c1_THW['y'],
     'Dot_product_THW': dot_Vme_THW_mask_intrp + dot_Ume_THW_mask_intrp}

data_frame = pd.DataFrame(data=d)

if add_enveo:
    data_frame = pd.concat([data_frame, cols_to_add], axis=1)

if add_std and add_enveo:
    csv_f_name = 'results_linearity_test_with_STD_ENV.csv'
if add_enveo:
    csv_f_name = 'results_linearity_test_with_ENV.csv'
else:
    csv_f_name = 'results_linearity_test.csv'

h_obs_path = data_frame.to_csv(os.path.join(plot_path, csv_f_name))

color_palette = sns.color_palette("deep")

rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 5
rcParams['axes.titlesize'] = 5
sns.set_context('poster')

if add_std:
    label = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{I}_{T}$',
             r'$abs(\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{STD, M} - u_{STD, I}))$' + ' + ' +
             r'$abs(\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{STD, M} - v_{STD, I}))$']
if add_std and add_enveo:
    label = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{I}_{T}$',
             r'$abs(\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{STD, M} - u_{STD, E}))$' + ' + ' +
             r'$abs(\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{STD, M} - v_{STD, E}))$',
             r'$abs(\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{STD, M} - u_{STD, I}))$' + ' + ' +
             r'$abs(\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{STD, M} - v_{STD, I}))$']
if add_enveo:
    label = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{I}_{T}$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{E})$' + ' + ' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{E})$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + ' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$']
else:
    label = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{I}_{T}$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + ' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$']

y_label = r'$\Delta$ $Q_{T}$ [$m^3$]'

r=1.4

fig1 = plt.figure(figsize=(20*r, 6*r), constrained_layout=True)
spec = gridspec.GridSpec(1, 4, wspace=0.25, hspace=0.05)

### dQ/dU and dQ/dV magnitude for year zero
ax0 = plt.subplot(spec[0])

p1, = ax0.plot(qoi_dict_c2['x'],
              qoi_dict_c2['y'] - qoi_dict_c1['y'],
              linestyle='dashed', color=color_palette[3])

p3, = ax0.plot(qoi_dict_c2['x'],
              dot_Vme_full_mask_intrp + dot_Ume_full_mask_intrp,
              color=color_palette[2], label='', linewidth=3)

if add_enveo:
    p2, = ax0.plot(qoi_dict_c2['x'],
                   dot_Vme_full_mask_intrp_ENV + dot_Ume_full_mask_intrp_ENV,
                   color=color_palette[1], label='', linewidth=3)
    plt.legend(handles=[p1, p2, p3],
               labels=label,
               frameon=True, fontsize=18, bbox_to_anchor=(3.0, -0.2))
else:
    plt.legend(handles = [p1, p3],
               labels=label,
               frameon=True, fontsize=18, bbox_to_anchor=(3.0, -0.2))

ax0.set_ylabel(y_label)
ax0.set_xlabel('Time [yrs]')
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc='lower right')
ax0.add_artist(at)

### For SPK ####################################################################
ax1 = plt.subplot(spec[1])

p1, = ax1.plot(qoi_dict_c2_SPK['x'],
               qoi_dict_c2_SPK['y'] - qoi_dict_c1_SPK['y'],
              linestyle='dashed', color=color_palette[3])

p3, = ax1.plot(qoi_dict_c2_SPK['x'],
               dot_Vme_SPK_mask_intrp + dot_Ume_SPK_mask_intrp,
              color=color_palette[2], label='', linewidth=3)

if add_enveo:
    p2, = ax1.plot(qoi_dict_c2_SPK['x'],
                   dot_Vme_SPK_mask_intrp_ENV + dot_Ume_SPK_mask_intrp_ENV,
                   color=color_palette[1], label='', linewidth=3)

ax1.set_xlabel('Time [yrs]')

at = AnchoredText('b', prop=dict(size=16), frameon=True, loc='upper left')
ax1.add_artist(at)

### For PIG ####################################################################
ax2 = plt.subplot(spec[2])

p1, = ax2.plot(qoi_dict_c2_PIG['x'],
               qoi_dict_c2_PIG['y'] - qoi_dict_c1_PIG['y'],
               linestyle='dashed', color=color_palette[3])
p3, = ax2.plot(qoi_dict_c2_PIG['x'],
               dot_Vme_PIG_mask_intrp + dot_Ume_PIG_mask_intrp,
               color=color_palette[2], label='', linewidth=3)

if add_enveo:
    p2, = ax2.plot(qoi_dict_c2_PIG['x'],
                   dot_Vme_PIG_mask_intrp_ENV + dot_Ume_PIG_mask_intrp_ENV,
                   color=color_palette[1], label='', linewidth=3)

ax2.set_xlabel('Time [yrs]')

at = AnchoredText('c', prop=dict(size=16), frameon=True, loc='upper left')
ax2.add_artist(at)

### For THW ####################################################################
ax3 = plt.subplot(spec[3])

p1, = ax3.plot(qoi_dict_c2_THW['x'],
               qoi_dict_c2_THW['y'] - qoi_dict_c1_THW['y'],
               linestyle='dashed', color=color_palette[3])
p3, = ax3.plot(qoi_dict_c2_THW['x'],
               dot_Vme_THW_mask_intrp + dot_Ume_THW_mask_intrp,
               color=color_palette[2], label='', linewidth=3)

if add_enveo:
    p2, = ax3.plot(qoi_dict_c2_THW['x'],
                   dot_Vme_THW_mask_intrp_ENV + dot_Ume_THW_mask_intrp_ENV,
                   color=color_palette[1], label='', linewidth=3)

ax3.set_xlabel('Time [yrs]')
at = AnchoredText('d', prop=dict(size=16), frameon=True, loc='upper left')
ax3.add_artist(at)

ax0.set_title('Full domain', loc='right')
ax1.set_title('Smith, Pope, Kohler', loc='right')
ax2.set_title('Pine Island', loc='right')
ax3.set_title('Thwaites', loc='right')

plt.tight_layout()

if add_std:
    file_plot_name = 'linearity_test_final_with_STD.png'
if add_std and add_enveo:
    file_plot_name = 'linearity_test_final_with_STD_and_ENVEO.png'
if add_enveo:
    file_plot_name = 'linearity_test_final_with_ENVEO.png'
else:
    file_plot_name = 'linearity_test_final.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
