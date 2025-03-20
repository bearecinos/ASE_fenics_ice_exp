import sys
import os
from pathlib import Path
import numpy as np
import glob
from fenics import *
from fenics_ice import config as conf
import argparse
import pandas as pd

#Plotting imports
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns

#Plotting imports
from configobj import ConfigObj

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
parser.add_argument("-path_toml_dirs_m",
                    type=str,
                    default="temp",
                    help="pass directory where measures "
                         "configurations are ran")
parser.add_argument("-path_toml_dirs_s",
                    type=str,
                    default="temp",
                    help="pass directory where itslive or "
                         "other configurations are ran with "
                         "a different velocity file")
parser.add_argument("-sub_plot_dir",
                    type=str,
                    default="temp",
                    help="pass sub plot directory to store csv file with output")
parser.add_argument("-exp_name",
                    type=str,
                    default="''",
                    help="pass experiment name suffix")
parser.add_argument("-plot",
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

# Paths to output data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

from ficetools import utils_funcs, graphics, velocity
from ficetools.backend import FunctionSpace, VectorFunctionSpace, Function, project

folder_measures = args.path_toml_dirs_m
folder_others = args.path_toml_dirs_s

run_path_other = os.path.join(MAIN_PATH,
                              folder_others)
path_tomls_folder = sorted(glob.glob(run_path_other))

run_path_measures = os.path.join(MAIN_PATH,
                                 folder_measures)
path_tomls_folder_m = sorted(glob.glob(run_path_measures))


tomlf_i = path_tomls_folder[0]
tomlf_m = path_tomls_folder_m[0]

toml_PIG_me = path_tomls_folder_m[1]
assert 'PIG' in toml_PIG_me
toml_SPK_me = path_tomls_folder_m[2]
assert 'SPK' in toml_SPK_me
toml_THW_me = path_tomls_folder_m[3]
assert 'THW' in toml_THW_me

toml_PIG_il = path_tomls_folder[1]
assert 'PIG' in toml_PIG_il
toml_SPK_il = path_tomls_folder[2]
assert 'SPK' in toml_SPK_il
toml_THW_il = path_tomls_folder[3]
assert 'THW' in toml_THW_il

# Reading all params configurations
params_il = conf.ConfigParser(tomlf_i)
params_me = conf.ConfigParser(tomlf_m)

params_me_SPK = conf.ConfigParser(toml_SPK_me)
params_me_PIG = conf.ConfigParser(toml_PIG_me)
params_me_THW = conf.ConfigParser(toml_THW_me)

params_il_SPK = conf.ConfigParser(toml_SPK_il)
params_il_PIG = conf.ConfigParser(toml_PIG_il)
params_il_THW = conf.ConfigParser(toml_THW_il)

out_il = utils_funcs.get_vel_ob_sens_dict(params_il)

out_il_SPK = utils_funcs.get_vel_ob_sens_dict(params_il_SPK)
out_il_PIG = utils_funcs.get_vel_ob_sens_dict(params_il_PIG)
out_il_THW = utils_funcs.get_vel_ob_sens_dict(params_il_THW)

out_me = utils_funcs.get_vel_ob_sens_dict(params_me)

out_me_SPK = utils_funcs.get_vel_ob_sens_dict(params_me_SPK)
out_me_PIG = utils_funcs.get_vel_ob_sens_dict(params_me_PIG)
out_me_THW = utils_funcs.get_vel_ob_sens_dict(params_me_THW)

# Now we get the rest of the information for sigma Q plot
qoi_dict_il = graphics.get_data_for_sigma_path_from_toml(tomlf_i,
                                                         main_dir_path=Path(MAIN_PATH),
                                                         add_qoi_posterior=False)

qoi_dict_m = graphics.get_data_for_sigma_path_from_toml(tomlf_m,
                                                        main_dir_path=Path(MAIN_PATH))

# Get qoi_dic per catchment
qoi_dict_il_SPK = graphics.get_data_for_sigma_path_from_toml(toml_SPK_il,
                                                             main_dir_path=Path(MAIN_PATH),
                                                             add_qoi_posterior=False)
qoi_dict_m_SPK = graphics.get_data_for_sigma_path_from_toml(toml_SPK_me,
                                                            main_dir_path=Path(MAIN_PATH),
                                                            add_qoi_posterior=False)

qoi_dict_il_PIG = graphics.get_data_for_sigma_path_from_toml(toml_PIG_il,
                                                             main_dir_path=Path(MAIN_PATH),
                                                             add_qoi_posterior=False)
qoi_dict_m_PIG = graphics.get_data_for_sigma_path_from_toml(toml_PIG_me,
                                                            main_dir_path=Path(MAIN_PATH),
                                                            add_qoi_posterior=False)

qoi_dict_il_THW = graphics.get_data_for_sigma_path_from_toml(toml_THW_il,
                                                             main_dir_path=Path(MAIN_PATH),
                                                             add_qoi_posterior=False)
qoi_dict_m_THW = graphics.get_data_for_sigma_path_from_toml(toml_THW_me,
                                                            main_dir_path=Path(MAIN_PATH),
                                                            add_qoi_posterior=False)

# Get time series
t_sens = np.flip(np.linspace(params_il.time.run_length,
                             0,
                             params_il.time.num_sens))

## Now for ITSLIVE and MEaSUREs
all_dfs_full_mask, df_merge = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il,
                                                                               dic_me=out_me,
                                                                               return_df_merge=True)

all_dfs_measures_SPK = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_SPK, dic_me=out_me_SPK)
all_dfs_measures_PIG = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_PIG, dic_me=out_me_PIG)
all_dfs_measures_THW = velocity.merge_measures_and_itslive_vel_obs_sens(dic_il=out_il_THW, dic_me=out_me_THW)

add_std = False
print(f"are you adding std instead of plotting velocities: {add_std}")

dot_Ume_full_mask, dot_Vme_full_mask = velocity.dot_product_per_pair(all_dfs_full_mask,
                                                                     df_merge,
                                                                     add_std=add_std)
dot_Ume_SPK_mask, dot_Vme_SPK_mask = velocity.dot_product_per_pair(all_dfs_measures_SPK,
                                                                   df_merge,
                                                                   add_std=add_std)
dot_Ume_PIG_mask, dot_Vme_PIG_mask = velocity.dot_product_per_pair(all_dfs_measures_PIG,
                                                                   df_merge,
                                                                   add_std=add_std)
dot_Ume_THW_mask, dot_Vme_THW_mask = velocity.dot_product_per_pair(all_dfs_measures_THW,
                                                                   df_merge,
                                                                   add_std=add_std)

dot_Ume_full_mask_intrp = np.interp(qoi_dict_m['x'],
                                    t_sens,
                                    dot_Ume_full_mask)

dot_Vme_full_mask_intrp = np.interp(qoi_dict_m['x'],
                                    t_sens,
                                    dot_Vme_full_mask)

dot_Ume_SPK_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Ume_SPK_mask)
dot_Vme_SPK_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Vme_SPK_mask)

dot_Ume_PIG_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Ume_PIG_mask)
dot_Vme_PIG_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Vme_PIG_mask)

dot_Ume_THW_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Ume_THW_mask)
dot_Vme_THW_mask_intrp = np.interp(qoi_dict_m['x'],
                                   t_sens,
                                   dot_Vme_THW_mask)

# We save all the data to plot it later
d = {
    'time': qoi_dict_m['x'],
    'delta_VAF_measures_all': qoi_dict_m['y'],
    'delta_VAF_synthetic_all': qoi_dict_il['y'],
    'Dot_product_all': dot_Vme_full_mask_intrp + dot_Ume_full_mask_intrp,
    'delta_VAF_measures_PIG': qoi_dict_m_PIG['y'],
    'delta_VAF_synthetic_PIG': qoi_dict_il_PIG['y'],
    'Dot_product_PIG': dot_Vme_PIG_mask_intrp + dot_Ume_PIG_mask_intrp,
    'delta_VAF_measures_SPK': qoi_dict_m_SPK['y'],
    'delta_VAF_synthetic_SPK': qoi_dict_il_SPK['y'],
    'Dot_product_SPK': dot_Vme_SPK_mask_intrp + dot_Ume_SPK_mask_intrp,
    'delta_VAF_measures_THW': qoi_dict_m_THW['y'],
    'delta_VAF_synthetic_THW': qoi_dict_il_THW['y'],
    'Dot_product_THW': dot_Vme_THW_mask_intrp + dot_Ume_THW_mask_intrp
     }

data_frame = pd.DataFrame(data=d)

csv_f_name = 'results_linearity_test_SOA_synthetic_'+ args.exp_name + '.csv'

h_obs_path = data_frame.to_csv(os.path.join(plot_path,
                                            csv_f_name))

if args.plot:
    ########### Now we plot things ####################################
    color_palette = sns.color_palette("deep")

    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 5
    rcParams['axes.titlesize'] = 5
    sns.set_context('poster')

    label = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{S}_{T}$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{S})$' + ' + ' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{S})$']

    y_label = r'$\Delta$ $Q_{T}$ [$m^3$]'

    r = 1.4

    fig1 = plt.figure(figsize=(20 * r, 6 * r), constrained_layout=True)
    spec = gridspec.GridSpec(1, 4, figure=fig1, wspace=0.25, hspace=0.05)

    ### dQ/dU and dQ/dV magnitude for year zero
    ax0 = fig1.add_subplot(spec[0])

    p1, = ax0.plot(qoi_dict_m['x'],
                   qoi_dict_m['y'] - qoi_dict_il['y'],
                   linestyle='dashed', color=color_palette[3])

    p3, = ax0.plot(qoi_dict_m['x'],
                   dot_Vme_full_mask_intrp + dot_Ume_full_mask_intrp,
                   color=color_palette[2], label='', linewidth=3)

    plt.legend(handles=[p1, p3],
               labels=label,
               frameon=True, fontsize=18, bbox_to_anchor=(3.0, -0.2))

    ax0.set_ylabel(y_label)
    ax0.set_xlabel('Time [yrs]')
    at = AnchoredText('a', prop=dict(size=16), frameon=True, loc='lower right')
    ax0.add_artist(at)

    ### For SPK ####################################################################
    ax1 = fig1.add_subplot(spec[1])

    p1, = ax1.plot(qoi_dict_m_SPK['x'],
                   qoi_dict_m_SPK['y'] - qoi_dict_il_SPK['y'],
                  linestyle='dashed', color=color_palette[3])

    p3, = ax1.plot(qoi_dict_m_SPK['x'],
                   dot_Vme_SPK_mask_intrp + dot_Ume_SPK_mask_intrp,
                  color=color_palette[2], label='', linewidth=3)

    ax1.set_xlabel('Time [yrs]')

    at = AnchoredText('b', prop=dict(size=16), frameon=True, loc='upper left')
    ax1.add_artist(at)

    ### For PIG ####################################################################
    ax2 = fig1.add_subplot(spec[2])

    p1, = ax2.plot(qoi_dict_m_PIG['x'],
                   qoi_dict_m_PIG['y'] - qoi_dict_il_PIG['y'],
                   linestyle='dashed', color=color_palette[3])
    p3, = ax2.plot(qoi_dict_m_PIG['x'],
                   dot_Vme_PIG_mask_intrp + dot_Ume_PIG_mask_intrp,
                   color=color_palette[2], label='', linewidth=3)

    ax2.set_xlabel('Time [yrs]')

    at = AnchoredText('c', prop=dict(size=16), frameon=True, loc='upper left')
    ax2.add_artist(at)

    ### For THW ####################################################################
    ax3 = fig1.add_subplot(spec[3])

    p1, = ax3.plot(qoi_dict_m_THW['x'],
                   qoi_dict_m_THW['y'] - qoi_dict_il_THW['y'],
                   linestyle='dashed', color=color_palette[3])
    p3, = ax3.plot(qoi_dict_m_THW['x'],
                   dot_Vme_THW_mask_intrp + dot_Ume_THW_mask_intrp,
                   color=color_palette[2], label='', linewidth=3)

    ax3.set_xlabel('Time [yrs]')
    at = AnchoredText('d', prop=dict(size=16), frameon=True, loc='upper left')
    ax3.add_artist(at)

    ax0.set_title('Full domain', loc='right')
    ax1.set_title('Smith, Pope, Kohler', loc='right')
    ax2.set_title('Pine Island', loc='right')
    ax3.set_title('Thwaites', loc='right')

    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_ylim(np.min(qoi_dict_m_PIG['y'] - qoi_dict_il_PIG['y']), 2.5e12)
        ax.grid(True)

    plt.tight_layout()

    file_plot_name = 'results_linearity_test_SOA_synthetic_'+ args.exp_name + '.png'

    fig_save_path = os.path.join(plot_path, file_plot_name)
    plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
