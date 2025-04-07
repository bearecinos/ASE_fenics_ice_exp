import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob
import pandas as pd

import matplotlib.ticker as ticker
from matplotlib import rcParams
from configobj import ConfigObj
import argparse

# Load configuration file for more order in paths
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
parser.add_argument("-sub_plot_dir",
                    type=str,
                    default="temp",
                    help="pass sub plot directory to store csv file with output")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))
# Define main repository path
MAIN_PATH = config['main_path']

# Paths to output data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

sens_file_paths = glob.glob(os.path.join(plot_path, "*.csv"))

path_foa = 'results_FOA_linearity_test_gamma_alpha_1e4_same_gaps.csv'
path_soa = 'results_linearity_test_SOA_gamma_alpha_1e4_same_gaps.csv'

for path in sens_file_paths:
    if os.path.basename(path) == path_foa:
        path_foa = path
    if os.path.basename(path) == path_soa:
        path_soa = path

assert 'results_FOA_linearity_test_gamma_alpha_1e4_same_gaps.csv' in path_foa
dfoa = pd.read_csv(path_foa)

assert 'results_linearity_test_SOA_gamma_alpha_1e4_same_gaps.csv' in path_soa
dsoa = pd.read_csv(path_soa)

rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['font.size'] = 14

label_foa = [r'$\Delta$ ($Q^{M}_{T}$ - $Q^{I}_{T}$)',
             r'$\frac{\partial Q_{M}}{\partial \alpha_{M}} \cdot (\alpha_{M} - \alpha_{I})$ +' + '\n' +
             r'$\frac{\partial Q_{M}}{\partial \beta_{M}} \cdot (\beta_{M} - \beta_{I})$']

label_soa = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{I}_{T}$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{I})$' + ' + ' + '\n' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{I})$']

r = 0.9
color_palette = sns.color_palette("deep")

# Create figure and axes in a 2x4 grid
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17 * r,  8 * r))

# Axis limits
xlim = (0, 40)
ylim = (-0.5e12, 2.5e12)

# Axis ticks
xticks = np.arange(0, 41, 10)
yticks = np.arange(-0.5e12, 2.51e12, 0.5e12)

# Plot manually on each axis (replace with your own data)
delta_qoi = dfoa['delta_VAF_measures_all'] - dfoa['delta_VAF_itslive_all']
dot_product = dfoa['Dot_product_all']
axes[0, 0].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[0, 0].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 0].set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')
axes[0, 0].set_title('Full domain \n', fontsize=16)
axes[0, 0].set_ylim(ylim)
axes[0, 0].set_yticks(yticks)

delta_qoi = dfoa['delta_VAF_measures_PIG'] - dfoa['delta_VAF_itslive_PIG']
dot_product = dfoa['Dot_product_PIG']
axes[0, 1].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[0, 1].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 1].set_title('Pine Island basin \n', fontsize=16)
axes[0, 1].set_ylim(ylim)
axes[0, 1].set_yticks(yticks)

delta_qoi = dfoa['delta_VAF_measures_THW'] - dfoa['delta_VAF_itslive_THW']
dot_product = dfoa['Dot_product_THW']
axes[0, 2].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[0, 2].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 2].set_title('Thwaites and \n Haynes basins', fontsize=16)
axes[0, 2].set_ylim(ylim)
axes[0, 2].set_yticks(yticks)

delta_qoi = dfoa['delta_VAF_measures_SPK'] - dfoa['delta_VAF_itslive_SPK']
dot_product = dfoa['Dot_product_SPK']
p1, = axes[0, 3].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
p2, = axes[0, 3].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 3].set_title('Smith, Pope and \n Kohler basins', fontsize=16)
axes[0, 3].set_ylim(ylim)
axes[0, 3].set_yticks(yticks)

axes[0, 3].legend(handles=[p1, p2],
               labels=label_foa,
               frameon=True, fontsize=14, loc='upper left')

## SOA ################# plots ###################

delta_qoi = dsoa['delta_VAF_measures_all'] - dsoa['delta_VAF_itslive_all']
dot_product = dsoa['Dot_product_all']
axes[1, 0].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 0].plot(dsoa['time'], dot_product, color=color_palette[2])
axes[1, 0].set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')
axes[1, 0].set_ylim(ylim)
axes[1, 0].set_yticks(yticks)

delta_qoi = dsoa['delta_VAF_measures_PIG'] - dsoa['delta_VAF_itslive_PIG']
dot_product = dsoa['Dot_product_PIG']
axes[1, 1].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 1].plot(dsoa['time'], dot_product, color=color_palette[2])
axes[1, 1].set_ylim(ylim)
axes[1, 1].set_yticks(yticks)

delta_qoi = dsoa['delta_VAF_measures_THW'] - dsoa['delta_VAF_itslive_THW']
dot_product = dsoa['Dot_product_THW']
axes[1, 2].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 2].plot(dsoa['time'], dot_product, color=color_palette[2])
axes[1, 2].set_ylim(ylim)
axes[1, 2].set_yticks(yticks)

delta_qoi = dsoa['delta_VAF_measures_SPK'] - dsoa['delta_VAF_itslive_SPK']
dot_product = dsoa['Dot_product_SPK']
p1, = axes[1, 3].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
p2, = axes[1, 3].plot(dsoa['time'], dot_product, color=color_palette[2])
axes[1, 3].set_ylim(ylim)
axes[1, 3].set_yticks(yticks)

axes[1, 3].legend(handles=[p1, p2],
               labels=label_soa,
               frameon=True, fontsize=14, loc='upper left')


for row in range(2):
    for col in range(4):
        ax = axes[row, col]
        if row == 0:
            ax.set_xlabel('')
            ax.set_xticklabels('')
            ax.set_xticks(xticks)
            ax.grid(True)
        else:
            ax.set_xlabel('Years')
            ax.set_xticks(xticks)
            ax.grid(True)

file_plot_name = 'results_linearity_ALL.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
