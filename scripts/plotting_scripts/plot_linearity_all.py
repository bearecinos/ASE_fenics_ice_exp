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

dfoa = pd.read_csv(sens_file_paths[1])
dsoa = pd.read_csv(sens_file_paths[2])
dsoa_syn = pd.read_csv(sens_file_paths[-1])

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

label_soa_syn = [r'$\Delta$ $Q^{M}_{T}$ - $Q^{S}_{T}$',
             r'$\frac{\partial Q_{M}}{\partial U_{M}} \cdot (u_{M} - u_{S})$' + ' + ' + '\n' +
             r'$\frac{\partial Q_{M}}{\partial V_{M}} \cdot (v_{M} - v_{S})$']

r = 0.9
color_palette = sns.color_palette("deep")

# Create figure and axes in a 2x4 grid
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16 * r, 12 * r))

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

delta_qoi = dfoa['delta_VAF_measures_PIG'] - dfoa['delta_VAF_itslive_PIG']
dot_product = dfoa['Dot_product_PIG']
axes[0, 1].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[0, 1].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 1].set_title('Pine Island basin \n', fontsize=16)

delta_qoi = dfoa['delta_VAF_measures_THW'] - dfoa['delta_VAF_itslive_THW']
dot_product = dfoa['Dot_product_THW']
axes[0, 2].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[0, 2].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 2].set_title('Thwaites and \n Haynes basins', fontsize=16)

delta_qoi = dfoa['delta_VAF_measures_SPK'] - dfoa['delta_VAF_itslive_SPK']
dot_product = dfoa['Dot_product_SPK']
p1, = axes[0, 3].plot(dfoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
p2, = axes[0, 3].plot(dfoa['time'], dot_product, color=color_palette[1])
axes[0, 3].set_title('Smith, Pope and \n Kohler basins', fontsize=16)

axes[0, 3].legend(handles=[p1, p2],
                  labels=label_foa,
                  frameon=True, fontsize=14, loc='upper left')

## SOA ################# plots ###################

delta_qoi = dsoa['delta_VAF_measures_all'] - dsoa['delta_VAF_itslive_all']
dot_product = dsoa['Dot_product_all']
axes[1, 0].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 0].plot(dsoa['time'], dot_product, color=color_palette[2])
axes[1, 0].set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')

delta_qoi = dsoa['delta_VAF_measures_PIG'] - dsoa['delta_VAF_itslive_PIG']
dot_product = dsoa['Dot_product_PIG']
axes[1, 1].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 1].plot(dsoa['time'], dot_product, color=color_palette[2])

delta_qoi = dsoa['delta_VAF_measures_THW'] - dsoa['delta_VAF_itslive_THW']
dot_product = dsoa['Dot_product_THW']
axes[1, 2].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
axes[1, 2].plot(dsoa['time'], dot_product, color=color_palette[2])

delta_qoi = dsoa['delta_VAF_measures_SPK'] - dsoa['delta_VAF_itslive_SPK']
dot_product = dsoa['Dot_product_SPK']
p1, = axes[1, 3].plot(dsoa['time'], delta_qoi, linestyle='dashed', color=color_palette[3])
p2, = axes[1, 3].plot(dsoa['time'], dot_product, color=color_palette[2])

axes[1, 3].legend(handles=[p1, p2],
                  labels=label_soa,
                  frameon=True, fontsize=14, loc='upper left')

###################### synthetic ###############################################

delta_qoi_sin = dsoa_syn['delta_VAF_measures_all'] - dsoa_syn['delta_VAF_synthetic_all']
dot_product_sin = dsoa_syn['Dot_product_all']
axes[2, 0].plot(dsoa['time'], delta_qoi_sin, linestyle='dashed', color='black')
axes[2, 0].plot(dsoa['time'], dot_product_sin, color=color_palette[4])
axes[2, 0].set_ylabel(r'$\Delta$ $Q_{T}$ [$m^3$]')

delta_qoi_sin = dsoa_syn['delta_VAF_measures_PIG'] - dsoa_syn['delta_VAF_synthetic_PIG']
dot_product_sin = dsoa_syn['Dot_product_PIG']
axes[2, 1].plot(dsoa['time'], delta_qoi_sin, linestyle='dashed', color='black')
axes[2, 1].plot(dsoa['time'], dot_product_sin, color=color_palette[4])

delta_qoi_sin = dsoa_syn['delta_VAF_measures_THW'] - dsoa_syn['delta_VAF_synthetic_THW']
dot_product_sin = dsoa_syn['Dot_product_THW']
axes[2, 2].plot(dsoa['time'], delta_qoi_sin, linestyle='dashed', color='black')
axes[2, 2].plot(dsoa['time'], dot_product_sin, color=color_palette[4])

delta_qoi_sin = dsoa_syn['delta_VAF_measures_SPK'] - dsoa_syn['delta_VAF_synthetic_SPK']
dot_product_sin = dsoa_syn['Dot_product_SPK']
p1, = axes[2, 3].plot(dsoa['time'], delta_qoi_sin, linestyle='dashed', color='black')
p2, = axes[2, 3].plot(dsoa['time'], dot_product_sin, color=color_palette[4])

axes[2, 3].legend(handles=[p1, p2],
                  labels=label_soa_syn,
                  frameon=True, fontsize=14, loc='upper left')

# Apply axis settings to all
for ax in axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True)
    ax.set_xlabel('Years')


file_plot_name = 'results_linearity_ALL.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)