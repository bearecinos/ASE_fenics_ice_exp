import sys
import os
import argparse
import numpy as np
import glob
from configobj import ConfigObj
import geopandas as gpd
import xarray as xr
import pandas as pd

def compute_uncertainty_and_percentage_of_change(dxarray, vxstd2, vystd2, dcsv, yriloc, catchment):
    """
    Computes the uncertainty and percentage of change from a reference value.
    dxarray: xarray dataarray containing dQ_dU and dQ_dV
    vxstd2: standard deviation of VX vel component squared
    vystd2: standard deviation of VY vel component squared
    dcsv: delta QoI (Q_year - Q_0) for a given catchment (e.g. all, THW, PIG, or SKP)

    based on delta Qoi = sqrt( (1/2 dQ_dU * stdvx^2)^2 + (1/2 dQ_dV * stdvy^2)^2 )
    % delta Qoi = delta Qoi/ reference value (Q_40 - Q_0)  * 100

    """
    if yriloc == 240:
        n_sens = '3'
    if yriloc == 960:
        n_sens = '14'

    # Compute each term of the propagation formula
    term_U = 0.5 * dxarray['dQ_dU_' + n_sens].data * vxstd2
    term_V = 0.5 * dxarray['dQ_dV_' + n_sens].data * vystd2

    # Combine terms with sqrt of sum of squares
    uncertainty_qoi = np.sqrt(term_U ** 2 + term_V ** 2)
    reference_value = np.abs(dcsv['delta_VAF_measures_' + catchment].iloc[yriloc])

    delta_qoi = (uncertainty_qoi / reference_value) * 100

    dict_output = {'uncertainty_qoi': uncertainty_qoi,
                   'reference_qoi': reference_value,
                   'delta_qoi_percentage': delta_qoi}

    return dict_output


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
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

from ficetools import utils_funcs, velocity

# Paths to output data
sub_plot_dir = args.sub_plot_dir
plot_path = os.path.join(MAIN_PATH, 'plots/'+ sub_plot_dir)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

dmm = xr.open_dataset(config['input_files']['measures_comp_interp'])
ase_bbox = {}
for key in config['mesh_extent'].keys():
    ase_bbox[key] = np.float64(config['mesh_extent'][key])

vxmm = dmm.vx
vymm = dmm.vy
std_vxmm = dmm.std_vx
std_vymm = dmm.std_vy

# Crop velocity data to the ase Glacier extend
vxmm_s, xmm_s, ymm_s = velocity.crop_velocity_data_to_extend(vxmm,
                                                              ase_bbox,
                                                              return_coords=True)

vymm_s = velocity.crop_velocity_data_to_extend(vymm, ase_bbox)
vxmm_std_s = velocity.crop_velocity_data_to_extend(std_vxmm, ase_bbox, return_xarray=True)
vymm_std_s = velocity.crop_velocity_data_to_extend(std_vymm, ase_bbox, return_xarray=True)

# Flip STD arrays so  they can be multiply by dQ/dobs
vxmm_std_s = vxmm_std_s.isel(y=slice(None, None, -1))
vymm_std_s = vymm_std_s.isel(y=slice(None, None, -1))

# If more years are required by reviewers we will
# put this as argparse arguments
# Years and num_sens code for simulations
n_sens = [3, 14]
years = [10, 40]

# Reading more shapefiles and netcdf needed
file_name_all = 'vel_obs_sens_regrid_ALL3_14.nc'
dv_sens = xr.open_dataset(os.path.join(plot_path, file_name_all))

sens_file_paths = glob.glob(os.path.join(plot_path, "*_14.nc"))

assert 'THW' in sens_file_paths[0]
dthw = xr.open_dataset(sens_file_paths[0])

assert 'SPK' in sens_file_paths[1]
dspk = xr.open_dataset(sens_file_paths[1])

assert 'PIG' in sens_file_paths[2]
dpig = xr.open_dataset(sens_file_paths[2])

sens_file_paths = glob.glob(os.path.join(plot_path, "*.csv"))

path_soa = 'results_linearity_test_SOA_gamma_alpha_1e4_same_gaps.csv'

for path in sens_file_paths:
    if os.path.basename(path) == path_soa:
        path_soa = path

assert 'results_linearity_test_SOA_gamma_alpha_1e4_same_gaps.csv' in path_soa
dsoa = pd.read_csv(path_soa)

# Compute squared velocity uncertainties
vx_std_sq = vxmm_std_s ** 2
vy_std_sq = vymm_std_s ** 2

output_all_yr3 = compute_uncertainty_and_percentage_of_change(dv_sens,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='all')

output_all_yr14 = compute_uncertainty_and_percentage_of_change(dv_sens,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='all')

output_pig_yr3 = compute_uncertainty_and_percentage_of_change(dpig,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='PIG')

output_pig_yr14 = compute_uncertainty_and_percentage_of_change(dpig,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='PIG')

output_THW_yr3 = compute_uncertainty_and_percentage_of_change(dthw,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='THW')

output_THW_yr14 = compute_uncertainty_and_percentage_of_change(dthw,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='THW')

output_SPK_yr3 = compute_uncertainty_and_percentage_of_change(dspk,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='SPK')

output_SPK_yr14 = compute_uncertainty_and_percentage_of_change(dspk,
                                                              vx_std_sq,
                                                              vy_std_sq,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='SPK')

print('Table for supplement')
print('-------------------------------------')
print('Catchment | Year 40 | Year 9 |')
print('All  | ' +
      str(output_all_yr14['delta_qoi_percentage'].sum(skipna=True)) +
      ' | ' +
      str(output_all_yr3['delta_qoi_percentage'].sum(skipna=True)) +
      ' | '
      )
print('PIG  | ' +
      str(output_pig_yr14['delta_qoi_percentage'].sum(skipna=True)) +
      ' | ' +
      str(output_pig_yr3['delta_qoi_percentage'].sum(skipna=True)) +
      ' | '
      )
print('THW  | ' +
      str(output_THW_yr14['delta_qoi_percentage'].sum(skipna=True)) +
      ' | ' +
      str(output_THW_yr3['delta_qoi_percentage'].sum(skipna=True)) +
      ' | '
      )
print('SPK  | ' +
      str(output_SPK_yr14['delta_qoi_percentage'].sum(skipna=True)) +
      ' | ' +
      str(output_SPK_yr3['delta_qoi_percentage'].sum(skipna=True)) +
      ' | '
      )

