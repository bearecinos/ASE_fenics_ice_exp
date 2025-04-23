import sys
import os
import argparse
import numpy as np
import glob
from configobj import ConfigObj
import pandas as pd
import geopandas as gpd
import xarray as xr
import pyproj
import salem
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

def compute_uncertainty_and_percentage_of_change(dxarray,
                                                 vxstd,
                                                 vystd,
                                                 dcsv,
                                                 yriloc,
                                                 catchment):
    """
    Computes the uncertainty and percentage of change from a reference value.
    dxarray: xarray dataarray containing dQ_dU and dQ_dV
    vxstd: standard deviation of VX vel component
    vystd: standard deviation of VY vel component
    dcsv: delta QoI (Q_year - Q_0) for a given catchment (e.g. all, THW, PIG, or SKP)

    based on delta Qoi = dQ_dU * stdvx + dQ_dV * stdvy
    % delta Qoi = delta Qoi/ reference value (Q_40 - Q_0)  * 100

    """
    if yriloc == 240:
        n_sens = '3'
    if yriloc == 960:
        n_sens = '14'

    # Compute each term of the propagation formula
    term_U = np.abs(dxarray['dQ_dU_' + n_sens].data * vxstd)
    term_V = np.abs(dxarray['dQ_dV_' + n_sens].data * vystd)

    # Combine terms with sqrt of sum of squares
    uncertainty_qoi = term_U + term_V
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
output_all_yr3 = compute_uncertainty_and_percentage_of_change(dv_sens,
                                                              vxmm_std_s,
                                                              vymm_std_s,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='all')

output_all_yr14 = compute_uncertainty_and_percentage_of_change(dv_sens,
                                                              vxmm_std_s,
                                                              vymm_std_s,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='all')

output_pig_yr3 = compute_uncertainty_and_percentage_of_change(dpig,
                                                              vxmm_std_s,
                                                              vymm_std_s,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='PIG')

output_pig_yr14 = compute_uncertainty_and_percentage_of_change(dpig,
                                                               vxmm_std_s,
                                                               vymm_std_s,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='PIG')

output_THW_yr3 = compute_uncertainty_and_percentage_of_change(dthw,
                                                              vxmm_std_s,
                                                              vymm_std_s,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='THW')

output_THW_yr14 = compute_uncertainty_and_percentage_of_change(dthw,
                                                               vxmm_std_s,
                                                               vymm_std_s,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='THW')

output_SPK_yr3 = compute_uncertainty_and_percentage_of_change(dspk,
                                                              vxmm_std_s,
                                                              vymm_std_s,
                                                              dsoa,
                                                              yriloc=240,
                                                              catchment='SPK')

output_SPK_yr14 = compute_uncertainty_and_percentage_of_change(dspk,
                                                               vxmm_std_s,
                                                               vymm_std_s,
                                                              dsoa,
                                                              yriloc=960,
                                                              catchment='SPK')
## We only plot the full catchment
vel_obs = utils_funcs.find_measures_file(2013,
                                         config['input_files']['measures_cloud'])

gv = velocity.define_salem_grid_from_measures(vel_obs, ase_bbox)
proj_gnd = pyproj.Proj('EPSG:3031')
gv = salem.Grid(nxny=(520, 710), dxdy=(gv.dx, gv.dy),
                x0y0=(-1702500.0, 500.0), proj=proj_gnd)

shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = shp.loc[[62, 63, 64, 137, 138, 139]]
assert shp_sel is not None
shp_sel.crs = gv.proj.crs
proj = pyproj.Proj('EPSG:3031')

model_gnd_40 = gpd.read_file(config['input_files']['model_gl_40'])
model_gnd_10 = gpd.read_file(config['input_files']['model_gl_10'])

r=1.2
### Making a centreline map with sensitivity output instead
minv = 0
maxv = 3e7
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)
cmap_sen = sns.color_palette("magma", as_cmap=True)

label_math = r'$Q_{VAF}$ (m$^3$)'

fig1 = plt.figure(figsize=(10*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.3)

ax0 = plt.subplot(spec[0])
ax0.set_aspect('equal')

divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(output_all_yr3['uncertainty_qoi'].x,
                               output_all_yr3['uncertainty_qoi'].y,
                               crs=gv.proj)
c = ax0.contourf(x_n, y_n, output_all_yr3['uncertainty_qoi'].data,
                 levels=levels,
                 cmap=cmap_sen,
                 extend="both")

smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)

smap.set_shapefile(shp_sel, linewidth=1.0, edgecolor=sns.xkcd_rgb["white"])

for g, geo in enumerate(model_gnd_10.geometry):
    smap.set_geometry(model_gnd_10.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["ocean blue"],
                      crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax0, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks)
cbar.set_label(label_math, fontsize=14)
n_text = AnchoredText('year ' + str(9),
                      prop=dict(size=12),
                      frameon=True, loc='upper right')
ax0.add_artist(n_text)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='lower left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
ax1.set_aspect('equal')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="5%", pad=0.5)

smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(output_all_yr14['uncertainty_qoi'].x, output_all_yr14['uncertainty_qoi'].y,
                               crs=gv.proj)
c = ax1.contourf(x_n, y_n, output_all_yr14['uncertainty_qoi'].data, levels=levels, cmap=cmap_sen, extend="both")

smap.set_vmin(minv)
smap.set_vmax(maxv)
smap.set_extend('both')
smap.set_cmap(cmap_sen)

smap.set_shapefile(shp_sel, linewidth=1.0, edgecolor=sns.xkcd_rgb["white"])

for g, geo in enumerate(model_gnd_40.geometry):
    smap.set_geometry(model_gnd_40.loc[g].geometry,
                      linewidth=1.0,
                      color=sns.xkcd_rgb["ocean blue"],
                      crs=gv.proj)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=ax1, orientation='horizontal', addcbar=False)
cbar = smap.colorbarbase(cax=cax, orientation="horizontal",
                         label='', ticks=ticks)
cbar.set_label(label_math, fontsize=14)
n_text = AnchoredText('year ' + str(40),
                      prop=dict(size=12),
                      frameon=True, loc='upper right')
ax1.add_artist(n_text)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='lower left')
ax1.add_artist(at)

file_plot_name = 'uncertainty_proxy.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)

print('Maximum value of uncertainty')
print(np.nanmax(output_all_yr14['uncertainty_qoi'].data))
print('Maximum value of percentage of change')
print(np.nanmax(output_all_yr14['delta_qoi_percentage'].data))

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

