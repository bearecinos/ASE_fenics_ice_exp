import sys
import os
import argparse
import numpy as np
from configobj import ConfigObj
import seaborn as sns
import geopandas as gpd
import xarray as xr
import rioxarray as rio
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter

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

shp_lake_mask = config['input_files']['lakes_mask']
ds = rio.open_rasterio(shp_lake_mask)

file_name = 'vel_obs_sens_regrid_ALL3_14.nc'
dv = xr.open_dataset(os.path.join(plot_path, file_name))

lake_x = ds.x.values
lake_y = ds.y.values
lake_y = np.flip(lake_y)

data_x = dv.x.data
data_y = dv.y.data

lake_xmin, lake_xmax = lake_x[0], lake_x[-1]
lake_ymin, lake_ymax = lake_y[0], lake_y[-1]

x_inds = np.where((data_x >= lake_xmin) & (data_x <= lake_xmax))[0]
y_inds = np.where((data_y >= lake_ymin) & (data_y <= lake_ymax))[0]

dv_sel = dv.isel(x=x_inds, y=y_inds)

# Step 1: Make the grids to interpolate
ds_y, ds_x = np.meshgrid(ds.y.values, ds.x.values, indexing='ij')
dv_y, dv_x = np.meshgrid(dv_sel.y.values, dv_sel.x.values, indexing='ij')

# Step 2: Flatten ds
points = np.column_stack((ds_x.ravel(), ds_y.ravel()))
values = ds[0].values.ravel()  # band 1

# Step 3: Interpolate to dv grid
target_points = np.column_stack((dv_x.ravel(), dv_y.ravel()))
interp_mask = griddata(points, values, target_points, method='nearest')  # preserves 0/1

# Step 4: Reshape to dv shape
interp_mask = interp_mask.reshape(dv_sel.dQ_dM_14.shape)

# Convert to xarray data array
interp_mask_xr = xr.DataArray(
    data=interp_mask,
    dims=("y", "x"),
    coords={"y": dv_sel.dQ_dM_14.y.values, "x": dv_sel.dQ_dM_14.x.values},
    name="lake_mask"
)

# Divide what is lake and no lake
lake_mask = interp_mask_xr.data == 1.0
nonlake_mask = interp_mask_xr.data == 0.0

lake_vals_40 = dv_sel.dQ_dM_14.where(lake_mask).values
nonlake_vals_40 = dv_sel.dQ_dM_14.where(nonlake_mask).values

lake_vals_10 = dv_sel.dQ_dM_3.where(lake_mask).values
nonlake_vals_10 = dv_sel.dQ_dM_3.where(nonlake_mask).values

lake_vals_nonan_40 = lake_vals_40[np.isfinite(lake_vals_40)]
nonlake_vals_nonan_40 = nonlake_vals_40[np.isfinite(nonlake_vals_40)]

lake_vals_nonan_10 = lake_vals_10[np.isfinite(lake_vals_10)]
nonlake_vals_nonan_10 = nonlake_vals_10[np.isfinite(nonlake_vals_10)]

print('Stats ------- ')
print('Maximum outside the lakes', np.max(nonlake_vals_nonan_40))
print('Maximum inside the lakes', np.max(lake_vals_nonan_40))

print('Maximum outside the lakes', np.max(nonlake_vals_nonan_10))
print('Maximum inside the lakes', np.max(lake_vals_nonan_10))

print(f"Lake mean:     {lake_vals_nonan_40.mean():.2f}")
print(f"Non-lake mean: {nonlake_vals_nonan_40.mean():.2f}")

print(f"Lake mean:     {lake_vals_nonan_10.mean():.2f}")
print(f"Non-lake mean: {nonlake_vals_nonan_10.mean():.2f}")

t_stat_40, p_val_40 = ttest_ind(lake_vals_nonan_40, nonlake_vals_nonan_40, equal_var=False)
print(f"T-test: t year 40 = {t_stat_40:.2f}, p = {p_val_40:.2e}")

t_stat_10, p_val_10 = ttest_ind(lake_vals_nonan_10, nonlake_vals_nonan_10, equal_var=False)
print(f"T-test: t year 10 = {t_stat_10:.2f}, p = {p_val_10:.2e}")

t_test_40 = f"T-test year 40: = {t_stat_40:.2f}, \n p = {p_val_40:.2e}"
t_test_10 = f"T-test year 10: = {t_stat_10:.2f}, \n p = {p_val_10:.2e}"


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Pick a specific Axes to plot on, say top-left (0, 0)
ax = axs[0]
ax.hist(nonlake_vals_nonan_10, bins=50, alpha=0.5, label="Surroundings", density=True)
ax.hist(lake_vals_nonan_10, bins=50, alpha=0.5, label="Lake", density=True)

ax.legend(loc='upper left')
ax.set_xlabel(r'log(10) $\frac{\partial Q}{\partial\hat{p}}$' + '\n'+ r'($m^2$ . yr)')
ax.set_ylabel("Density")
ax.set_title("Sensitivity distribution at year 10")
#ax.set_xlim(0, np.percentile(nonlake_vals_nonan_10, 99))
at = AnchoredText(t_test_10, prop=dict(size=10), frameon=True, loc='lower left')
ax.add_artist(at)

ax = axs[1]
ax.hist(nonlake_vals_nonan_40, bins=50, alpha=0.5, label="Surroundings", density=True)
ax.hist(lake_vals_nonan_40, bins=50, alpha=0.5, label="Lake", density=True)
#ax.set_xlim(0, np.percentile(nonlake_vals_nonan_40, 99))
ax.legend(loc='upper left')
ax.set_xlabel(r'log(10) $\frac{\partial Q}{\partial\hat{p}}$' + '\n'+ r'($m^2$ . yr)')
ax.set_ylabel("Density")
ax.set_title("Sensitivity distribution at year 40")
at = AnchoredText(t_test_40, prop=dict(size=10), frameon=True, loc='lower left')
ax.add_artist(at)

file_plot_name = 'lakes_spatial_stats.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)