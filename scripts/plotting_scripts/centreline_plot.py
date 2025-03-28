import sys
import salem
import os
import argparse
import glob
import numpy as np
from configobj import ConfigObj
import seaborn as sns
import geopandas as gpd
import xarray as xr
import pyproj

from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter

def cumulative_centerline_distance(x, y):
    """
    Compute cumulative distance along a centerline given x, y in EPSG:3031.
    Parameters:
        x, y: 1D arrays of coordinates (in meters)
    Returns:
        cum_dist: 1D array of cumulative distances (in meters)
    """
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cum_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
    return cum_dist

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

sens_file_paths = glob.glob(os.path.join(plot_path, "*_14.nc"))

centerlines = glob.glob(os.path.join(config['input_files']['centrelines'], '*.shp'))

# Reading netcdf files with re-grided model output
assert 'THW' in sens_file_paths[0]
dthw = xr.open_dataset(sens_file_paths[0])

assert 'SPK' in sens_file_paths[1]
dspk = xr.open_dataset(sens_file_paths[1])

assert 'PIG' in sens_file_paths[2]
dpig = xr.open_dataset(sens_file_paths[2])

# Reading manually delineated centrelines.
# Some of these centrelines are read from last point to
# the beginning of the calving front tos coordinates need to
# be flip
assert "pope" in centerlines[0]
pope = gpd.read_file(centerlines[0])
pope_line = pope.geometry.iloc[0]
reversed_coords = list(pope_line.coords)[::-1]
reversed_line = LineString(reversed_coords)
pope.geometry.iloc[0] = reversed_line

assert "smith" in centerlines[1]
smith = gpd.read_file(centerlines[1])
smith_line = smith.geometry.iloc[0]
reversed_coords = list(smith_line.coords)[::-1]
reversed_line = LineString(reversed_coords)
smith.geometry.iloc[0] = reversed_line

assert "kohler" in centerlines[2]
kohler = gpd.read_file(centerlines[2])
kohler_line = kohler.geometry.iloc[0]
reversed_coords = list(kohler_line.coords)[::-1]
reversed_line = LineString(reversed_coords)
kohler.geometry.iloc[0] = reversed_line

assert "pig" in centerlines[3]
pig = gpd.read_file(centerlines[3])

assert "thw" in centerlines[4]
thw = gpd.read_file(centerlines[4])

assert "hay" in centerlines[5]
hay = gpd.read_file(centerlines[5])
hay_line = hay.geometry.iloc[0]
reversed_coords = list(hay_line.coords)[::-1]
reversed_line = LineString(reversed_coords)
hay.geometry.iloc[0] = reversed_line

## Reading mask of parts of the icestream grounded and
## above floatation
model_gnd_10 = xr.open_dataset(os.path.join(plot_path, 'model_gl_10.nc'))
model_gnd_40 = xr.open_dataset(os.path.join(plot_path, 'model_gl_40.nc'))

# If more years are required by reviewers we will
# put this as argparse arguments
# Years and num_sens code for simulations
n_sens = [3, 14]
years = [10, 40]

# We interpolate model output to
# points along a centreline every 100 meters
step = 100

pig_sens_3, pig_sens_14 = utils_funcs.interp_model_output_to_centreline(dpig,
                                                                        pig,
                                                                        n_sens=n_sens,
                                                                        step=step)

thw_sens_3, thw_sens_14 = utils_funcs.interp_model_output_to_centreline(dthw,
                                                                        thw,
                                                                        n_sens=n_sens,
                                                                        step=step)

hay_sens_3, hay_sens_14 = utils_funcs.interp_model_output_to_centreline(dthw,
                                                                        hay,
                                                                        n_sens=n_sens,
                                                                        step=step)

pope_sens_3, pope_sens_14 = utils_funcs.interp_model_output_to_centreline(dspk,
                                                                          pope,
                                                                          n_sens=n_sens,
                                                                          step=step)

smith_sens_3, smith_sens_14 = utils_funcs.interp_model_output_to_centreline(dspk,
                                                                            smith,
                                                                            n_sens=n_sens,
                                                                            step=step)

kohler_sens_3, kohler_sens_14 = utils_funcs.interp_model_output_to_centreline(dspk,
                                                                              kohler,
                                                                              n_sens=n_sens,
                                                                              step=step)

## We interpolate teh model mask to the centreline

mask_pig_3, mask_pig_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                      model_gnd_40,
                                                                      pig,
                                                                      years=years,
                                                                      step=step)

mask_thw_3, mask_thw_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                      model_gnd_40,
                                                                      thw,
                                                                      years=years,
                                                                      step=step)

mask_hay_3, mask_hay_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                      model_gnd_40,
                                                                      hay,
                                                                      years=years,
                                                                      step=step)

mask_pope_3, mask_pope_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                        model_gnd_40,
                                                                        pope,
                                                                        years=years,
                                                                        step=step)

mask_smith_3, mask_smith_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                          model_gnd_40,
                                                                          smith,
                                                                          years=years,
                                                                          step=step)

mask_kohler_3, mask_kohler_14 = utils_funcs.interp_model_mask_to_centreline(model_gnd_10,
                                                                            model_gnd_40,
                                                                            kohler,
                                                                            years=years,
                                                                            step=step)

label_year10 = 'Year 9 '+ r'$\frac{\partial Q}{\partial\hat{p}}$'
label_year40 = 'Year 40 '+ r'$\frac{\partial Q}{\partial\hat{p}}$'

label_floating_10 = 'Floating part Year 9'
label_floating_40 = 'Floating part Year 40'

r = 0.9
color_palette = sns.color_palette("deep")

# Create figure and axes in a 2x4 grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14 * r, 6 * r), gridspec_kw={'hspace': 0.5})

ax0 = axes[0, 0]
pigx = pig_sens_14.x.data
pigy = pig_sens_14.y.data
x = pig_sens_14.points
distance_pig = cumulative_centerline_distance(pigx, pigy)
tick_positions = np.arange(0, len(x), 1000)
tick_labels = [np.round(distance_pig[i], decimals=0) for i in tick_positions]
pig_sens_14.plot(ax=ax0, label='')
pig_sens_3.plot(ax=ax0, label='')
ax0.fill_between(x, pig_sens_3, where=(mask_pig_3 == 0.0),  facecolor='none', edgecolor='gray', hatch='///', alpha=0.5, label='')
ax0.fill_between(x, pig_sens_14, where=(mask_pig_14 == 0.0), color='gray', alpha=0.5, label='')
ax0.set_xticks(tick_positions)
ax0.set_xticklabels(tick_labels)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc='lower left')
ax0.add_artist(at)
ax0.set_title('Pine Island', fontsize=12)

ax1 = axes[0, 1]
thwx = thw_sens_14.x.data
thwy = thw_sens_14.y.data
distance_thw = cumulative_centerline_distance(thwx, thwy)
x = thw_sens_14.points
tick_positions = np.arange(0, len(x), 1000)
tick_labels = [np.round(distance_thw[i], decimals=0) for i in tick_positions]
thw_sens_14.plot(ax=ax1, label='')
thw_sens_3.plot(ax=ax1, label='')
ax1.fill_between(x, thw_sens_3, where=(mask_thw_3 == 0.0),  facecolor='none', edgecolor='gray', hatch='///', alpha=0.5, label='')
ax1.fill_between(x, thw_sens_14, where=(mask_thw_14 == 0.0), color='gray', alpha=0.5, label='')
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc='lower left')
ax1.add_artist(at)
ax1.set_title('Thwaites', fontsize=12)

ax2 = axes[0, 2]
hayx = hay_sens_14.x.data
hayy = hay_sens_14.y.data
distance_hay = cumulative_centerline_distance(hayx, hayy)
x = hay_sens_14.points
tick_positions = np.arange(0, len(x), 300)
tick_labels = [np.round(distance_hay[i], decimals=0) for i in tick_positions]
hay_sens_14.plot(ax=ax2, label='')
hay_sens_3.plot(ax=ax2, label='')
ax2.fill_between(x, hay_sens_3, where=(mask_hay_3 == 0.0),  facecolor='none', edgecolor='gray', hatch='///', alpha=0.3, label='')
ax2.fill_between(x, hay_sens_14, where=(mask_hay_14 == 0.0), color='gray', alpha=0.5, label='')
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
at = AnchoredText('c', prop=dict(size=12), frameon=True, loc='lower left')
ax2.add_artist(at)
ax2.set_title('Haynes', fontsize=12)

ax3 = axes[1, 0]
x = pope_sens_14.points
popex = pope_sens_14.x.data
popey = pope_sens_14.y.data
distance_pope = cumulative_centerline_distance(popex, popey)
tick_positions = np.arange(0, len(x), 100)
tick_labels = [np.round(distance_pope[i], decimals=0) for i in tick_positions]
pope_sens_14.plot(ax=ax3, label='')
pope_sens_3.plot(ax=ax3, label='')
ax3.fill_between(x, pope_sens_3, where=(mask_pope_3 == 0.0),  facecolor='none', edgecolor='gray', hatch='///', alpha=0.3, label='')
ax3.fill_between(x, pope_sens_14, where=(mask_pope_14 == 0.0), color='gray', alpha=0.5, label='')
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels)
at = AnchoredText('d', prop=dict(size=12), frameon=True, loc='lower left')
ax3.add_artist(at)
ax3.set_title('Pope', fontsize=12)

ax4 = axes[1, 1]
x = smith_sens_14.points
smithx = smith_sens_14.x.data
smithy = smith_sens_14.y.data
distance_smith = cumulative_centerline_distance(smithx, smithy)
tick_positions = np.arange(0, len(x), 200)
tick_labels = [np.round(distance_smith[i], decimals=0) for i in tick_positions]
smith_sens_14.plot(ax=ax4, label='')
smith_sens_3.plot(ax=ax4, label='')
ax4.fill_between(x, smith_sens_3, where=(mask_smith_3 == 0.0),  facecolor='none', edgecolor='gray', hatch='///', alpha=0.3, label='')
ax4.fill_between(x, smith_sens_14, where=(mask_smith_14 == 0.0), color='gray', alpha=0.5, label='')
ax4.set_xticks(tick_positions)
ax4.set_xticklabels(tick_labels)
at = AnchoredText('e', prop=dict(size=12), frameon=True, loc='lower left')
ax4.add_artist(at)
ax4.set_title('Smith east', fontsize=12)

ax5 = axes[1, 2]
x = kohler_sens_14.points
kohlerx = kohler_sens_14.x.data
kohlery = kohler_sens_14.y.data
distance_kohler = cumulative_centerline_distance(kohlerx, kohlery)
tick_positions = np.arange(0, len(x), 500)
tick_labels = [np.round(distance_kohler[i], decimals=0) for i in tick_positions]
kohler_sens_14.plot(ax=ax5, label=label_year40)
kohler_sens_3.plot(ax=ax5, label=label_year10)
ax5.fill_between(x, kohler_sens_3, where=(mask_kohler_3 == 0.0),  facecolor='none',
                 edgecolor='gray', hatch='///', alpha=0.3, label=label_floating_10)
ax5.fill_between(x, kohler_sens_14, where=(mask_kohler_14 == 0.0), color='gray',
                 alpha=0.5, label=label_floating_40)
ax5.set_xticks(tick_positions)
ax5.set_xticklabels(tick_labels)
at = AnchoredText('f', prop=dict(size=12), frameon=True, loc='lower left')
ax5.add_artist(at)
ax5.set_title('Kohler west', fontsize=12)

ax5.legend(loc='lower right')

for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel(r'log(10) $\frac{\partial Q}{\partial\hat{p}}$', rotation=360, fontsize=12, labelpad=40)
        if col != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
            ax.set_yticks([])

file_plot_name = 'centreline_sensitivities.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)

# Making the map with centrelines

vel_obs = utils_funcs.find_measures_file(2013,
                                         config['input_files']['measures_cloud'])
ase_bbox = {}
for key in config['mesh_extent'].keys():
    ase_bbox[key] = np.float64(config['mesh_extent'][key])

gv = velocity.define_salem_grid_from_measures(vel_obs, ase_bbox)
proj_gnd = pyproj.Proj('EPSG:3031')
gv = salem.Grid(nxny=(520, 710), dxdy=(gv.dx, gv.dy),
                x0y0=(-1702500.0, 500.0), proj=proj_gnd)

shp = gpd.read_file(config['input_files']['ice_boundaries'])
shp_sel = shp.loc[[62, 63, 64, 137, 138, 139]]
assert shp_sel is not None
shp_sel.crs = gv.proj.crs
proj = pyproj.Proj('EPSG:3031')

file_name = 'MEaSUREs_mosaic_in_itslive_grid_croped.nc'
dv = xr.open_dataset(os.path.join(plot_path, file_name))
vv = (dv.vx**2 + dv.vy**2)**0.5

minv = 0
maxv = 2000
levels = np.linspace(minv, maxv, 200)
ticks = np.linspace(minv, maxv, 3)


fig, axes = plt.subplots(figsize=(6, 8))
smap = salem.Map(gv, countries=False)
x_n, y_n = smap.grid.transform(dv.x, dv.y,
                               crs=gv.proj)
c = axes.contourf(x_n, y_n, vv.data, levels=levels, extend="both")

smap.set_shapefile(shp_sel, linewidth=2, edgecolor=sns.xkcd_rgb["grey"])

smap.set_shapefile(pig, linewidth=2)
smap.set_shapefile(thw, linewidth=2)
smap.set_shapefile(hay, linewidth=2)
smap.set_shapefile(pope, linewidth=2)
smap.set_shapefile(smith, linewidth=2)
smap.set_shapefile(kohler, linewidth=2)

smap.set_lonlat_contours(add_ytick_labels=False, xinterval=10, yinterval=2, linewidths=1.5,
                          linestyles='-', colors='grey', add_tick_labels=False)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize(ax=axes, orientation='horizontal', addcbar=False)

file_plot_name = 'centrelines_map.png'

fig_save_path = os.path.join(plot_path, file_plot_name)
plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
