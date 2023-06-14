"""
Reads data from a bespoke .mat file of positions in order to construct a mask
of Dotson ice shelf (and possible new shelf) to set differing melt parameters here

"""
import os
import sys
import numpy as np
import h5py
from scipy import io 
from scipy.interpolate import RegularGridInterpolator
import argparse
from configobj import ConfigObj
from matplotlib.path import Path
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))


# IMPORTANT: here is where we set our parameters

melt_depth_therm_PIG = 500.
melt_depth_therm_TG = 500.
melt_depth_therm_SG = 600.
melt_max_PIG = 100.
melt_max_TG = 50.
melt_max_SG = 50.

# Main directory path
# This needs changing in bow
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

output_path = os.path.join(MAIN_PATH,
                            'output/02_gridded_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)


# grid on which to find mask
ase_bbox = {}
for key in config['data_input_extent'].keys():
    ase_bbox[key] = np.float64(config['data_input_extent'][key])
              
xmin = ase_bbox['xmin']
xmax = ase_bbox['xmax']
ymin = ase_bbox['ymin']
ymax = ase_bbox['ymax']

x = np.arange(xmin,xmax+1,1.e3)
y = np.arange(ymin,ymax+1,1.e3)

X,Y = np.meshgrid(x,y)

# read the matlab file

C = io.loadmat(config['input_files']['shelves_outline'])
xs = C['x']
ys = C['y']
M = C['M']
interp = RegularGridInterpolator((xs.flatten(),ys.flatten()), M.T, method='nearest',fill_value=3,bounds_error=False)
Mint = interp((X,Y))

melt_depth_therm_field = np.zeros(np.shape(Mint))
melt_depth_therm_field[Mint==1] = melt_depth_therm_PIG
melt_depth_therm_field[Mint==2] = melt_depth_therm_TG
melt_depth_therm_field[Mint==3] = melt_depth_therm_SG

melt_max_field = np.zeros(np.shape(Mint))
melt_max_field[Mint==1] = melt_max_PIG
melt_max_field[Mint==2] = melt_max_TG
melt_max_field[Mint==3] = melt_max_SG


with h5py.File(os.path.join(output_path,
                            'ase_melt_depth_params.h5'), 'w') as outty:
    data = outty.create_dataset("melt_depth_therm", melt_depth_therm_field.shape, dtype='f')
    data[:] = melt_depth_therm_field
    data = outty.create_dataset("melt_max", melt_max_field.shape, dtype='f')
    data[:] = melt_max_field
    data = outty.create_dataset("x", x.shape, dtype='f')
    data[:] = x
    data = outty.create_dataset("y", y.shape, dtype='f')
    data[:] = y
