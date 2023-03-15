"""
Slice data from BedMachine netCDF and store in fenics_ice ready format.

It also applies gaussian filtering,
and redefines the ice base where required for hydrostatic
equilibrium.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import netCDF4
import h5py
from scipy.ndimage import gaussian_filter
import argparse
from configobj import ConfigObj

parser = argparse.ArgumentParser()
parser.add_argument("-sigma",
                    type=float,
                    default=0.0,
                    help="Sigma value for gauss filter - "
                         "zero means no filtering")
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Main directory path
# This needs changing in bow
MAIN_PATH = config['main_path']
fice_tools = config['ficetoos_path']
sys.path.append(fice_tools)

from ficetools import mesh as meshtools

gauss_sigma = args.sigma
filt = gauss_sigma > 0.0

bedmac_file = os.path.join(MAIN_PATH, config['input_files']['bedmachine'])

# Out files
output_path = os.path.join(MAIN_PATH,
                           'output/02_gridded_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)

out_file = Path(os.path.join(MAIN_PATH,
                             config['output_files']['ase_bedmachine']))

if filt:
    out_file = out_file.stem + \
               "_filt_" + \
               str(gauss_sigma) + \
               out_file.suffix

rhoi = 917.0
rhow = 1030.0

ase_bbox = {}
for key in config['data_input_extent'].keys():
    ase_bbox[key] = np.float64(config['data_input_extent'][key])

indata = netCDF4.Dataset(bedmac_file)

# xx = indata['x']
# yy = indata['y']

bed, xx, yy = meshtools.slice_netcdf(indata,
                                     'bed',
                                     ase_bbox,
                                     return_transform=False)

surf, _, _ = meshtools.slice_netcdf(indata,
                                    'surface',
                                    ase_bbox,
                                    return_transform=False)

thick, _, _ = meshtools.slice_netcdf(indata,
                                     'thickness',
                                     ase_bbox,
                                     return_transform=False)

mask, _, _ = meshtools.slice_netcdf(indata,
                                    'mask',
                                    ase_bbox,
                                    return_transform=False)

#####################################################################
# Smooth surf, then redefine ice base as min(bed, surf-floatthick)
#####################################################################


def filt_and_show(arr, sigma):
    """
    # Convenience function for param sweep sigma (1.5)
    arr: array to which the filter will be applied to
    sigma: gaussian filter for topography
    out: plots the gaussian filter applied to
         bedmachine
    """
    result = gaussian_filter(arr, sigma)
    plt.matshow(result[50:150, 100:200])
    plt.show()


def gaussian_nan(arr, sigma, trunc=4.0):
    """
    Clever approach to gaussian filter w/ nans:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    arr: array with nans
    sigma: filter to applied
    trunc: Truncate the filter at this many standard deviations.
        Default is 4.0.
    """
    arr1 = arr.copy()
    arr1[np.isnan(arr)] = 0.0

    arr2 = np.ones_like(arr)
    arr2[np.isnan(arr)] = 0.0

    arr1_filt = gaussian_filter(arr1, sigma, truncate=trunc)
    arr2_filt = gaussian_filter(arr2, sigma, truncate=trunc)

    result = arr1_filt / arr2_filt
    result[np.isnan(arr)] = np.nan
    return result


if filt:

    # Smooth the surface
    surf_filt = surf.copy()
    # mask ocean/nunatak
    surf_filt[mask <= 1] = np.nan
    # gauss filter accounting for nans
    surf_filt = gaussian_nan(surf_filt, gauss_sigma)
    # surf_filt[mask <= 1] = 0.0

    assert np.nanmin(surf_filt) >= 0.0

    # Compute flotation thickness (i.e. max thick)
    float_thick = surf_filt * (rhow / (rhow - rhoi))
    # float_thick[surf_filt == 0.0] = 0.0

    # And the ice base (max of flotation base or BedMachine bed)
    base_float = surf_filt - float_thick
    ice_base = np.maximum(base_float, bed)

    thick_mod = surf_filt - ice_base

    thick_mod = np.clip(thick_mod, a_min=0.0, a_max=None)
    thick_mod[np.isnan(thick_mod)] = 0.0

else:

    thick_mod = thick
    surf_filt = surf

file_name_out = os.path.join(MAIN_PATH,
                             config['output_files']['ase_bedmachine'])

with h5py.File(file_name_out, 'w') as outty:
    data = outty.create_dataset("bed", bed.shape, dtype='f')
    data[:] = bed
    data = outty.create_dataset("thick", thick_mod.shape, dtype='f')
    data[:] = thick_mod
    data = outty.create_dataset("surf", surf_filt.shape, dtype='f')
    data[:] = surf_filt
    data = outty.create_dataset("x", xx.shape, dtype='f')
    data[:] = xx
    data = outty.create_dataset("y", yy.shape, dtype='f')
    data[:] = yy
