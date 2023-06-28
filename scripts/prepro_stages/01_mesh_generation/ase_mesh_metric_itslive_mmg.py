"""
Produce a metric which can be used by gmsh to make a good quality variable
mesh for ASE.

Steps:

  - Loads velocity data & computes eigenstrains
  - Uses ITSLive composite solution for strain maps with full coverage
  - Crops this data to the require extent
    (we use the extent from the ASE_vel_timeseries)
  - Produces a metric from the strain
  - Loads a mask indicating ice/ocean/rock
  - Polygonizes mask raster into ice/ocean (ignores rock)
  - Labels boundaries of ice polygon (calving front or natural)
  - Uses MMG (subprocess) to adapt mesh to metric
  - Produces FEniCS-ready Mesh & MeshValueCollection

Possible future work:

  - Other options for metric:
       non-linear strain dependence
       proximity to calving front
  - Generalise extent definition (ASE specific at present)
  - Package methods into a class?
"""
import os
import sys
import numpy as np
from netCDF4 import Dataset as NCDataset
import meshio
import xarray as xr
from configobj import ConfigObj
import numpy.ma as ma
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini",
                    help="pass config file")
parser.add_argument("-lc",
                    type=float,
                    default=1000,
                    help="mesh size parameter in gmsh")
parser.add_argument("-len_min",
                    type=float,
                    default=100,
                    help="We set a uniform mesh size by "
                         "modifying the GMSH options, "
                         "just for the minimum value, "
                         "we let the maximum be set by the domain scale")
parser.add_argument("-mmg_hgrad",
                    type=float,
                    default=1.3,
                    help="maximum edge-length ratio of "
                         "neighbouring elements (float) mmg param")
parser.add_argument("-mmg_hausd",
                    type=float,
                    default=100,
                    help="max distance by which edges "
                         "can be moved from original")
parser.add_argument("-smooth_coast",
                    type=bool,
                    default=False,
                    help="If specified we smooth the calving front")

args = parser.parse_args()
config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

# Mesh params
lc = args.lc
len_min = args.len_min

# MMG params
hgrad = args.mmg_hgrad
hausd = args.mmg_hausd

#smooth coastline and calving front
smooth = args.smooth_coast

# Main directory path
MAIN_PATH = config['main_path']
gmsh_lib_path = config['gmsh_path']
fice_tools = config['ficetoos_path']

if gmsh_lib_path:
    sys.path.append(gmsh_lib_path)

sys.path.append(fice_tools)

import gmsh
from ficetools import mesh as meshtools
from ficetools import velocity as vel_tools

# In files
velocity_netcdf = os.path.join(config['input_files']['itslive'],
                               'ANT_G0240_0000.nc')
bedmachine = config['input_files']['bedmachine']
use_bedmachine = True

# Out files
output_path = os.path.join(MAIN_PATH,
                            'output/01_mesh')
if not os.path.exists(output_path):
    os.makedirs(output_path)

mesh_outfile = os.path.join(output_path, 'ase_variable_ocean')

lc_params = {}
for key in config['mesh_lc_params'].keys():
    lc_params[key] = np.float64(config['mesh_lc_params'][key])

mesh_extent = {}
for key in config['mesh_extent'].keys():
    mesh_extent[key] = np.float64(config['mesh_extent'][key])

meshtools.set_mmg_lib(config['mmg_path'])

###############################################################
# Load the velocity data, compute eigenstrain, produce metric
###############################################################
vel_data = xr.open_dataset(velocity_netcdf)

### Get ASE time series extend for referece only!!!
velocity_ase = config['input_files']['velocity_ase_series']
vel_ase = NCDataset(velocity_ase)

#years = meshtools.get_netcdf_vel_years(vel_data)
#dx = meshtools.get_dx(vel_data)
xx_ase, yy_ase = meshtools.get_netcdf_coords(vel_ase)

ase_extent = {'xmin': min(xx_ase),
               'xmax': max(xx_ase),
               'ymin': min(yy_ase),
               'ymax': max(yy_ase)}

# Now crop ITSlive to ase extend domain
vx = vel_data.vx
vy = vel_data.vy

vx_live = vel_tools.crop_velocity_data_to_extend(vx, ase_extent, return_xarray=True)
vy_live = vel_tools.crop_velocity_data_to_extend(vy, ase_extent, return_xarray=True)

xx_live = vx_live.x.values
yy_live = vx_live.y.values
dx = xx_live[1]-xx_live[0]

# Rename variables already cropped!
# We could use the function meshtools.get_eigenstrain_rate
# But there are some issues with the how the mask is define
# on that function so we need to make our own with numpy.ma
# as the itslive data has a different shape than the ASE_series
vxc = vx_live.data
vyc = vy_live.data

dvx_dy, dvx_dx = np.gradient(vxc, dx)
dvy_dy, dvy_dx = np.gradient(vyc, dx)

dvx_dy *= -1.0
dvy_dy *= -1.0

xy_shear = (dvx_dy + dvy_dx) * 0.5

shape1 = dvx_dx.shape[0]
shape2 = dvx_dx.shape[1]

strain_hor = np.reshape(np.stack((dvx_dx,
                                  xy_shear,
                                  xy_shear,
                                  dvy_dy), axis=2),
                        newshape=(shape1, shape2, 2, 2))

eig, __ = np.linalg.eigh(strain_hor)

dvx_dy_ma = ma.masked_invalid(dvx_dy)
dvx_dx_ma = ma.masked_invalid(dvx_dx)

# Mask according to both gradient directions
# Taking gradients in x & y directions results
# in different missing values, so combine and mask
mask = (dvx_dy_ma.mask | dvx_dx_ma.mask)
eig[mask] = np.nan

all_strain_mats = np.stack(list(eig))
eigsum = np.sum(np.abs(all_strain_mats), axis=2)

metric = meshtools.simple_strain_metric(eigsum, lc_params)

###############################################################
# Polygonize mask raster, generate gmsh domain
###############################################################

# Get the mask from BedMachine_Antarctica
bedmachine_data = NCDataset(bedmachine)
mask, mask_transform = meshtools.slice_netcdf(bedmachine_data, 'mask', mesh_extent)

# mask: [0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = float ice, 4 = Lake Vostok]
# We want to ignore nunataks (1) and just impose min thick, and at this stage we don't
# care about grounded/floating, so:
#
# 0 -> 0
# 1,2,3,4 -> 1
mask = (mask >= 1).astype(np.int64)

gmsh_ring, ice_labels, ocean_labels = meshtools.generate_boundary(mask,
                                                                  mask_transform,
                                                                  simplify_tol=lc_params['simplify_tol'],
                                                                  bbox=mesh_extent,
                                                                  smooth_front=smooth)

ice_tag, ocean_tags = meshtools.build_gmsh_domain(gmsh_ring,
                                                  ice_labels,
                                                  ocean_labels,
                                                  lc=lc,
                                                  mesh_size_min=lc_params['min'],
                                                  mesh_size_max=lc_params['max']
                                                  )

meshtools.tags_to_file({'ice': [ice_tag],
                        'ocean': ocean_tags},
                       mesh_outfile+"_BCs.txt")

#gmsh.option.setNumber("Mesh.CharacteristicLengthMin", len_min)
gmsh.model.geo.synchronize()

# Create the (not yet adapted) mesh
gmsh.model.mesh.generate(2)
gmsh.write(mesh_outfile+".msh")

# Add the post-processing data (the metric) via interpolation
interped_metric = meshtools.interp_to_gmsh(metric, xx_live, yy_live)
meshtools.write_medit_sol(interped_metric, mesh_outfile+".sol")

gmsh.finalize()

###############################################################
# Adapt with MMG, then produce FEniCS ready Mesh and MeshValueCollection
###############################################################

meshtools.gmsh_to_medit(mesh_outfile+".msh", mesh_outfile+".mesh")

# Adapt with MMG
meshtools.run_mmg_adapt(mesh_outfile+".mesh",
                        mesh_outfile+".sol",
                        hgrad=hgrad,
                        hausd=hausd)

meshtools.remove_medit_corners(mesh_outfile+".o.mesh")

# This 'mixed' mesh works fine for MMG (tris and lines), but fenics
# can't handle it. Need to feed in a triangle-only mesh and a facet-function.
adapted_mesh = meshio.read(mesh_outfile+".o.mesh")


# Extract the pts & triangle elements, write to file
fenics_mesh = meshtools.extract_tri_mesh(adapted_mesh)
meshio.write(mesh_outfile+".xdmf", fenics_mesh)

fmesh = meshtools.load_fenics_mesh(mesh_outfile+".xdmf")
mvc = meshtools.lines_to_mvc(adapted_mesh, fmesh, marker_name="medit:ref")

meshtools.write_mvc(mvc, mesh_outfile+"_ff.xdmf")

# Clear up intermediate files
meshtools.delete_intermediates(mesh_outfile)

# Now lets print some stats on the mesh
mesh_fenics_ice = meshtools.load_fenics_mesh(mesh_outfile+".xdmf")

print('--------- mesh stats ---------')
print('This is the maximum cell size')
print(mesh_fenics_ice.hmax())
print('This is the minimum cell size')
print(mesh_fenics_ice.hmin())
print('Total number of elements after refinement')
print(len(mesh_fenics_ice.cells()))
