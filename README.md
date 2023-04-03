# Amundsen Basin FEniCs_ice experiments

This repository consist of several experiments with Fenics_ice over the Amundsen Basin glaciers.

Note: this repository documentation is a work in progress 👷

The code has been developed to:

- Create a finite element mesh of the region of interest, where simulations will be carried out.
- Generate the all input data (crop to the study domain) needed by Fenics_ice.
- And to carry out model experiments

Details of how to apply the fenics_ice code to a real domain can be found on this [wiki.](https://github.com/bearecinos/smith_glacier/wiki)

Due to several memory issues with `mpi4py` and `hdf5` explained in [here](https://github.com/EdiGlacUQ/fenics_ice/issues/117), we have fixed the fenics_ice environment for the Amundsen region.

# Installation & usages

1. Clone the repositories:

~~~
git clone https://github.com/EdiGlacUQ/fenics_ice.git
git clone https://github.com/EdiGlacUQ/tlm_adjoint.git
git clone https://github.com/bearecinos/ASE_fenics_ice_exp.git
~~~
> **Note**: this repository is private at the moment so use ssh cloning for access 

2. Create an environment using **ASE_fenics_ice_exp** `environment.yml` 
~~~
conda env create --file path_to_repo\ASE_fenics_ice_exp\environment.yml
~~~

3. Activate the environment via `conda activate fenics_ice_ase` and install fenics_ice and tml_adjoint via pip:

~~~
cd fenics_ice
pip install -e .
~~~

~~~
cd tlm_adjoint
pip install -e .
~~~

4. Run all serial tests:
~~~
pytest -v --order-scope=module --color=yes
~~~
5. Run all parallel tests::
~~~
mpirun -n 2 pytest -v --order-scope=module --color=yes
~~~
6. Make sure you add to your `.bashrc` a path to the `fenics_ice` repo:
~~~
export FENICS_ICE_BASE_DIR="path/to/fenics_ice"
~~~

- For MMGtools installation please refer to this [link](https://github.com/bearecinos/smith_glacier/wiki/How-to-install#mmgtools-installation)
- For ficetools dependencies please refer to this [link](https://github.com/bearecinos/smith_glacier/wiki/How-to-install#dependencies-outside-of-fenics_ice-environment). 

> **Important**: If you are installing this in Edi-Geoscience bow VM make sure you install `gmsh` 
> according to [this documentation](https://github.com/bearecinos/smith_glacier/issues/12#issuecomment-1494475900). 
> And do not install `gmsh` from conda or pip as stated in the ficetools link above.