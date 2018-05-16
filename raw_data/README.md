# Raw Data

This folder holds potential energy surface data for a variety of systems. 
Displacements are generated with MLChem,
and energies are evaluated either with *ab initio* quantum chemisty packages (Molpro, Psi4, etc) or they are generated with the [PyPES-lib](https://github.com/dlc62/pypes-lib) library, which hosts 
analytic potential energies surfaces from the literature for many molecules.
In the case of using PyPES-lib, the energies are relative to the global minimum.
