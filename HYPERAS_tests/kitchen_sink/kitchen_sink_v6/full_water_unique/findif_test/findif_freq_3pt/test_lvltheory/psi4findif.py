import numpy as np
import pandas as pd
import psi4
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import math
import keras

psi4.set_memory('500 MB')
psi4.core.set_output_file("output.dat", True)
# put optimized mp2/6-31g geometry here
mol = psi4.geometry("""
symmetry c1
O           
H 1 0.8     
H 1 0.8 2 90.0
""")
mol.update_geometry()

psi4.set_options({'basis': '6-31g',
                  'scf_type': 'df',
                  'points': 5,
                  'disp_size': 0.05,
                  'e_convergence': 1e-8})

e = psi4.energy('mp2',molecule=mol)
