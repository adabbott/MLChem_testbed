import numpy as np
import pandas as pd
import psi4
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler 
import math
import keras

pd.set_option('display.max_rows',None, 'display.max_columns', 3)

psi4.set_memory('500 MB')
psi4.core.set_output_file("output.dat", True)
# put optimized mp2/6-31g geometry here
mol = psi4.geometry("""
symmetry c1
O           -0.000000000000     0.000000000000    -0.063113521818
H            0.000000000000    -0.794821898387     0.500828695799
H           -0.000000000000     0.794821898387     0.500828695799
""")
mol.update_geometry()

psi4.set_options({'basis': '6-31g',
                  'scf_type': 'df',
                  'points': 3,
                  'units': 'ang',
                  #'disp_size': 0.005,
                  'e_convergence': 1e-8})

e, wfn = psi4.frequency('mp2/6-31g', molecule=mol, return_wfn=True, dertype=0)

# load in displacements in cartesians, as list of numpy arrays
displacements = psi4.core.fd_geoms_freq_0(mol, -1)
# stupid psi4 does disps in bohr
disps = [i.to_array()*0.52917720859 for i in displacements]
print(disps)

# convert cartesian reps to interatomic distance reps
def get_interatom_distances(cart):
    n = len(cart)
    matrix = np.zeros((n,n))
    for i,j in combinations(range(len(cart)),2):
        R = np.linalg.norm(cart[i]-cart[j])
        #create lower triangle matrix
        matrix[j,i] = R
    return matrix

# get len 3 numpy arrays of the three interatom distances: OH1 OH2 and H1H2
idm = [get_interatom_distances(i)[np.tril_indices(3,-1)]  for i in disps]

# use Law of Cosines to replace H1H2 distance with OH1H2 angle
def convert(arry):
    a = arry[0]  
    b = arry[1]  
    c = arry[2]  
    angle = math.degrees(math.acos( (a**2 + b**2 - c**2) / (2*a*b)))
    return angle

final = []
for i in idm:
    angle = convert(i)
    i[2] = angle
    final.append(i)
# "final" is a list of numpy arrays, [r1, r2, a1] which are ready to be added to the dataset, scaled, fed into the model, return the energies, then reap
final = np.asarray(final)
print(final)
ntest = len(final)
print("Number of displacements:",ntest)


# load model obtained from /home/adabbott/Git/MLChem_testbed/HYPERAS_tests/kitchen_sink/kitchen_sink_v6/full_water_unique/Final_model
model = keras.models.load_model("best_model.h5")

# create dataframe of data, add unknown geometries to data, scale, feed in geometries to model
# reproduce model training and testing dataset
data = pd.read_csv("PES.dat")
data = data.drop_duplicates(subset = 'E')
data = pd.concat([data, pd.DataFrame(final,columns=data.columns[0:-1])], sort=False)
data = data.values
scaler = MinMaxScaler(feature_range=(-1,1))
X = data[:,0:-1]
y = data[:,-1].reshape(-1,1)
sX = scaler.fit_transform(X)
print(sX[-ntest:])

predictions = model.predict(sX[-ntest:])
# try just scaling known y's
y = y[:-ntest]
sy = scaler.fit_transform(y)

# now inverse transform the predictions
p = scaler.inverse_transform(predictions)
p = p.reshape(1,-1).squeeze().tolist()
print(p)


shit_freqs = psi4.core.fd_freq_0(mol, p, -1)





