# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 21:28:00 2021

@author: fhu14

Comparing to newer version of DFTB+

Courtesy of Francis for the computed results
"""
#%% Imports, definitions
import pickle
from h5py import File
from SKF import main
from MasterConstants import atom_nums, atom_masses
import os
import numpy as np

#%% Code behind

#%% Past comparison
# #Load in the computed results
# df = pickle.load(open("dftb+_res.pkl", "rb"))

# #Load in the sparse and dense molecules with their predictions
# d = pickle.load(open("dense_skf_comp.p", "rb"))
# s = pickle.load(open("sparse_skf_comp.p", "rb"))

# d_name_confs = [(mol['name'], mol['iconfig']) for mol in d]
# s_name_confs = [(mol['name'], mol['iconfig']) for mol in s]

# assert(d_name_confs == s_name_confs)

# #Compare each one to the dataframe in terms of total energy
# d_v_df = []
# s_v_df = []

# for i, mol in enumerate(d):
#     s_mol = s[i]
#     name, iconf = mol['name'], mol['iconfig']
#     pred_ener_d = mol['predictions']['Etot']['Etot']
#     pred_ener_s = s_mol['predictions']['Etot']['Etot']
#     dplus_ener = df.loc[(df['mol'] == name) & (df['i_conf'] == iconf)]['dftb_plus.total_energy'].item()
#     d_v_df.append(abs(dplus_ener - pred_ener_d))
#     s_v_df.append(abs(dplus_ener - pred_ener_s))

#%% More careful analysis
# More in-depth analysis of SKFs
model_path = "skf_8020_100knot_new_repulsive_eachepochupdate/Split0/saved_models.p"
all_models = pickle.load(open(model_path, 'rb'))

ref_direct = "Auorg_1_1/auorg-1-1"
compute_S_block = True
rep_mode = 'new'

opts = { #Isolated options for writing out SKFs
        
"repulsive_settings" : {

    "rep_setting": "new",
    "opts" : {
        "nknots" : 25,
        "cutoff" : "full",
        "deg" : 3,
        "bconds" : "vanishing",
        "constr" : "+2"
    },
    "gammas_path" : "fold_molecs_test_8020/gammas.p"
},

"skf_settings" : {

    "skf_extension" : "",
    "skf_ngrid" : 50,
    "skf_strsep" : "  ",
    "spl_ngrid" : 500

}
}

dense_direc = "dense_skf"
sparse_direc = "sparse_skf"

if not os.path.isdir(dense_direc):
    os.mkdir(dense_direc)
if not os.path.isdir(sparse_direc):
    os.mkdir(sparse_direc)

with open(os.path.join(dense_direc, "notes.txt"), 'w+') as handle:
    handle.write(f"Derived from {model_path}\n")
with open(os.path.join(sparse_direc, "notes.txt"), 'w+') as handle:
    handle.write(f"Derived from {model_path}\n")

#Write out the sparse and the dense SKFs
#Toggling between sparse and dense is done manually inside skfwriter.py

#Sparse has a grid dist of 0.02 and 500 grid points
# main(all_models, atom_nums, atom_masses, compute_S_block, ref_direct, rep_mode, opts, "  ", 50, "sparse_skf")
# print("Wrote sparse skfs")

#Dense has a grid dist of 0.01 and 1000 grid points
# main(all_models, atom_nums, atom_masses, compute_S_block, ref_direct, rep_mode, opts, "  ", 50, "dense_skf")
# print("Wrote dense skfs")

#%% Safety check for SKFs

dense_direc = "dense_skf"
sparse_direc = "sparse_skf"

#Do a safety check to ensure that the grid distances and grid points are correctly formatted
for filename in os.listdir(dense_direc):
    if filename.split(".")[1] == 'skf':
        content = open(os.path.join(os.getcwd(), dense_direc, filename)).read().splitlines()
        top_line = content[0].split()
        assert(float(top_line[0]) == 0.01)
        assert(float(top_line[1]) == 1000)
        
for filename in os.listdir(sparse_direc):
    if filename.split(".")[1] == 'skf':
        content = open(os.path.join(os.getcwd(), sparse_direc, filename)).read().splitlines()
        top_line = content[0].split()
        assert(float(top_line[0]) == 0.02)
        assert(float(top_line[1]) == 500)

ref_sparse = "sparse_skf/ref_params.p"
ref_dense = "dense_skf/ref_params.p"

d_ref = pickle.load(open(ref_dense, 'rb'))
s_ref = pickle.load(open(ref_sparse, 'rb'))

for key in d_ref:
    if isinstance(d_ref[key], np.ndarray):
        assert(np.allclose(d_ref[key], s_ref[key]))
    else:
        assert(d_ref[key] == s_ref[key])

print("Safety check passed")

#%% Molecule predictions
#Compare the predictions from the model between the two sets (they should be the same)
s_set = "sparse_skf/sparse_skf_comp.p"
d_set = "dense_skf/dense_skf_comp.p"

s_set = pickle.load(open(s_set, 'rb'))
d_set = pickle.load(open(d_set, 'rb'))

s_name_conf = [(mol['name'], mol['iconfig']) for mol in s_set]
d_name_conf = [(mol['name'], mol['iconfig']) for mol in d_set]

assert(s_name_conf == d_name_conf)

disagreements = []

for i, mol in enumerate(s_set):
    d_mol = d_set[i]
    disagreements.append(abs(mol['predictions']['Etot']['Etot'] - d_mol['predictions']['Etot']['Etot']))

assert(max(disagreements) == 0 and min(disagreements) == 0)
assert((sum(disagreements) / len(disagreements)) == 0)







