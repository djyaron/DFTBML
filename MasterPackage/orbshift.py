#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:07:22 2022

@author: yaron
"""

#%% Imports, definitions
import numpy as np
import scipy
from DFTBpy import DFTB
from OrbShift import dftbpy_compare_skf_sets, run_dftbpy
from PlottingUtil import generate_pardict


path1 = "/home/yaron/code/dftbscratch/orig/"
path2 = "/home/yaron/code/dftbscratch/shifted/"

elements = [1]
set1_base_pardict = generate_pardict(path1, elements)
set2_base_pardict = generate_pardict(path2, elements)

cart = np.array([[1, 0.0, 0.0, -0.37], [1, 0.0, 0.0, 0.37]] )

r1, dftb1, eorbs1 = run_dftbpy(set1_base_pardict, cart, 0,1)
r2, dftb2, eorbs2 = run_dftbpy(set2_base_pardict, cart, 0, 1)

e1,f1,rho1 = dftb1.SCF()
e2,f2,rho2 = dftb2.SCF()

#%%

eorbs1, _ = scipy.linalg.eigh(a=f1[0], b = dftb1.GetOverlap())
eorbs2, _ = scipy.linalg.eigh(a=f2[0], b = dftb2.GetOverlap())

#%%
eo1, _ = scipy.linalg.eigh(a=f1[0], b = np.eye(2))
eo2, _ = scipy.linalg.eigh(a=f2[0], b = np.eye(2))

#%%
S = dftb1.GetOverlap()
svals, svecs = np.linalg.eigh(S)
#%%
X = np.copy(svecs)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j] /= np.sqrt(svals[j])

Stest = np.dot(X.T, np.dot(S,X))

FS1 = np.dot(X.T, np.dot(f1[0],X))
FS2 = np.dot(X.T, np.dot(f2[0],X))

etest1, _ = np.linalg.eigh(FS1)
etest2, _ = np.linalg.eigh(FS2)


