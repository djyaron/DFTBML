# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:03:29 2021

@author: fhu14
"""
#%% Imports, definitions
import numpy as np
from rotations import rotation_matrix_bond

#%% Code behind

def testRotateOper(verbose = True, eps = 1e-12):
    # make a random axis and symmetric operator
    if verbose:
        print("Testing rotateOper.py: Generating a random bond axis.")
    rbond = np.random.rand(3)
    o1 = np.random.rand(3,3)
    o1 = o1 + o1.transpose()
    
    R = rotation_matrix_bond(rbond, o1)[0]
    
    # is it an orthonormal matrix
    t1 = np.dot(R.T, R);
    t2 = np.dot(R, R.T);
    one = np.eye(3)
    d1 = np.max(np.abs(t1-one));
    d2 = np.max(np.abs(t2-one));
    if verbose:
        print("Testing rotateOper.py: Deviation of R from orthnormality:", \
              np.max([d1,d2]))
    
    # Does R rotate ebond to [1,0,0]
    ebond = rbond/np.linalg.norm(rbond)
    x = [1., 0., 0.]
    xnew = np.dot(R,ebond)
    d3 = np.max(np.abs(xnew - x))
    if verbose:
        print("Testing rotateOper.py: Deviation of R*ebond", \
              "from x axis:", d3) 
    
    # does rotation of o1 from space to molecular frame end up being diagonal
    omol = np.dot(R,np.dot(o1,R.T))
    d4 = abs(omol[1,2]) + abs(omol[2,1])
    if verbose:
        print("Testing rotateOper.py: Deviation of R*operator*R.T", \
              "from being diagonal in [1:2,1:2] subblock:", d4) 
    
    
    if np.max([d1,d2,d3,d4]) < eps:
        return True
    else:
        return False
    
#%% Main block
if __name__ == "__main__":
    testRotateOper()
