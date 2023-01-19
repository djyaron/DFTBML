# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:13:27 2021

@author: fhu14
"""
import numpy as np

ANGSTROM2BOHR = 1.889725989

def TriuToSymm(matrix):
    for ind in range(matrix.shape[0]):
        matrix[ind:, ind] = matrix[ind, ind:]
    return matrix

def BMatArr(matrix):
    return np.array(np.bmat(matrix))