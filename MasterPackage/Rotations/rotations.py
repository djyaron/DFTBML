# -*- coding: utf-8 -*-
#%% Imports, definitions
import math
import numpy as np
import unittest

#%% Code behind

def get_axis_rotation_matrix(axis, theta):
    # http://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
    # https://en.wikipedia.org/wiki/Rotation_matrix
    ct = math.cos(theta)
    nct = 1 - ct
    st = math.sin(theta)
    r = np.linalg.norm(axis)
    if r == 0.0:
        return np.matrix(np.eye(3))
    ux = axis[0] / r
    uy = axis[1] / r
    uz = axis[2] / r
    rot = np.matrix([
        [ct + ux ** 2 * nct, ux * uy * nct - uz * st, ux * uz * nct + uy * st],
        [uy * ux * nct + uz * st, ct + uy ** 2 * nct, uy * uz * nct - ux * st],
        [uz * ux * nct - uy * st, uz * uy * nct + ux * st, ct + uz ** 2 * nct],
    ])
    return rot

def rotation_matrix_atom(operator):
    eigen_vals, eigen_vectors = np.linalg.eig(operator)
    rot_matrix = eigen_vectors.T
    return rot_matrix, eigen_vals
    
def rotation_matrix_bond(bond_vector,operator):
    '''
    We will transform from a space-fixed (xyz) to molecule fixed frame (mol).
    The molecule fixed frame will have the first axis lie along "bond_vector"
    the remaining axes diagonalize "operator" in the 2x2 space perpendicular
    to bond_vector, with the second axis having a larger eigenvalue than the
    first vector. 
    
    Required input:
        bond_vector: array of length 3 in space-fixed frame (see above)
        operator:    3x3 array in space fixed-frame (see above)
    
    Output:
        R:   <mol|xyz> overlap matrix, so that transformations are:
            vector_mol  = R   * vector_xyz
            vector_xyz  = R.T * vector_mol
            O_mol       = R   * O_xyz  * R.T
            O_xyz       = R.T * O_mol  * R
       eigen_vals:  eigenvalues of the operator about the rotation axis
             to indicate how unique the chosen rotation is
    '''
    # Begin by constructing a rotation matrix, R1, that rotates e_x to e_bond
    e_bond = (bond_vector) / np.linalg.norm(bond_vector)
    # Need a unit vector perpendicular to e_bond, which we can get by crossing
    # e_x, e_y or e_z into e_bond. The caveat is that we should not cross
    # e_bond into a vector that is parallel to e_bond. We will therefore
    # select e_x e_y or e_z to be that along with e_bond has the smallest
    # projection
    axis_for_cross = np.argmin(np.abs(e_bond))
    e_cross = [0,0,0]
    e_cross[axis_for_cross] = 1
    e_1 = np.cross(e_bond,e_cross)
    e_1 = e_1/np.linalg.norm(e_1)
    # choose third axis so (e_bond, e_1, e_2) form a right handed 
    # coordinate system
    e_2 = np.cross(e_bond,e_1)
    # assemble as columns to make R1, so that R1 * e_bond = e_x
    # (bond_vector is the bond expressed in space-fixed coordinates,
    #  so we want R * bond_vector to lie along [1,0,0] direction in mol coords
    R1 = np.vstack([e_bond,e_1,e_2])
    
    # Next, rotate about the bond axis to principle axes of "operator"
    # Rotate operator from the space-fixed frame to that above
    O1 = np.dot(R1, np.dot(operator,R1.T) )
    # Take the 2x2 space that is perpendicular to e_bond
    O1_2x2 = O1[1:, 1:]
    eigen_vals, eigen_vectors = np.linalg.eigh(O1_2x2)
    # Sort eigenvalues [High, ..., Low]
    idx = eigen_vals.argsort()[::-1]
    R2_2x2 = eigen_vectors[:,idx]
    # expand to a 3x3 matrix that leaves e_bond unchanged.
    R2 = np.matrix(np.eye(3))
    #R2[1:,1:] = R2_2x2.T

    # At this point, R2 * O1 * R2.T = diagonal in upper 2x2 subspace
    # Therefore   R2 * R1 *O1 * R1.T * R2.T is diagonal in upper 2x2 substance
    R = np.dot(R2,R1)
    return R, eigen_vals
