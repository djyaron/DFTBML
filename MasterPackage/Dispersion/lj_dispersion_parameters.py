# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:54:30 2021

@author: fhu14

This is a set of parameters used for the Lennard-Jones dispersion form, 
translated from the paper "UFF, a Full Periodic Table Force Field for Molecular
Mechanics and Molecular Dynamics Simulations" by Rappe et. al. 

We are interested in the van der waals (VDW) parameters, which are not
geometry/hybridization dependent, specifically the van der waal distance (r) and
the well depth (d).

The parameters are defined for each atom type. We will make no distinction
between hybridization differences, and the data will be represented on the basis
of atomic numbers to work better with the other code infrastructure.

For the VDW distance, the numbers here are in Angstroms and for the well depth,
the numbers are originall in kcal/mol, but are converted to Hartrees following
the conversion factor 627.5 kcal/mol = 1 Ha. 

As more elements are needed, more of the data will be added to this module.
"""

KCAL_TO_HARTREE = 1/627.5 #To go from kcal to Ha, multiply by this factor

VDW_dists = {
    
    1 : 2.886,
    6 : 3.851,
    7 : 3.660,
    8 : 3.500,
    16 : 4.035,
    79 : 3.293
    
    }

VDW_well = {
    
    1 : 0.044 * KCAL_TO_HARTREE,
    6 : 0.105 * KCAL_TO_HARTREE,
    7 : 0.069 * KCAL_TO_HARTREE,
    8 : 0.060 * KCAL_TO_HARTREE,
    16 : 0.274 * KCAL_TO_HARTREE,
    79 : 0.039 * KCAL_TO_HARTREE
    
    }

#THIS VALUE IS A PLACEHOLDER, FIND OUT THE ACTUAL DISTANCE AT WHICH THE VDW
#   INTERACTION GOES FROM REPULSIVE TO ATTRACTIVE!
cutoff_distance = 2.00 #Angstroms


