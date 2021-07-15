# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:51:38 2021

@author: fhu14

Test suite for different functionalities present in SKF writer.
"""

#%% Imports, definitions
from SKF import determine_index
from MasterConstants import Model


#%% Code behind

def test_determine_index():
    r"""Tests that the proper indices are used for the model values
    """
    
    print("Testing get index...")
    
    dummy_mod_specs = [
        
        Model(oper='H', Zs=(6, 1), orb='ss'),
        Model(oper='H', Zs=(1, 6), orb='sp'),
        Model(oper='H', Zs=(7, 7), orb='pp_pi'),
        Model(oper='H', Zs=(7, 7), orb='pp_sigma'),
        
        Model(oper='S', Zs=(6, 1), orb='ss'),
        Model(oper='S', Zs=(1, 6), orb='sp'),
        Model(oper='S', Zs=(7, 7), orb='pp_pi'),
        Model(oper='S', Zs=(7, 7), orb='pp_sigma')
        
        ]
    
    target_inds = [9, 8, 6, 5, 9, 8, 6, 5]
    
    for i, mod_spec in enumerate(dummy_mod_specs):
        assert(determine_index(mod_spec) == target_inds[i])
    
    print("Testing index passed successfully")

def run_skf_tests():
    test_determine_index()


#%% Main block
if __name__ == '__main__':
    run_skf_tests()

