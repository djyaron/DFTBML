# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:33:44 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import List, Dict
import pickle
import numpy as np
import os
from DataManager import total_feed_combinator
import Geometry.geometry

#%% Code behind

def compare_feeds(reference_file: str, reconstituted_feeds: List[Dict]) -> None:
    r"""Function for checkig that the reconstituted feeds matches the original data
    
    Arguments:
        reference_file (str): Name of the pickle file that stores the original data
        reconstituted_feeds (List[Dict]): Reconstructed feeds that need checking
    
    Returns:
        None
    
    Raises:
        AssertionError: If any of the tests fail
    
    Notes: For ragged data like dipole matrices and charges, that information is 
        handled with a small for loop.
    """
    reference_file_feeds = pickle.load(open(reference_file, 'rb'))
    assert(len(reconstituted_feeds) == len(reference_file_feeds))
    for i in range(len(reference_file_feeds)):
        curr_ref_fd = reference_file_feeds[i]
        feedi = reconstituted_feeds[i]
        #Assert same basis sizes
        assert( set(curr_ref_fd['basis_sizes']).difference(set(feedi['basis_sizes'])) == set() )
        assert( set(feedi['basis_sizes']).difference(set(curr_ref_fd['basis_sizes'])) == set() )
        
        assert( curr_ref_fd['onames'] == feedi['onames'] )
        
        assert( set(curr_ref_fd['models']).difference(set(feedi['models'])) == set() )
        assert( set(feedi['models']).difference(set(curr_ref_fd['models'])) == set() )
        
        #Need to check mod_raw
        for mod_spec in curr_ref_fd['mod_raw']:
            assert( curr_ref_fd['mod_raw'][mod_spec] == feedi['mod_raw'][mod_spec] )
        
        assert( set(curr_ref_fd['gather_for_rot'].keys()).difference(set(feedi['gather_for_rot'].keys())) == set() )
        assert( set(feedi['gather_for_rot'].keys()).difference(set(curr_ref_fd['gather_for_rot'].keys())) == set() )
        
        for shp in curr_ref_fd['gather_for_rot']:
            assert( np.allclose (curr_ref_fd['gather_for_rot'][shp], feedi['gather_for_rot'][shp]) )
        
        for shp in curr_ref_fd['rot_tensors']:
            assert( 
                ((feedi['rot_tensors'][shp] is None) and (curr_ref_fd['rot_tensors'][shp] is None)) or\
                    np.allclose(curr_ref_fd['rot_tensors'][shp], feedi['rot_tensors'][shp])
                )
        #Assert the same for all things indexed by basis sizes
        for bsize in curr_ref_fd['glabels'].keys():
            assert( len(curr_ref_fd['glabels'][bsize]) == len(feedi['glabels'][bsize]) )
            assert( list(curr_ref_fd['glabels'][bsize]) == list(feedi['glabels'][bsize]) )
            
            assert( len(curr_ref_fd['names'][bsize]) == len(feedi['names'][bsize]) )
            assert( list(curr_ref_fd['names'][bsize]) == list(feedi['names'][bsize]) )
            
            assert( len(curr_ref_fd['iconfigs'][bsize]) == len(feedi['iconfigs'][bsize]) )
            assert( list(curr_ref_fd['iconfigs'][bsize]) == list(feedi['iconfigs'][bsize]) )
            
            assert( np.allclose(curr_ref_fd['gather_for_rep'][bsize], feedi['gather_for_rep'][bsize]))
            assert( np.allclose(curr_ref_fd['segsum_for_rep'][bsize], feedi['segsum_for_rep'][bsize]))
            
            assert( np.allclose(curr_ref_fd['norbs_atom'][bsize], feedi['norbs_atom'][bsize]) )
            
            assert( np.allclose(curr_ref_fd['atom_ids'][bsize], feedi['atom_ids'][bsize]) )
            
            if 'S' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['S'][bsize], feedi['S'][bsize]) )
            
            if 'G' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['G'][bsize], feedi['G'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['Etot'][bsize], feedi['Etot'][bsize]) )      

            assert( np.allclose (curr_ref_fd['Eelec'][bsize], feedi['Eelec'][bsize]) )

            assert( np.allclose (curr_ref_fd['Erep'][bsize], feedi['Erep'][bsize]) ) 
            
            if 'Sevals' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['Sevals'][bsize], feedi['Sevals'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['dQ'][bsize], feedi['dQ'][bsize]) ) 

            assert( np.allclose (curr_ref_fd['eorb'][bsize], feedi['eorb'][bsize]) ) 
            
            for i in range(len(curr_ref_fd['dipole_mat'][bsize])):
                assert( np.allclose (curr_ref_fd['dipole_mat'][bsize][i], feedi['dipole_mat'][bsize][i]))
            
            assert( np.allclose (curr_ref_fd['dipoles'][bsize], feedi['dipoles'][bsize]))
            
            # assert( np.allclose (curr_ref_fd['charges'][bsize], feedi['charges'][bsize]))
            for i in range(len(curr_ref_fd['charges'][bsize])):
                assert(np.allclose(curr_ref_fd['charges'][bsize][i], feedi['charges'][bsize][i]))
            
            assert( np.allclose (curr_ref_fd['occ_rho_mask'][bsize], feedi['occ_rho_mask'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['occ_eorb_mask'][bsize], feedi['occ_eorb_mask'][bsize]) )
            
            if 'phiS' in curr_ref_fd:
                assert( np.allclose (curr_ref_fd['phiS'][bsize], feedi['phiS'][bsize]) )
            
            assert( np.allclose (curr_ref_fd['rho'][bsize], feedi['rho'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['zcounts'][bsize], feedi['zcounts'][bsize]) ) 
            
            assert( np.allclose (curr_ref_fd['qneutral'][bsize], feedi['qneutral'][bsize]) )
            
            for oper in curr_ref_fd['gather_for_oper'].keys():
                assert(np.allclose(curr_ref_fd['gather_for_oper'][oper][bsize], feedi['gather_for_oper'][oper][bsize] ))
            
            for oper in curr_ref_fd['onames']:
                if (oper != 'H') and (oper in curr_ref_fd):
                    assert(np.allclose(curr_ref_fd[oper][bsize], feedi[oper][bsize]))
                    
    print("Tests passed!")

def run_h5handler_tests():
    batch_filename = os.path.join(os.getcwd(), "fold_molecs", "Fold0", "batches.h5")
    molec_filename = os.path.join(os.getcwd(), "fold_molecs", "Fold0", "molecs.h5")
    reference_filename = os.path.join(os.getcwd(), "fold_molecs", "Fold0", "reference_data.p")
    reconstituted_feeds = total_feed_combinator.create_all_feeds(batch_filename, molec_filename, True)
    compare_feeds(reference_filename, reconstituted_feeds)