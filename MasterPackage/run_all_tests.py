# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:32:54 2021

@author: fhu14

Simple driver file for running all tests

NOTE: For best results, the test_files directory should be the most 
up to date and DFTB+ version 20+ should be installed (preferably a unix/linux environment)

The regex test can ONLY be run using this setup, hence the platform check
before running the regex test. All other tests can be run on windows (with a few minor modifications
to the test files)
"""
from Tests import *
import platform

def run_all_tests():
    
    print("Conducting unit tests")
    
    print()
    
    print("Testing InputParser...")
    run_parser_tests()
    print("InputParser tests passed.")
    
    print() 
    
    print("Testing spline backend...")
    run_spline_tests()
    print("Spline backend tests completed.")
    
    print()
    
    print("Testing batch...")
    run_batch_tests()
    print("Batch tests passed.")
    
    print()
    
    print("Testing fold generation, saving, precomputation, and saving...")
    run_fold_precomp_tests(False)
    print("fold generation, saving, precomputation tests complete.")
    
    print()
    
    print("Testing h5handler...")
    run_h5handler_tests()
    print("h5handler tests passed.")
    
    print()
    
    print("Testing rotation operator...")
    run_rotation_test()
    print("Rotation test passed.")
    
    print()
    
    print("Testing dispersion...")
    run_dispersion_tests()
    print("Dispersion tests passed.")
    
    print()
    
    print("Testing Dftbplus...")
    run_dftbplus_tests()
    print("Dftbplus tests passed")
    
    print()
    
    print("Testing DFTB Layer...")
    run_layer_tests()
    print("DFTB Layer tests passed")
    
    print()
    
    print("Testing DFTBrepulsive...")
    run_repulsive_tests()
    print("DFTBrepulsive tests passed")
    
    print()
    
    # print("Testing total model against benchmarks...")
    # run_total_model_tests() total_model_tests() no longer work due to architecture corrections
    # print("Total model tests passed")
    
    # print()
    
    print("Testing skf writer functionalities...")
    run_skf_tests(True)
    print("SKF functionality tests passed")
    
    print()
    
    print("Testing gammas construction")
    run_gammas_tests()
    print("Gammas tests passed")
    
    print()
    
    print("Testing data invariants...")
    run_invariant_tests()
    print("Data invariant tests passed successfully")
    
    print()
    
    if platform.platform().lower().startswith("linux"):
    
        print("Testing regex parsing consistency...")
        run_re_tests()
        print("Regex parsing tests passed")
    
        print()
    
    print("All unit tests passed successfully")

if __name__ == "__main__":
    run_all_tests()


