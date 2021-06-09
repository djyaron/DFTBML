# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:32:54 2021

@author: fhu14

Simple driver file for running all tests
"""
from Tests import *

if __name__ == "__main__":
    
    print("Testing InputParser...")
    run_parser_tests()
    print("InputParser tests passed")
    
    print() 
    
    print("Testing spline backend...")
    run_spline_tests()
    print("Spline backend tests completed")
    
    print()
    
    print("Testing batch...")
    run_batch_tests()
    print("Batch tests passed")
    
    print()
    
    print("Testing h5handler...")
    run_h5handler_tests()
    print("h5handler tests passed")
    
    print()
    
    print("Testing fold generation, saving, precomputation, and saving...")
    # run_fold_precomp_tests()
    print("fold generation, saving, precomputation tests complete.")



