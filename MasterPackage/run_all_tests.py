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
    tests_passed = []
    tests_failed = []
    
    print("Conducting unit tests")
    
    print()
    
    print("Testing InputParser...")
    try:
        run_parser_tests()
    except Exception as e :
        print(f"InputParser tests failed with error: {e}")
        tests_failed.append(("InputParser", e))
    else:
        print("InputParser tests passed.")
        tests_passed.append("InputParser")
    
    print() 
    
    print("Testing spline backend...")
    try:
        run_spline_tests()
    except Exception as e:
        print(f"Spline backend tests failed with error: {e}")
        tests_failed.append(("Spline backend", e))
    else:
        tests_passed.append("Spline backend")
        print("Spline backend tests completed.")
    
    print()
    
    print("Testing batch...")
    try:
        run_batch_tests()
    except Exception as e:
        print(f"Batch tests failed with error: {e}")
        tests_failed.append(("Batch", e))
    else:
        print("Batch tests passed.")
        tests_passed.append("Batch")
    
    print()
    
    print("Testing fold generation, saving, precomputation, and saving...")
    try:
        run_fold_precomp_tests(False)
    except Exception as e:
        print("Fold precomp tests failed.")
        tests_failed.append(("Fold precomp", e))
    else:
        print("fold generation, saving, precomputation tests complete.")
        tests_passed.append("Fold precomp")
    
    print()
    
    print("Testing h5handler...")
    try:
        run_h5handler_tests()
    except Exception as e:
        print(f"h5handler tests failed with error: {e}")
        tests_failed.append(("h5handler", e))
    else:
        print("h5handler tests passed.")
        tests_passed.append("h5handler")
    
    print()
    
    print("Testing rotation operator...")
    try:
        run_rotation_test()
    except Exception as e:
        print(f"Rotation operator tests failed with error: {e}")
        tests_failed.append(("Rotation operator", e))
    else:
        print("Rotation test passed.")
        tests_passed.append("Rotation operator")
    
    print()
    
    print("Testing dispersion...")
    try:
        run_dispersion_tests()
    except Exception as e:
        print("Dispersion tests failed.")
        tests_failed.append(("Dispersion", e))
    else:
        print("Dispersion tests passed.")
        tests_passed.append("Dispersion")
    
    print()
    
    print("Testing Dftbplus...")
    try:
        run_dftbplus_tests()
    except Exception as e:
        print(f"Dftbplus tests failed: {e}")
        tests_failed.append(("Dftbplus", e))
    else:
        print("Dftbplus tests passed")
        tests_passed.append("Dftbplus")
    
    print()

    # Layer tests are not functional    
    # print("Testing DFTB Layer...")
    # run_layer_tests()
    # print("DFTB Layer tests passed")
    
    print()
    
    print("Testing DFTBrepulsive...")
    try:
        run_repulsive_tests()
    except Exception as e:
        print(f"Repulsive tests failed: {e}")
        tests_failed.append(("Repulsive", e))
    else:
        print("DFTBrepulsive tests passed")
        tests_passed.append("Repulsive")
    
    print()
    
    # print("Testing total model against benchmarks...")
    # run_total_model_tests() total_model_tests() no longer work due to architecture corrections
    # print("Total model tests passed")
    
    # print()
    
    print("Testing skf writer functionalities...")
    try:
        run_skf_tests(True)
    except Exception as e:
        print(f"SKF tests failed: {e}")
        tests_failed.append(("SKF", e))
    else:
        print("SKF functionality tests passed")
        tests_passed.append("SKF")
    
    print()
    
    print("Testing gammas construction")
    try:
        run_gammas_tests()
    except Exception as e:
        print(f"Gammas tests failed: {e}")
        tests_failed.append(("Gammas", e))
    else:
        print("Gammas tests passed")
        tests_passed.append("Gammas")
    
    print()
    
    print("Testing data invariants...")
    try:
        run_invariant_tests()
    except Exception as e:
        print(f"Invariant tests failed: {e}")
        tests_failed.append(("Invariants", e))
    else:
        print("Data invariant tests passed successfully")
        tests_passed.append("Invariants")
    
    print()
    
    if platform.platform().lower().startswith("linux"):
    
        print("Testing regex parsing consistency...")
        try:
            run_re_tests()
        except Exception as e:
            print(f"Regex tests failed: {e}")
            tests_failed.append(("Regex", e))
        else:
            print("Regex parsing tests passed")
            tests_passed.append("Regex")
    
        print()
    
    
    if tests_failed:
        print("Tests failed:")
        for test_name, err in tests_failed:
            print(f"\t{test_name}: {err}")
    
    print(f"Tests passed: {tests_passed}")
    print(f"{len(tests_passed)}/{len(tests_passed) + len(tests_failed)} tests passed.")


if __name__ == "__main__":
    run_all_tests()
