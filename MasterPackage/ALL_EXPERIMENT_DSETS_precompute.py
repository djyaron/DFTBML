# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:39:26 2022

@author: fhu14
"""
'''
Separate module for precomputing with a lock from if __name__ == "__main__":
'''
#Taking no changes, putting the imports into the if __name__ == "__main__" block too
if __name__ == "__main__":
    
    from precompute_driver import precompute_folds
    from DatasetGeneration import process_settings_files
    
    current_default_file = "ALL_EXPERIMENT_DSETS/refactor_default_tst.json"
    current_settings_file = "ALL_EXPERIMENT_DSETS/base_dset_expanded_10000_cc_reordered/dset_settings.json"
    s_obj, opts = process_settings_files(current_settings_file, current_default_file)
    
    assert(s_obj.top_level_fold_path == "ALL_EXPERIMENT_DSETS/base_dset_expanded_10000_cc_reordered")
    assert(s_obj.spline_deg == 5)
    
    print("Beginning precomputation process for base_dset_expanded_10000_cc_reordered")
    
    precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)