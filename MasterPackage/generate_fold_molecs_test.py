# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:28:53 2021

@author: fhu14
"""

#%% Imports, definitions
import pickle, os, re
from InputParser import parse_input_dictionaries, collapse_to_master_settings
from FoldManager import compute_graphs_from_folds

#%% Code behind

if __name__ == "__main__":
    #Copy over the right number of molecules into the test folder
    
    src_name = "fold_molecs"
    
    dest_dir = "fold_molecs_test"
    
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
        
    pattern = r"Fold[0-9]+_molecs.p"
    
    valid_names = list(filter(lambda x : re.match(pattern, x), os.listdir(src_name)))
    
    for name in valid_names:
        full_name = os.path.join(src_name, name)
        molecs = pickle.load(open(full_name, 'rb'))
        new_molecs = molecs[:200]
        dest_name = os.path.join(dest_dir, name)
        with open(dest_name, 'wb') as handle:
            pickle.dump(new_molecs, handle)
    
    #Perform the precompute on this subset of molecules
    
    settings_filename = "settings_refactor_tst.json"
    default_filename = "refactor_default_tst.json"
    
    s = collapse_to_master_settings(parse_input_dictionaries(settings_filename, default_filename))
    
    compute_graphs_from_folds(s, dest_dir, True)
    
    print("fold_molecs_test generated")