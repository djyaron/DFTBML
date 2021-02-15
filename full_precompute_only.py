# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 23:27:51 2021

@author: Frank

NOTE: PSC RUN FILE

The only purpose of this file is to run on a psc environment and 
perform the precompute stage using all the molecules
"""
import json
from fold_generator import compute_graphs_from_folds, Settings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("settings", help = "Name of settings file to use in prcomputation stage")
parser.add_argument("fold_direc", help = "Name of the top level directory containing all the molecules")


if __name__ == "__main__":
    args = parser.parse_args()
    settings_file_name = args.settings
    fold_directory_name = args.fold_direc
    
    with open(settings_file_name, 'r') as handle:
        settings = json.load(handle)
    
    settings_obj = Settings(settings)
    compute_graphs_from_folds(settings_obj, fold_directory_name, copy_molecs = True)
    pass
