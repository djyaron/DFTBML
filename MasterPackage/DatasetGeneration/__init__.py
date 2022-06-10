# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:57:04 2022

@author: fhu14

This is a rework of the dataset_gen_script.py used in previous versions of 
PaperPackage. That code was inelegant and messy, and it is intended that this
packaged approach is more comprehensive, concise, and maintainable

The first set of datasets will all be generated using the 'cc' energy target.
From this first set of 'cc' datasets, the 'wt' datasets will be generated
by basically inheriting the same molecular configurations just with a different
target. This workflow reduces the number of calls to the interface 
get_ani1data which can be time consuming
"""

from .generate_master_dataset import generate_master_dset, convert_dataset_to_dictionary
from .create_training_dataset import generate_dset, create_larger_dataset, create_smaller_training_set,\
    create_comparative_datasets, create_transfer_dataset, copy_dset
from .util import test_strict_molecule_set_equivalence, name_config_nonoverlap, name_nonoverlap,\
    count_nheavy_empirical_formula, test_strict_molecule_nonequivalence
from .dataset_diagnostics import obtain_formula_distribution, plot_distribution
from .dataset_precomputation import populate_settings_files, check_dataset_paths,\
    precompute_datasets, perform_precompute_settings_check, process_settings_files
