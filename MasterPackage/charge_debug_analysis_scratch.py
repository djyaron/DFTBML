# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:20:35 2021

@author: fhu14
"""
"""
Script for debugging inconsistencies in the charge sign. 

First thing's first, going to see if the predictions from DFTB+ agree with
the saved spline models. Going to use saved_model_driver_2.py as the 
method for passing in the saved spline models. 

Going to go step-by-step debugging as usual
"""


#%% Comparison of saved spline charges with DFTB+ output

from saved_model_driver_2 import *
#for the saved_model_driver_2 import, make sure the global variables are all
#   set
import numpy as np
import os, shutil

DFTBEXEC = os.path.join(os.getcwd(), "../../../dftbp/dftbplus-21.1.x86_64-linux/bin/dftb+")

mod_filename = "PaperPackage/master_dset_reduced_300_300_epoch_run/Split0/saved_models.p"
ref_param_filename = "PaperPackage/master_dset_reduced_300_300_epoch_run/ref_params.p"
skf_dir = os.path.join(os.getcwd(), "PaperPackage/master_dset_reduced_300_300_epoch_run")

all_batches = pass_feeds_through(mod_filename, ref_param_filename, True)
all_mols = list(reduce(lambda x, y : x + y, all_batches))
add_dftb(all_mols, skf_dir, DFTBEXEC, par_dict, do_our_dftb = False, do_dftbplus = True, parse = 'detailed')

#Compare the charges
abs_charge_err = []
for mol in all_mols:
    abs_charge_err.extend(np.abs(mol['pzero']['charges'] - mol['predictions']['charges']))

print(f"The MAE for charges is {sum(abs_charge_err) / len(abs_charge_err)}")


