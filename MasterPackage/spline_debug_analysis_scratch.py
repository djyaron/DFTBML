# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 19:28:33 2021

@author: fhu14
"""

# import pickle
# import numpy as np

# mods = pickle.load(open('skf_8020_100knot_new_repulsive_eachepochupdate/Split0/saved_models.p', 'rb'))

# rep = mods['rep']
# grid = np.zeros(50) #Dummy grid, just want to see how Zs are treated in create_xydata()

# rep.mod.create_xydata(grid, True)

##############################################################################

#%% Plotting out splines over distance distributions 

# import pickle, os
# from PlottingUtil import plot_skf_dist_overlay, read_skf_set
# from DFTBrepulsive import SKFSet

# mols0 = pickle.load(open("comparison_dset_2/Fold0_molecs.p", "rb"))
# mols1 = pickle.load(open("comparison_dset_2/Fold1_molecs.p","rb"))
# dset = mols0 + mols1

# skf_path = "comparison_dset_2_result"
# dest = skf_path + "/skf_plots"
# if (not os.path.isdir(dest)):
#     os.mkdir(dest)

# skfset = read_skf_set(skf_path)

# plot_skf_dist_overlay(skfset, dest = None, mode = 'plot', dset = dset)

##############################################################################

#%% Plotting histograms to analyze MAE differences in deviations between the losslessly saved models and the DFTB+ with SKF.

# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# skf_dir_name = "zero_epoch_run"

# path = skf_dir_name + "/saved_model_driver_result.p"

# mols = pickle.load(open(path, 'rb'))
# assert(len(mols) == 2329)

# d = []
# for mol in mols:
#     d.append(abs(mol['pzero']['t'] - mol['predictions']['Etot']['Etot']))

# print(f"MAE in Ha is {sum(d) / len(d)}")

# fig, axs = plt.subplots()
# axs.set_xlabel("Absolute difference (Ha)")
# axs.set_ylabel("Frequency")
# axs.set_title(f"{skf_dir_name} MAE Histogram")
# bins = np.arange(0, 0.2, 0.001)
# axs.hist(d, bins = bins)

# plt.show()

# fig.savefig(os.path.join(os.getcwd(), skf_dir_name, "SKF MAE Plot.png"))


##############################################################################

#%% Analyzing the MAE difference between the spline predictions and DFTBpy predictions using the SKFs

# #Since all the DFTBpy predictions originate from the Auorg SKF files, it follows
# #that all the 'dzero' predictions should be about the same across all 
# #molecules.

# import pickle, os

# skf_dir_names = ["zero_epoch_run", "refacted_joined_spline_run",
#                  "single_epoch_skf_test"]

# #Exclude skf_8020_no_trained_S because it uses a different dataset with a
# #different ordering

# print("Pairwise order check, excluding no S set")

# all_mols_from_sets = []

# for name in skf_dir_names:
#     mol_path = name + "/saved_model_driver_result.p"
#     mols = pickle.load(open(mol_path, "rb"))
#     assert(len(mols) == 2329)
#     all_mols_from_sets.append(mols)

# for i in range(len(all_mols_from_sets)):
#     for j in range(i + 1, len(all_mols_from_sets)):
#         print(i, j)
#         first_mols = all_mols_from_sets[i]
#         second_mols = all_mols_from_sets[j]
#         assert(len(first_mols) == len(second_mols))
#         for k in range(len(first_mols)):
#             assert(first_mols[k]['dzero']['t'] == second_mols[k]['dzero']['t'])
#             assert(first_mols[k]['dzero']['r'] == second_mols[k]['dzero']['r'])
#             assert(first_mols[k]['dzero']['e'] == second_mols[k]['dzero']['e'])
            
# #Do a special safety check for 'skf_8020_no_trained_S' molecules

# print("Pairwise order check, including no S set")

# ref_mol = pickle.load(open("skf_8020_no_trained_S/saved_model_driver_result.p", "rb"))

# ref_mol_dict = {(mol['name'], mol['iconfig']) : mol for mol in ref_mol}

# ordering = [(mol['name'], mol['iconfig']) for mol in all_mols_from_sets[0]]

# ref_mol = [ref_mol_dict[x] for x in ordering]

# for i in range(len(all_mols_from_sets)):
#     curr_mols = all_mols_from_sets[i]
#     assert(len(ref_mol) == len(curr_mols))
#     for j, mol in enumerate(ref_mol):
#         assert(curr_mols[j]['dzero']['t'] == ref_mol[j]['dzero']['t'])
#         assert(curr_mols[j]['dzero']['r'] == ref_mol[j]['dzero']['r'])
#         assert(curr_mols[j]['dzero']['e'] == ref_mol[j]['dzero']['e'])
    

# #Above safety check ensures that the Auorg predictions for dzero are consistent across 
# #molecules. Now onto the main comparison

# print("Dzero comparisons")

# for name in skf_dir_names + ['skf_8020_no_trained_S']:
#     mol_path = name + "/saved_model_driver_result.p"
#     mols = pickle.load(open(mol_path, "rb"))
#     assert(len(mols) == 2329)
#     disagreements = []
#     for mol in mols:
#         disagreements.append(abs(mol['dzero']['t'] - mol['predictions']['Etot']['Etot']))
#     print(f"MAE in Ha: {name}, {sum(disagreements) / len(disagreements)}")
#     #Also interested in maximum and minimums (to indicate any outliers)
#     print(f"Maximum disagreement: {max(disagreements)}")
#     print(f"Minimum disagreement: {min(disagreements)}")

##############################################################################

#%% Analysis of the breakdown between the electronic and repulsive errors.

# import pickle, os

# skf_dir_name = "single_epoch_skf_test"
# mol_path = skf_dir_name + "/saved_model_driver_result.p"
# mols = pickle.load(open(mol_path, 'rb'))

# r_dis, e_dis = [], []
# for mol in mols:
#     if 'r' in mol['pzero']:
#         r_dis.append(abs(mol['pzero']['r'] - mol['predictions']['Etot']['Erep']))
#     if 'e' in mol['pzero']:
#         e_dis.append(abs(mol['pzero']['e'] - mol['predictions']['Etot']['Eelec']))

# print(skf_dir_name)
# print(f"MAE in Ha for the electronic is {sum(e_dis) / len(e_dis)}")
# print(f"MAE in Ha for the repulsive is {sum(r_dis) / len(r_dis)}")

##############################################################################

#%% Analysis of alternate workflow for reading in, interpolating out, and 
# splicing together trained electronic and non-trained repulsive.

# The pipeline is as follows:
# 1) interpolate the Auorg SKFs to generate models
# 2) write out SKFs from the freshly initialized models
# 3) compare the new SKFs against auorg and then against the lossless models

#Refer to spline_debug.txt for more information
#Because this analysis requires precomputation, the working .json file is 
#analysis_refactor_tst.json
#Going to use the fold_molecs_test_8020 dataset

# import pickle, os
# import numpy as np
# from Precompute import precompute_stage
# from DFTBLayer import DFTB_Layer
# from Training import exclude_R_backprop, sort_gammas_ctracks, charge_update_subroutine
# from InputLayer import DFTBRepulsiveModel, combine_gammas_ctrackers
# from DataManager import load_gammas_per_fold, load_config_tracker_per_fold
# from Auorg_1_1 import ParDict
# from InputParser import parse_input_dictionaries, inflate_to_dict,\
#     collapse_to_master_settings
# from SKF import write_skfs
# from DFTBPlus import add_dftb
# import shutil
# from MasterConstants import atom_nums, atom_masses

# settings_filename = "analysis_refactor_tst.json"
# defaults_filename = "refactor_default_tst.json"
# temp_dir = "temp_skfs"

# if (not os.path.isdir(temp_dir)):
#     os.mkdir(temp_dir)

# #Read in and construct the settings object for hyperparameters.
# s = parse_input_dictionaries(settings_filename, defaults_filename)
# opts = inflate_to_dict(s)
# s = collapse_to_master_settings(s)

# pardict = ParDict() #Going to use the Auorg pardict

# #Load in the molecules
# # mols0 = pickle.load(open("fold_molecs_test_8020/Fold0_molecs.p", "rb"))
# # mols1 = pickle.load(open("fold_molecs_test_8020/Fold1_molecs.p", "rb"))
# # all_mols = mols0 + mols1
# # assert(len(all_mols) == 2329)
# # assert(isinstance(all_mols, list))

# #Obtain the models and other necessary information through precompute
# all_models, model_variables, training_feeds,\
#     validation_feeds, training_dftblsts, validation_dftblsts,\
#         losses, all_losses, loss_tracker, training_batches,\
#             validation_batches = precompute_stage(s, pardict, 0, s.split_mapping, 
#                                                   None, None)

# #Now grab the information required for the repulsive model
# train_gammas, train_c_trackers, valid_gammas, valid_c_trackers = None, None, None, None
# #Exclude the R models if new rep setting
# if s.rep_setting == 'new':
#     exclude_R_backprop(model_variables)
#     gammas = load_gammas_per_fold(s.top_level_fold_path)
#     c_trackers = load_config_tracker_per_fold(s.top_level_fold_path)
#     train_gammas, train_c_trackers, valid_gammas, valid_c_trackers = sort_gammas_ctracks(s.split_mapping, 0, gammas, c_trackers)
# #Need to initialize the repulsive model
# dftblayer = DFTB_Layer(device = s.tensor_device, dtype = s.tensor_dtype, eig_method = s.eig_method, repulsive_method = s.rep_setting)
    
# validation_losses, training_losses = list(), list()

# times_per_epoch = list()
# print("Running initial charge update")
# charge_update_subroutine(s, training_feeds, training_dftblsts, validation_feeds, validation_dftblsts, all_models)

# gamma_T, c_tracker_T = combine_gammas_ctrackers(train_gammas, train_c_trackers)
# gamma_V, c_tracker_V = combine_gammas_ctrackers(valid_gammas, valid_c_trackers)

# all_models['rep'] = DFTBRepulsiveModel([c_tracker_T, c_tracker_V], 
#                                                    [gamma_T, gamma_V], 
#                                                    s.tensor_device, s.tensor_dtype, s.rep_integration) #Hardcoding mode as 'internal' right now
# print("Obtaining initial estimates for repulsive energies")
# all_models['rep'].initialize_rep_model(training_feeds, validation_feeds, 
#                                         training_batches, validation_batches, 
#                                         dftblayer, all_models, opts, all_losses, 
#                                         s.train_ener_per_heavy)

# #Pickle the models
# with open(os.path.join(os.getcwd(), temp_dir, "saved_models.p"), "wb") as handle:
#     pickle.dump(all_models, handle)

# train_s_block = "S" in s.opers_to_model
# ref_direc = os.path.join(os.getcwd(), "Auorg_1_1", "auorg-1-1")
# write_skfs(all_models, atom_nums, atom_masses, train_s_block, ref_direc, s.rep_setting,
#            dest = temp_dir, spl_ngrid = s.spl_ngrid)

# #Once the SKFs are written out, the next step is to do the saved_model_driver_2.py
# #externally.

##############################################################################

#%% Comparing dzero t target to pzero t target for past saved_model_driver results.

# import pickle

# skf_dir = "zero_epoch_run"
# mols = pickle.load(open(skf_dir + "/saved_model_driver_result.p", "rb"))
# assert(len(mols) == 2329)

# d = []
# for mol in mols:
#     d.append(abs(mol['pzero']['t'] - mol['dzero']['t']))

# print(f"MAE in Ha for {skf_dir}: {sum(d) / len(d)}")

##############################################################################

#%% Framework for testing losslessly saved model predicted values against dftbpy for figuring out this interpolation error.

#Through testing, DFTB+ and DFTBpy appear to agree in terms of the values 
# predicted for operator elements. As such, debugging against DFTBpy is a good
# approach since we can delve into that code.

#What we want to compare is the values predicted for the operator elements from DFTBpy,
#   which is contained in mod_raw. Mod_raw is already saved, in the feed h5 files,
#   so realistically all you need to do is compare the predictions at the same distances 
#   from the losslessly saved models to the dftbpy mod_raw values.

#The mod_raw values come from the DFTBpy backend. The workflow for the mod_raw
#   is as follows:
#       1) a batch object is created with a par_dict
#       2) When DFTB calculations are invoked with the batch passed in, the 
#           values calculated in matelem are added to the batch through
#           invoking add_raw, which adds to self.raw_data.
#       3) When get_mod_raw() is invoked, the values in RawData are returned. 
#   Thus it follows that values in mod_raw are a valid representation of the 
#   operator element values predicted by DFTBpy using par_dict at 
#   various distances.

#TODO: Need to do a precompute with the molecules because the mod_raw values
#   contained come from the auorg pardict. Not the same as the in-house SKF 
#   pardict, so this needs to be rectified. Normally, mod_raw values do not
#   matter and only the distances matter, so good to have caught this.
#   
#   Might also be good to pull the saved models from skf_full_pairwise_linear 
#   to see if not using the cutoff trick is helpful.


import pickle, os
import shutil #To remove any temp files later
from DataManager import total_feed_combinator
from InputLayer import Input_layer_pairwise_linear, Input_layer_hubbard
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from MasterConstants import Model
from DFTBpy import _Gamma12

#no_convex_run and fold_molecs_test_8020_inhouse must go together since they are
#   intertwined by being the same parameter dictionary
skf_dir = "no_convex_run"
dataset_dir = "fold_molecs_test_8020_inhouse"

model_name = skf_dir + "/Split0/saved_models.p"
fold0_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold0/batches.h5", dataset_dir + "/Fold0/molecs.h5")
fold1_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold1/batches.h5", dataset_dir + "/Fold1/molecs.h5")

assert(len(fold0_feeds) == 187)
assert(len(fold1_feeds) == 47)

saved_models = pickle.load(open(model_name, "rb"))

all_feeds = fold0_feeds + fold1_feeds

print("data loading successful")

disagreements = []
model_spec_tracker = []

cond1 = []
cond2 = []
cond3 = []

#Going to ignore the repulsive because we only want the electronic Hamiltonian (H),
#   and overlap (S)
for feed in all_feeds:
    for model_spec in feed['models']:
        if (model_spec.oper not in ["G", "R"]) and (len(model_spec.Zs) == 2): #Only interested in hamiltonian and overlap
            # print(model_spec)
            curr_mod_raw = feed['mod_raw'][model_spec]
            
            curr_model = saved_models[model_spec]
            assert(isinstance(curr_model, Input_layer_pairwise_linear))
            # feed_dict = curr_model.get_feed(curr_mod_raw)
            # for key in feed_dict:
            #     if (feed_dict[key] is not None):
            #         feed_dict[key] = torch.tensor(feed_dict[key])
            rgrid = np.array([elem.rdist for elem in curr_mod_raw])
            dgrids_consts = curr_model.pairwise_linear_model.linear_model(rgrid, 0)
            model_variables = curr_model.get_variables().detach().cpu().numpy()
            pred = np.dot(dgrids_consts[0], model_variables) + dgrids_consts[1]
            
            assert(len(pred) == len(curr_mod_raw))
            assert(all([elem.Zs == model_spec.Zs for elem in curr_mod_raw]))
            mod_raw_dftb = [elem.dftb for elem in curr_mod_raw]
            pred_np = pred
            mod_raw_dftb = np.array(mod_raw_dftb)
            MAE = np.mean(np.abs(pred_np - mod_raw_dftb)) #Compare mod_raw to predictions
            disagreements.append(MAE)
            model_spec_tracker.append(model_spec)
            cond1.append(model_spec)
        else: #G and on-diagonal models need to go through get_values() pipeline
            curr_model = saved_models[model_spec]
            if (model_spec.oper == "G") and (len(model_spec.Zs) == 1) and (len(model_spec.orb) == 2)\
                and (model_spec.orb[0] != model_spec.orb[1]):
                curr_mod_raw = feed['mod_raw'][model_spec]
                hub1_mod = Model("G", model_spec.Zs, model_spec.orb[0] * 2)
                hub2_mod = Model("G", model_spec.Zs, model_spec.orb[1] * 2)
                hub1 = saved_models[hub1_mod].get_variables()[0].detach().item()
                hub2 = saved_models[hub2_mod].get_variables()[0].detach().item()
                result = np.repeat(_Gamma12(0.0, hub1, hub2), len(curr_mod_raw))
                mod_raw_dftb = np.array([elem.dftb for elem in curr_mod_raw])
                MAE = np.mean(np.abs(result - mod_raw_dftb))
                disagreements.append(MAE)
                model_spec_tracker.append(model_spec)
                #Models of the form Model(G, (6, ), 'ps')
                cond2.append(model_spec)
            else:
                curr_mod_raw = feed['mod_raw'][model_spec]
                feed_dict = curr_model.get_feed(curr_mod_raw)
                for key in feed_dict:
                    if (feed_dict[key] is not None):
                        feed_dict[key] = torch.tensor(feed_dict[key])
                pred = curr_model.get_values(feed_dict)
                mod_raw_dftb = np.array([elem.dftb for elem in curr_mod_raw])
                pred_np = pred.detach().numpy()
                MAE = np.mean(np.abs(pred_np - mod_raw_dftb))
                disagreements.append(MAE)
                model_spec_tracker.append(model_spec)
                #Models of the form Model(G/H, (
                cond3.append(model_spec)
            

print(f"The average of MAE disagreements for all models is {sum(disagreements) / len(disagreements)}")
def plot_disagreements(x):
    fig, axs = plt.subplots()
    bins = np.arange(0, 0.01, 0.001)
    axs.hist(x, bins = bins)
    axs.set_title("MAE disagreements per model against mod_raw values")
    axs.set_ylabel("frequency")
    axs.set_xlabel("MAE (Ha)")
    plt.show()

plot_disagreements(disagreements)

#Further analysis, figure out the MAE per model
model_dict = dict()
for val, mod_spec in zip(disagreements, model_spec_tracker):
    if mod_spec not in model_dict:
        model_dict[mod_spec] = [val]
    else:
        model_dict[mod_spec].append(val)

# assert(len(model_dict) == 68)

MAE_dict = dict()
for key in model_dict:
    model_dict[key] = np.array(model_dict[key])
    MAE_dict[key] = np.mean(model_dict[key])
    
#Plot histograms of the errors for each of the models in model_dict
for model in model_dict:
    fig, axs = plt.subplots()
    data = model_dict[model]
    step = 0.001
    bins = np.arange(min(data) - step, max(data) + step, step)
    axs.hist(data, bins)
    axs.set_xlabel("MAE (Ha)")
    axs.set_title(f"MAE disagreements for {model}")
    axs.set_ylabel("Frequency")
    plt.show()

##############################################################################

#%% Comparing interpolated univariate spline against losslessly saved models

import pickle, os
import shutil #To remove any temp files later
from DataManager import total_feed_combinator
from InputLayer import Input_layer_pairwise_linear, Input_layer_hubbard
import torch
import numpy as np
import TestSKF
from Spline import get_dftb_vals
import matplotlib.pyplot as plt

#no_convex_run and fold_molecs_test_8020_inhouse must go together since they are
#   intertwined by 
skf_dir = 'no_convex_run'
dataset_dir = "fold_molecs_test_8020_inhouse"

model_name = skf_dir + "/Split0/saved_models.p"
#Usually only two folds are present in the dataset
fold0_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold0/batches.h5", dataset_dir + "/Fold0/molecs.h5")
fold1_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold1/batches.h5", dataset_dir + "/Fold1/molecs.h5")
assert(len(fold0_feeds) == 187)
assert(len(fold1_feeds) == 47)

saved_models = pickle.load(open(model_name, 'rb'))
all_feeds = fold0_feeds + fold1_feeds
par_dict = TestSKF.ParDict()
print("data loading successful")

disagreements = []
model_spec_tracker = []

for feed in all_feeds:
    for model_spec in feed['models']:
        if (model_spec.oper not in ["R", "G"]) and (len(model_spec.Zs) == 2):
            curr_mod_raw = feed['mod_raw'][model_spec]
            
            curr_model = saved_models[model_spec]
            assert(isinstance(curr_model, Input_layer_pairwise_linear))
            feed_dict = curr_model.get_feed(curr_mod_raw)
            for key in feed_dict:
                if (feed_dict[key] is not None):
                    feed_dict[key] = torch.tensor(feed_dict[key])
            pred = curr_model.get_values(feed_dict)
            
            dists = np.array([elem.rdist for elem in curr_mod_raw])
            ygrid = get_dftb_vals(model_spec, par_dict, dists)
            # fig, axs = plt.subplots()
            # axs.plot(dists, pred, label = "lossless model")
            # axs.plot(dists, ygrid, label = "interpolated univariate spline")
            # axs.set_xlabel("Angstroms")
            # axs.legend()
            # plt.show()
            pred_np = pred.detach().numpy()
            MAE = np.mean(np.abs(pred_np - ygrid))
            disagreements.append(MAE)

print(f"The average of MAE disagreements for all models is {sum(disagreements) / len(disagreements)}")

##############################################################################

#%% Testing interpolated univariate spline between knots

#This cell is mostly to allow for debug mode to be used with skf_interpolation_plot

from PlottingUtil import skf_interpolation_plot

skf_dir = "no_convex_run"
mode = "plot"
dest = None

skf_interpolation_plot(skf_dir, mode, dest)


##############################################################################
#%% Testing lossless model and interpolated univariate spline agreement on skf knots

#   and off SKF knots (in between the gridpoints of the SKF files)

#In theory, the fifth degree interpolated univariate spline and the 
#   losslessly saved models should agree on the knots. Since 
#   get_dftb_vals generates the same values as the precomputed 
#   dataset mod_raw values, it's a useful function for sidestepping 
#   new precomputes.

import pickle, os
import shutil #To remove any temp files later
from DataManager import total_feed_combinator
from InputLayer import Input_layer_pairwise_linear, Input_layer_hubbard
import torch
import numpy as np
import TestSKF
from Spline import get_dftb_vals
import matplotlib.pyplot as plt
from DFTBrepulsive import SKFSet
from MasterConstants import Model, ANGSTROM2BOHR, RawData
from typing import List

def empty_int(int_series) -> bool:
    r"""Tests if an integral is zero or not
    
    Arguments:
        int_series (Series): A pandas series that represents the data for a 
            given integral
    
    Returns:
        Whether the integral is empty or not. True for empty false for not empty.
    
    Notes: An integral is empty if its maximum and minimum value are both
        0
    """
    return int_series.max() == int_series.min() == 0

def extract_orb(integral_name: str) -> str:
    r"""Given an integral label, extracts the involved orbitals from the 
        interaction.
        
    Arguments: 
        integral_name (str): The name of the integral
    
    Returns:
        orb (str): The orbital type
    
    Example:
        name = "Spp0"
        orb = extract_elem(name)
        orb == "pp_sigma"
    """
    assert(len(integral_name) == 4)
    op, orb, orb_num = integral_name[0], integral_name[1:-1], int(integral_name[-1])
    ind_shell = ['pp']
    if orb in ind_shell:
        orb = orb + "_sigma" if orb_num == 0 else orb + "_pi"
    return orb

def construct_dummy_raw(rgrid) -> List[RawData]:
    r"""Constructs a list of dummy raw_data values where only rdist values matter.
    
    Arguments:
        rgrid (np.ndarray): The array of grid distances. Distances should be 
            in angstroms.
    
    Returns:
        dummy_raw (List[RawData]): The list of dummy raw data.
    """
    dummy_raw = list()
    for dist in rgrid:
        new_raw = RawData(0, 0, 0, 0, 0, 0, 0, dist)
        dummy_raw.append(new_raw)
    return dummy_raw

skf_dir = 'no_convex_run'
dataset_dir = "fold_molecs_test_8020_inhouse"

model_name = skf_dir + "/Split0/saved_models.p"
#Usually only two folds are present in the dataset
# fold0_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold0/batches.h5", dataset_dir + "/Fold0/molecs.h5")
# fold1_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold1/batches.h5", dataset_dir + "/Fold1/molecs.h5")
# assert(len(fold0_feeds) == 187)
# assert(len(fold1_feeds) == 47)

saved_models = pickle.load(open(model_name, 'rb'))
# all_feeds = fold0_feeds + fold1_feeds
par_dict = TestSKF.ParDict()
print("data loading successful")

skfset = SKFSet.from_dir(skf_dir)
assert(len(skfset.keys()) == len(par_dict.keys()))
all_ops = ["H", "S"]

disagreements = []

for elem_pair in skfset.keys():
    curr_skf = skfset[elem_pair]
    for op in all_ops:
        for int_name in getattr(curr_skf, op).keys():
            if not (empty_int(getattr(curr_skf, op)[int_name])):
                rgrid = (curr_skf.intGrid() / ANGSTROM2BOHR) + (0.01 / ANGSTROM2BOHR) #rgrid is in Angstroms
                orb = extract_orb(int_name)
                model = Model(op, elem_pair, orb)
                model_rev = Model(op, (elem_pair[-1], elem_pair[0]), orb)
                #ygrid is the result returned by InterpolatedUnivariateSpline
                ygrid = get_dftb_vals(model, par_dict, rgrid)
                
                #Now to get the values from the losslessly saved models
                curr_model = saved_models[model] if (model in saved_models) else saved_models[model_rev]
                assert(isinstance(curr_model, Input_layer_pairwise_linear))
                dgrids_consts = curr_model.pairwise_linear_model.linear_model(rgrid, 0)
                model_variables = curr_model.get_variables().detach().cpu().numpy()
                y_vals = np.dot(dgrids_consts[0], model_variables) + dgrids_consts[1]
                
                assert(len(y_vals) == len(ygrid))
                MAE = np.mean(np.abs(y_vals - ygrid))
                print(MAE)
                disagreements.append(MAE)

##############################################################################
#%% Comparison of coulomb (G) models for on-atom p s mixing

#With the above spline fixings, the only thing that's different now are the 
#   G models for on-atom p and s orbital interactions. For example, 
#   Model (G, (8,), sp)

#This debugging is straightforward, compare the results from get_dftb_vals
#   and the values predicted by the models. This might be a problem with 
#   how values are written out.

import pickle, os
from Spline import get_dftb_vals
from TestSKF import ParDict
from MasterConstants import Model
from DataManager import total_feed_combinator
import torch
import numpy as np

skf_dir = "no_convex_run"
dataset_dir = "fold_molecs_test_8020_inhouse"
pardict = ParDict()

model_name = skf_dir + "/Split0/saved_models.p"
saved_models = pickle.load(open(model_name, 'rb'))
fold0_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold0/batches.h5", dataset_dir + "/Fold0/molecs.h5")
fold1_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold1/batches.h5", dataset_dir + "/Fold1/molecs.h5")

assert(len(fold0_feeds) == 187)
assert(len(fold1_feeds) == 47)
all_feeds = fold0_feeds + fold1_feeds

#The idea is that if this problem is fixed for this model then it will be fixed
#   for all of them.
model_spec = Model("G", (7,), 'ps')
curr_model = saved_models[model_spec]

for feed in all_feeds:
    if model_spec in feed['models']:
        curr_mod_raw = feed['mod_raw'][model_spec]
        
        feed_dict = curr_model.get_feed(curr_mod_raw)
        for key in feed_dict:
            if (feed_dict[key] is not None):
                feed_dict[key] = torch.tensor(feed_dict[key])
        pred = curr_model.get_values(feed_dict)
        pred_np = pred.detach().numpy()
        
        true_vals = get_dftb_vals(model_spec, pardict, np.array([elem.rdist for elem in curr_mod_raw]))

##############################################################################
#%% Distance analysis (checking how many distance values are requested above cutoff of 3.5)

import pickle, os
from Spline import get_dftb_vals
from TestSKF import ParDict
from MasterConstants import Model
from DataManager import total_feed_combinator
import torch
import numpy as np

dataset_dir = "fold_molecs_test_8020_inhouse"

fold0_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold0/batches.h5", dataset_dir + "/Fold0/molecs.h5")
fold1_feeds = total_feed_combinator.create_all_feeds(dataset_dir + "/Fold1/batches.h5", dataset_dir + "/Fold1/molecs.h5")
assert(len(fold0_feeds) == 187)
assert(len(fold1_feeds) == 47)
all_feeds = fold0_feeds + fold1_feeds

cutoff = 3.5

num_greater = 0
num_total = 0
for feed in all_feeds:
    for mod_spec in feed['models']:
        if mod_spec.oper != 'R':
            mod_raw = feed['mod_raw'][mod_spec]
            num_total += len(mod_raw)
            distances = [elem.rdist for elem in mod_raw]
            for d in distances:
                if d > cutoff:
                    num_greater += 1

##############################################################################
#%% Comparing differences between low cutoff and long cutoff corrected
from PlottingUtil import compare_differences

set1_name = "corrected_model_architecture_run"
set2_name = "higher_cutoff_run_vanishing"
dest = "cmar_v_hcrv_diff_plots"

compare_differences(set1_name, set2_name, dest, "scatter", units = 'kcal')

#%% Creating separated datasets for replicating spline behavior

'''
Need to create two datasets that do not have the same empirical formulas to 
ensure that training on different sets results in the same overall splines.

Process is as follows:
    1) Pull a series of configurations from the ANI1 molecule
    2) Separate into two sets based on molecular formula
    3) Do the fold generation and separate into folds for precomputation
    4) Do the precompute for both sets
    5) Run the mode through the saved data and generate SKF files
    6) Plot for comparison and generate diffs
    
The code here will only handle steps 1 - 4
'''
from FoldManager import get_ani1data
import random
from functools import reduce
import os
import pickle
from InputParser import parse_input_dictionaries, collapse_to_master_settings, inflate_to_dict
from precompute_driver import precompute_folds

#Step 1: get dataset

allowed_Zs = [1,6,7,8]
heavy_atoms = [1,2,3,4,5,6,7,8]
max_config = 15
target = {"Etot" : "cc",
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
ani_path = "ANI-1ccx_clean_fullentry.h5"

dset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, ani_path)

#Step 2: separate the two sets based on empirical formula

mol_form_dict = dict()

for molecule in dset:
    curr_name = molecule['name']
    if curr_name not in mol_form_dict:
        mol_form_dict[curr_name] = [molecule]
    else:
        mol_form_dict[curr_name].append(molecule)

molecule_names = mol_form_dict.keys()
num_to_sample = len(molecule_names) // 2
set1_names = random.sample(molecule_names, num_to_sample)
set2_names = list(set(molecule_names) - set(set1_names))

dset1_raw = [mol_form_dict[name] for name in set1_names]
dset2_raw = [mol_form_dict[name] for name in set2_names]

dset1_final = list(reduce(lambda x, y : x + y, dset1_raw))
dset2_final = list(reduce(lambda x, y : x + y, dset2_raw))

#Do a quick check
dset1_final_names = {mol['name'] for mol in dset1_final}
dset2_final_names = {mol['name'] for mol in dset2_final}

assert(dset1_final_names - dset2_final_names == dset1_final_names)
assert(dset2_final_names - dset1_final_names == dset2_final_names)
assert(dset1_final_names.intersection(dset2_final_names) == set())

#Step 3: Do the precompute generation for the different datasets. Here, the 
#   majority of the data is going to be focused on training rather than validation,
#   going to do a 80-20 split in each dataset.

#Need to save the data first

dest1 = "comparison_dset_1"
dest2 = "comparison_dset_2"

if not os.path.isdir(dest1):
    os.mkdir(dest1)

if not os.path.isdir(dest2):
    os.mkdir(dest2)


prop_train = 0.8

num_train1 = int(prop_train * len(dset1_final))
num_train2 = int(prop_train * len(dset2_final))

train_fold1, valid_fold1 = dset1_final[:num_train1], dset1_final[num_train1:]
train_fold2, valid_fold2 = dset2_final[:num_train2], dset2_final[num_train2:]

with open(os.path.join(os.getcwd(), dest1, "Fold0_molecs.p"), "wb") as handle:
    pickle.dump(train_fold1, handle)

with open(os.path.join(os.getcwd(), dest1, "Fold1_molecs.p"), "wb") as handle:
    pickle.dump(valid_fold1, handle)

with open(os.path.join(os.getcwd(), dest2, "Fold0_molecs.p"), "wb") as handle:
    pickle.dump(train_fold2, handle)

with open(os.path.join(os.getcwd(), dest2, "Fold1_molecs.p"), "wb") as handle:
    pickle.dump(valid_fold2, handle)
    
#Now do the precomputation on the data to generate the information needed
#   for the model

settings_filename = "settings_refactor_tst.json"
defaults_filename = "refactor_default_tst.json"

resulting_settings_obj = parse_input_dictionaries(settings_filename, defaults_filename)
opts = inflate_to_dict(resulting_settings_obj)
s = collapse_to_master_settings(resulting_settings_obj)

#Do the precompute for each one
precompute_folds(s, opts, dest1, True)
precompute_folds(s, opts, dest2, True)

#%% Create dataset that includes more underrepresented pairwise interactions

from PlottingUtil import plot_distance_histogram
from FoldManager import get_ani1data_boosted
import os
import pickle
from PlottingUtil import plot_distance_histogram

allowed_Zs = [1,6,7,8]
heavy_atoms = [i + 1 for i in range(8)]
target_atoms = [7, 8]
max_config = 8
boosted_config = 15
target = {"Etot" : "cc",
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
ani1_path = "ANI-1ccx_clean_fullentry.h5"

all_mols = get_ani1data_boosted(allowed_Zs, heavy_atoms, target_atoms, 'all',
                                max_config, boosted_config, target, ani1_path)


prop_train = 0.8
num_train = int(prop_train * len(all_mols))

with open("representative_dataset/Fold0_molecs.p", "wb") as handle:
    pickle.dump(all_mols[:num_train], handle)

with open("representative_dataset/Fold1_molecs.p", 'wb') as handle:
    pickle.dump(all_mols[num_train:], handle)

plot_distance_histogram("representative_dataset", None)

#%% Combining the underlying molecules to get accurate exclusion benchmark

#In comparing the performance for resulting SKF files obtained from training
#   on different datasets, it's important to have the same testing set for 
#   both. Otherwise, results are not comparable. 

#To do this, simply combine the molecules together for each comparison dataset 
#   and use that master directory as the exclusion directory when using the 
#   run_organics_script.py.

import pickle, os

dset_1 = "comparison_dset_1"
dset_2 = "comparison_dset_2"
total_mol_dir = "cd1_cd2_total"

comp_dset_1_mols = pickle.load(open(os.path.join(dset_1, "Fold0_molecs.p"), 'rb'))\
    + pickle.load(open(os.path.join(dset_1, "Fold1_molecs.p"), 'rb'))

comp_dset_2_mols = pickle.load(open(os.path.join(dset_2, "Fold0_molecs.p"), 'rb'))\
    + pickle.load(open(os.path.join(dset_2, "Fold1_molecs.p"), 'rb'))

if (not os.path.exists(total_mol_dir)):
    os.mkdir(total_mol_dir)

with open(os.path.join(total_mol_dir, "Fold0_molecs.p"), "wb") as handle:
    pickle.dump(comp_dset_1_mols, handle)

with open(os.path.join(total_mol_dir, "Fold1_molecs.p"), "wb") as handle:
    pickle.dump(comp_dset_2_mols, handle)

with open(os.path.join(total_mol_dir, "info.txt"), "w+") as handle:
    handle.write(f"Fold0_molecs.p contains all the molecules from {dset_1}\n")
    handle.write(f"Fold1_molecs.p contains all the molecules from {dset_2}")

#%% Testing reproduction for electronic energies

from DFTBPlus import run_organics, load_ani1
import Auorg_1_1
import os, pickle

data_path = "ANI-1ccx_clean_fullentry.h5"
max_config = 13
maxheavy = 8
allowed_Zs = [1,6,7,8]
target = 'cc'
exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
par_dict = Auorg_1_1.ParDict()

skf_dir = os.path.join(os.getcwd(), "Experiments_and_graphs", "comparison_dset_1_result")
ref_params = pickle.load(open('Experiments_and_graphs/comparison_dset_1_result/ref_params.p', 'rb'))

error_Ha, error_Kcal, diffs, mols_1 = run_organics(data_path, max_config, maxheavy, allowed_Zs, target, skf_dir, exec_path, par_dict, 
             'new', dftbrep_ref_params = ref_params, filter_test = True,
             filter_dir = "cd1_cd2_total", parse = 'detailed', dispersion = False, return_dset = True)

skf_dir = os.path.join(os.getcwd(), "Experiments_and_graphs", "comparison_dset_2_result")
ref_params = pickle.load(open('Experiments_and_graphs/comparison_dset_2_result/ref_params.p', 'rb'))

error_Ha, error_Kcal, diffs, mols_2 = run_organics(data_path, max_config, maxheavy, allowed_Zs, target, skf_dir, exec_path, par_dict, 
             'new', dftbrep_ref_params = ref_params, filter_test = True,
             filter_dir = "cd1_cd2_total", parse = 'detailed', dispersion = False, return_dset = True)

assert(len(mols_1) == len(mols_2))
elec_disagreements = []
for mol1, mol2 in zip(mols_1, mols_2):
    if ('e' in mol1['pzero']) and ('e' in mol2['pzero']):
        elec_disagreements.append(abs(mol1['pzero']['e'] - mol2['pzero']['e']))
        
total_disagreements = []
for mol1, mol2 in zip(mols_1, mols_2):
    total_disagreements.append(abs(mol1['pzero']['t'] - mol2['pzero']['t']))

print(f"Average electronic energy disagreement in kcal/mol: {(sum(elec_disagreements) / len(elec_disagreements)) * 627}")
print(f"Average disagreement for total energy in kcal/mol: {(sum(total_disagreements) / len(total_disagreements)) * 627}")

#The diffs that are returned are direct differences, need to take abs value of diffs for comparison
#Bad index 625 when max_config = 7 using fmt _8020
#Bad index shifts to 936 when max_config = 8 using fmt_8020

#Analysis of agreement between the electronic components of energies

#%% Analyzing reproduction of 'cc' energies

'''
The individual components of the energy (Eelec, Erep, Eref) disagree, but 
want to see if 'cc' predictions for molecules is in agreement between
different SKF sets trained on different molecules. 

If the energies are in agreement, then it's clear the underlying physics
is not perfectly reproduced, but there is some interplay between the different
energy components. If the energies are not in agreement, then we may have a 
problem. However, the MAE performance between the two comparison dataset result
skf sets is basically the same (difference of a few hundreths kcal), so 
will be interesting to see what the result is. Should agree, I hope.
'''
from DFTBPlus.run_dftbplus import generate_linear_ref_mat
from DFTBPlus import run_organics
import Auorg_1_1
import os, pickle
import numpy as np

data_path = "ANI-1ccx_clean_fullentry.h5"
max_config = 13
maxheavy = 8
allowed_Zs = [1,6,7,8]
target = 'cc'
exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
par_dict = Auorg_1_1.ParDict()

skf_dir = os.path.join(os.getcwd(), "Experiments_and_graphs", "comparison_dset_1_result")
ref_params = pickle.load(open('Experiments_and_graphs/comparison_dset_1_result/ref_params.p', 'rb'))

error_Ha, error_Kcal, diffs, mols_1 = run_organics(data_path, max_config, maxheavy, allowed_Zs, target, skf_dir, exec_path, par_dict, 
             'new', dftbrep_ref_params = ref_params, filter_test = True,
             filter_dir = "cd1_cd2_total", parse = 'detailed', dispersion = False, return_dset = True)

skf_dir = os.path.join(os.getcwd(), "Experiments_and_graphs", "comparison_dset_2_result")
ref_params = pickle.load(open('Experiments_and_graphs/comparison_dset_2_result/ref_params.p', 'rb'))

error_Ha, error_Kcal, diffs, mols_2 = run_organics(data_path, max_config, maxheavy, allowed_Zs, target, skf_dir, exec_path, par_dict, 
             'new', dftbrep_ref_params = ref_params, filter_test = True,
             filter_dir = "cd1_cd2_total", parse = 'detailed', dispersion = False, return_dset = True)

assert(len(mols_1) == len(mols_2))

#Now going to generate the predicted cc energies for both molecule sets
atypes = tuple(allowed_Zs)
XX_1 = generate_linear_ref_mat(mols_1, atypes)
XX_2 = generate_linear_ref_mat(mols_2, atypes)
ref_1 = pickle.load(open('Experiments_and_graphs/comparison_dset_1_result/ref_params.p', 'rb'))
ref_2 = pickle.load(open('Experiments_and_graphs/comparison_dset_2_result/ref_params.p', 'rb'))
coef_1, coef_2 = ref_1['coef'], ref_2['coef']
intercept_1, intercept_2 = ref_1['intercept'], ref_2['intercept']

predicted_dt_1 = np.array([molec['pzero']['t'] for molec in mols_1])
predicted_dt_2 = np.array([molec['pzero']['t'] for molec in mols_2])

cc_1 = predicted_dt_1 + (np.dot(XX_1, coef_1) + intercept_1)
cc_2 = predicted_dt_2 + (np.dot(XX_2, coef_2) + intercept_2)

MAE_diff = np.mean(np.abs(cc_1 - cc_2))
print(f"MAE difference in cc energies is {MAE_diff * 627} in kcal/mol")


#%% Testing out range-constrained plotting
from PlottingUtil import read_skf_set, plot_overlay_skf_sets, compare_differences, plot_skf_dist_overlay, plot_multi_overlay_skf_sets
import os
import pickle

skset1 = os.path.join(os.getcwd(), "comparison_dset_1_last_only_bcond_result") #Vanishing boundary conditions
skset2 = os.path.join(os.getcwd(), "comparison_dset_2_last_only_bcond_result")

range_dict = {
    "1,1" : 0.500,
    "6,6" : 1.04,
    "1,6" : 0.602,
    "7,7" : 0.986,
    "6,7" : 0.948,
    "1,7" : 0.573,
    "1,8" : 0.599,
    "6,8" : 1.005,
    "7,8" : 0.933,
    "8,8" : 1.062
    }

universal_ceil = 4.5

range_dict = {(int(k[0]), int(k[2])) : (v, universal_ceil) for k, v in range_dict.items()}

print(range_dict)

dest = None

# plot_overlay_skf_sets(skset1, skset2, 'cdr1', 'cdr2', None, 'plot', range_dict = range_dict)
# compare_differences(os.path.join(os.getcwd(), "comparison_dset_1_identical_run_1"),
#                     os.path.join(os.getcwd(), "comparison_dset_1_identical_run_2"),
#                     dest, 'plot', units = 'kcal', range_dict = range_dict)
# dset = pickle.load(open("comparison_dset_1/Fold0_molecs.p", "rb")) + pickle.load(open("comparison_dset_1/Fold1_molecs.p", "rb"))
# assert(len(dset) == 2210 + 553)
# plot_skf_dist_overlay(skset1, dest, 'plot', dset, range_dict = range_dict)

# names = ["comparison_dset_1_identical_run_2", "Auorg_1_1/auorg-1-1", "MIO_0_1/mio-0-1"]
# labels = ["DFTBML", "Auorg", "MIO"]
# plot_multi_overlay_skf_sets(names, labels, dest, 'plot', range_dict = range_dict)

compare_differences(skset1, skset2, None,
                        "plot", units = "kcal", range_dict = range_dict)

#%% Generating transfer_dataset that is train on < 4/6 heavy atoms, validate + test on heavy

'''One of the biggest questions is can we train the model on molecules with
< 4 heavy atoms and achieve good results when validating/testing on molecules
withb > 4 heavy atoms? Let's see! This will generate the transfer_dataset.

In this construction, Fold0_molecs.p will be the light training molecules
and Fold1_molecs.p will be the heavy validation molecules. 
'''

from FoldManager import get_ani1data, count_nheavy
import random
import os, shutil
import pickle

allowed_Zs = [1,6,7,8]
heavy_atoms = [i + 1 for i in range(8)]
max_config = 23
target = {"Etot" : "cc",
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
data_path = "ANI-1ccx_clean_fullentry.h5"

dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, data_path)
nums = list(map(lambda x : count_nheavy(x), dataset))

assert(len(dataset) == len(nums))

lower_limit = 6
prop_train = 0.8
destination = "transfer_dataset_6_8"

lower_molecs, higher_molecs = [], []
for count, molec in zip(nums, dataset):
    if count <= lower_limit:
        lower_molecs.append(molec)
    else:
        higher_molecs.append(molec)

for molec in lower_molecs:
    assert(count_nheavy(molec) <= lower_limit)

for molec in higher_molecs:
    assert(count_nheavy(molec) > lower_limit)

num_total = int(len(lower_molecs) / prop_train)
print(f"Total number of molecules is {num_total}")
num_higher = num_total - len(lower_molecs)
final_higher = random.sample(higher_molecs, num_higher)
print(f"Total number of higher molecules is {len(final_higher)}")

assert(len(final_higher) + len(lower_molecs) == num_total)

for molec in lower_molecs:
    assert(count_nheavy(molec) <= lower_limit)

for molec in final_higher:
    assert(count_nheavy(molec) > lower_limit)

total_path = os.path.join(os.getcwd(), destination)
if os.path.exists(total_path):
    print(f"Removing directory {total_path}")
    shutil.rmtree(total_path)

os.mkdir(total_path)

fold0_path = os.path.join(total_path, "Fold0_molecs.p")
fold1_path = os.path.join(total_path, "Fold1_molecs.p")

random.shuffle(lower_molecs)
random.shuffle(final_higher)

with open(fold0_path, 'wb') as handle:
    pickle.dump(lower_molecs, handle)

with open(fold1_path, 'wb') as handle:
    pickle.dump(final_higher, handle)

print("Finished generating transfer_dataset")

#%% Generate super small training set with a fixed validation set

'''
Here, the validation set does not matter because there is no stepping through 
the stochastic gradient descent on the validation predictions. The workflow 
looks as follows:
    1) Copy over the validation set from a previous dataset. This is usually
        the Fold1_molecs.p
    2) Pull a small number of molecules from ani1 and make sure that 
        none of them are contained in the validation set
    3) Save this as the new Fold1_molecs.p along with the copied Fold1_molecs.p.
        Note that this is unlikely to achieve a good 80-20 split, but this doesn't
        matter here.
    4) Precompute for both Fold0 and Fold1 
    5) Run through the model with that dataset
    6) Analyze the results
    
'''
from FoldManager import get_ani1data
import pickle, os, shutil
import random

validation_copy = "comparison_dset_1/Fold1_molecs.p"
copy_validation = pickle.load(open(validation_copy, 'rb'))

validation_name_confs = set( [ (mol['name'], mol['iconfig']) for mol in copy_validation ] )
assert(len(validation_name_confs) == len(copy_validation))

allowed_Zs = [1,6,7,8]
heavy_atoms = [i + 1 for i in range(8)]
max_config = 20
target = {"Etot" : "cc",
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
data_path = "ANI-1ccx_clean_fullentry.h5"

train_dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, data_path)
print(f"Length of total training dataset: {len(train_dataset)}")

train_dset_clean = []
for mol in train_dataset:
    if (mol['name'], mol['iconfig']) not in validation_name_confs:
        train_dset_clean.append(mol)

train_name_confs = set([(mol['name'], mol['iconfig']) for mol in train_dset_clean])
assert(len(train_name_confs) == len(train_dset_clean))
assert(train_name_confs.isdisjoint(validation_name_confs))

num_train = 1_000

training_molecs = random.sample(train_dset_clean, num_train)
train_name_confs = set([(mol['name'], mol['iconfig']) for mol in training_molecs])
assert(len(train_name_confs) == len(training_molecs))
assert(train_name_confs.isdisjoint(validation_name_confs))

dest = "small_train_dset_1000"
full_path = os.path.join(os.getcwd(), dest)
if os.path.exists(full_path):
    print(f"Removing existing directory at {full_path}")
    shutil.rmtree(full_path)

os.mkdir(full_path)

fold0_path = os.path.join(full_path, "Fold0_molecs.p")
fold1_path = os.path.join(full_path, "Fold1_molecs.p")

with open(fold0_path, 'wb') as handle:
    pickle.dump(training_molecs, handle)

with open(fold1_path, 'wb') as handle:
    pickle.dump(copy_validation, handle)
    
print("Finished generating small_dataset")

#%% Charge and dipole improvement analysis
import pickle
lt_path = "comparison_dset_1_identical_run_2/Split0/loss_tracker.p"
lt = pickle.load(open(lt_path, 'rb'))

val_charges = lt['charges'][0]
train_charges = lt['charges'][1]

val_dipoles = lt['dipole'][0]
train_dipoles = lt['dipole'][1]

charge_improvement =  abs(val_charges[-1] - val_charges[0]) / 100 #multiplication factor
dipole_improvement = abs(val_dipoles[-1] - val_dipoles[0]) / 100 #multiplication factor

print(f"charge improvement: {charge_improvement}")
print(f"Initial: {val_charges[0] / 100}, Final: {val_charges[-1] / 100}")
print(f"dipole improvement: {dipole_improvement}")
print(f"Initial: {val_dipoles[0] / 100}, Final: {val_dipoles[-1] / 100}")


#%% Analysis of total loss without convex covering

#All losses are unitless since they are scaled by the reciprocal accuracy factor
#   so this should be easy to see the jumps following charge updates

import matplotlib.pyplot as plt
import pickle
import numpy as np

path = "comparison_dset_1_identical_run_2/Split0/loss_tracker.p"
lt = pickle.load(open(path, 'rb'))
valid_Etot = np.array(lt['Etot'][0])
valid_dipole = np.array(lt['dipole'][0])
valid_charges = np.array(lt['charges'][0])
assert(len(valid_Etot) == len(valid_dipole) == len(valid_charges))

epochs = np.array([i for i in range(len(valid_Etot))])

tot_loss = valid_Etot + valid_dipole + valid_charges
plt.plot(epochs, tot_loss)

#%% Generating a dataset with DFT triple-zeta as the target

'''
The DFT triple-zeta energy is encoded as 'wt' in the backend of the code, so 
we will have to change the target dictionary to accomodate this. Otherwise,
it will be the standard 80-20 split for a dataset. 
'''
from FoldManager import get_ani1data
import pickle, os, shutil
import random

prop_train = 0.8
allowed_Zs = [1,6,7,8]
heavy_atoms = [i + 1 for i in range(8)]
max_config = 6
target = {"Etot" : "wt", #DFT triple-zeta energy target
           "dipole" : "wb97x_dz.dipole",
           "charges" : "wb97x_dz.cm5_charges"}
data_path = "ANI-1ccx_clean_fullentry.h5"

dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, data_path)

print(f"Number of molecules in dataset: {len(dataset)}")
num_train = int(prop_train * len(dataset))
train_dataset = dataset[:num_train]
valid_dataset = dataset[num_train:]

random.shuffle(train_dataset)
random.shuffle(valid_dataset)

print(f"num train: {len(train_dataset)}, num_valid: {len(valid_dataset)}")

with open("DFT_dset/Fold0_molecs.p", "wb") as handle:
    pickle.dump(train_dataset, handle)

with open("DFT_dset/Fold1_molecs.p", "wb") as handle:
    pickle.dump(valid_dataset, handle)

print("Data saved to DFT_dset")


#%% Looking at the third derivative of fifth order splines

from Spline import get_dftb_vals, spline_linear_model
from MasterConstants import *
import numpy as np
import matplotlib.pyplot as plt
from Auorg_1_1 import ParDict

par_dict = ParDict()

tst_model = Model(oper = "H", Zs = (6,6), orb = "ss")
ngrid = 15
rlow, rhigh = 1.04, 4.5
rgrid = np.linspace(rlow, rhigh, ngrid)
ygrid = get_dftb_vals(tst_model, par_dict, rgrid)
bconds = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)]
spline_dict = spline_linear_model(rgrid, None, (rgrid, ygrid), bconds, max_der=4, deg=5)

#Double-check that the predicted values agree with the interpolated values
interpolated_values = np.dot(spline_dict['X'][0], spline_dict['coefs']) + spline_dict['const'][0]
diffs = ygrid - interpolated_values
error = np.mean(np.abs(diffs))
print(f"Initial interpolation error with spline degree 5 is: {error}")

#Now try to analyze the derivatives
interp_second_der = np.dot(spline_dict['X'][2], spline_dict['coefs']) + spline_dict['const'][2]
interp_third_der = np.dot(spline_dict['X'][3], spline_dict['coefs']) + spline_dict['const'][3]


#Emulate the process that a spline model goes through to generate predictions and see what happens

# spline_dict = spline_linear_model(xknots, xeval, xyfit, bconds, max_der=2, deg=3)

# spline_linear_model(rgrid, None, (rgrid, ygrid), bcond)

#%% Reduced framework for analyzing the movement of splines through the code

from Spline import get_dftb_vals, spline_linear_model, SplineModel
from MasterConstants import *
import numpy as np
import matplotlib.pyplot as plt
from Auorg_1_1 import ParDict
import pickle
from InputLayer import Input_layer_pairwise_linear
import torch

def chebyshev_nodes(a, b, n):
    r"""
    Creates a knot sequence following the Chebyshev node sequence on the interval
    [a, b]. Equation is taken from https://en.wikipedia.org/wiki/Chebyshev_nodes
    """
    nodes = []
    operands = [
        (((2 * k) - 1) * np.pi) / (2 * n)\
            for k in range(1, n + 1)
        ]
    second_terms = (0.5 * (b - a)) * np.cos(operands)
    cheby_nodes = (0.5 * (a + b)) + second_terms
    assert(len(cheby_nodes) == n)
    #reverse the cheby_nodes so that the order goes from smallest to largest
    cheby_nodes = np.flip(cheby_nodes)
    #Check that nodes are sorted from low to high
    assert(np.all(np.diff(cheby_nodes) >= 0) )
    return cheby_nodes


#Use some actual results to verify proper spline model construction
saved_mod_path = "cat_results/results/master_dset_expanded_cc_first_half_result/Split0/saved_models.p"
mods = pickle.load(open(saved_mod_path, 'rb'))

tst_model_spec = Model(oper = "H", Zs = (6,6), orb = "ss")
num_knots = 100
target_upper_bound = 4.5
target_knots = 100
rlow, rhigh = (1.04, 4.5)
xknots = np.linspace(rlow, rhigh, num_knots)
remaining_diff = target_upper_bound - xknots[-1]
starting_val = xknots[-1]
knot_spacing = remaining_diff / (target_knots - num_knots)
remaining_knots = []
for i in range(target_knots - num_knots):
    remaining_knots.append(starting_val + ((i + 1) * knot_spacing))

# xknots = np.append(xknots, remaining_knots)

xknots = np.linspace(rlow, rhigh, num_knots)

# TODO: end knot spacings should be played around with (uniform spacing between all knots does not work)

par_dict = ParDict()
device = 'cpu'
dtype = torch.double
tst_bconds = [Bcond(-1, 0, 0), Bcond(-1, 1, 0), Bcond(-1, 2, 0), Bcond(-1, 3, 0),
              Bcond(-1, 4, 0)]
tst_bconds_2 = [Bcond(0, 2, 0), Bcond(-1, 2, 0), 
                Bcond(0, 3, 0), Bcond(-1, 3, 0),
                Bcond(0, 4, 0), Bcond(-1, 4, 0)]
tst_bconds_3 = [Bcond(-1, 0, 0), Bcond(-1, 1, 0), Bcond(-1, 2, 0)]
tst_bcond_5 = [Bcond(-1, 1, 0)]
#Assert that the knot construction is correct
# assert(np.allclose(mods[tst_model_spec].pairwise_linear_model.xknots, xknots))
# assert(xknots[-1] == mods[tst_model_spec].pairwise_linear_model.xknots[-1])
# assert(xknots[0] == mods[tst_model_spec].pairwise_linear_model.xknots[0])
#Check the boundary condition is correct for the trained model for comparison purposes
# assert(mods[tst_model_spec].pairwise_linear_model.bconds == [Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)])

config = {
    'xknots' : xknots,
    'equal_knots' : False, #Do not assert equal knot distributions (only for joined splines I think)
    'cutoff' : xknots[-1],
    'bconds' : 'last_only',
    'deg' : 5, #fourth degree splines
    'max_der' : 4 #asking for the third derivative max
    }

spline = SplineModel(config)
model = Input_layer_pairwise_linear(tst_model_spec, spline, par_dict, config['cutoff'], device, dtype,
                                    )

#Imagine getting the first, second, and third derivative values
xgrid = model.pairwise_linear_model.xknots
# assert(np.allclose(xgrid, xknots))
zero_der = model.pairwise_linear_model.linear_model(xgrid, 0) #len spline_dict['X'] = 5 here
first_der = model.pairwise_linear_model.linear_model(xgrid, 1) #len spline_dict['X'] = 1 here
second_der = model.pairwise_linear_model.linear_model(xgrid, 2) #len spline_dict['X'] = 2 here
third_der = model.pairwise_linear_model.linear_model(xgrid, 3) #len spline_dict['X'] - 3 here
# fourth_der = model.pairwise_linear_model.linear_model(xgrid, 4)
#len_spline_dict['X'] = 4 here

#Assert lengths/assumptions about how linear_model works
assert(len(first_der) == 2)
assert(len(second_der) == 2)
assert(len(third_der) == 2)

#At this point, we can start looking at the evaluation of the derivatives. In the code, this is done
#   through the ModelPenalty class method
m = torch.nn.ReLU() #use the ReLU activation to filter out positive/negative values
c = model.get_variables().detach().numpy()
der2_A, der2_b = second_der[0], second_der[1]
pred_second_der = np.dot(der2_A, c) + der2_b

der3_A, der3_b = third_der[0], third_der[1]
pred_third_der = np.dot(der3_A, c) + der3_b

der0_A, der0_b = zero_der[0], zero_der[1]
pred_zero_der = np.dot(der0_A, c) + der0_b

pred_first_der = np.dot(first_der[0], c) + first_der[1]

# pred_fourth_der = np.dot(fourth_der[0], c) + fourth_der[1]

for elem in [ pred_second_der, pred_third_der, pred_zero_der, pred_first_der]: #2, 3, 0, 1
    fig, axs = plt.subplots()
    axs.scatter(xgrid, elem)
    # for val in xknots:
    #     axs.axvline(x = val)
    axs.set_title("Using boundary condition last only")
    plt.show()

# with open("tmp_result_fifth_order.p", "wb") as handle:
#     pickle.dump(pred_second_der, handle)

# Comparing 2nd derivative predictions for fifth order versus third order
# with open("tmp_result_fifth_order.p", "rb") as handle:
#     fifth_order_2nd_der = pickle.load(handle)
    
#     fig, axs = plt.subplots()
#     axs.plot(pred_second_der, label = "Third order")
#     axs.plot(fifth_order_2nd_der, label = "Fifth order")
#     axs.legend()

#%% Comparing the second derivative of a trained model against the initial model (should be used in tandem with previous cell)
trained_model = mods[tst_model_spec]
#Set xgrid to xknots, as is done in the form penalty
xgrid = trained_model.pairwise_linear_model.xknots
trained_second_der = trained_model.pairwise_linear_model.linear_model(xgrid, 2)
trained_zero_der = trained_model.pairwise_linear_model.linear_model(xgrid, 0)
trained_first_der = trained_model.pairwise_linear_model.linear_model(xgrid, 1)
tder2_A, tder2_b = trained_second_der[0], trained_second_der[1]
tder0_A, tder0_b = trained_zero_der[0], trained_zero_der[1]
tder1_A, tder1_b = trained_first_der[0], trained_first_der[1]
c = trained_model.get_variables().detach().numpy()
trained_pred_der2 = np.dot(tder2_A, c) + tder2_b
trained_pred_der0 = np.dot(tder0_A, c) + tder0_b
trained_pred_der1 = np.dot(tder1_A, c) + tder1_b
#Try overlaying the second derivative and the zero derivative
fig, axs = plt.subplots()
axs.plot(xgrid, trained_pred_der0)
axs2 = axs.twinx()
axs2.plot(xgrid, trained_pred_der2)
axs.legend()
plt.show()

#%% Testing out third derivative penalty without spacing end knots 

'''
It might be the case that we have to space the end knots as a numerical stability
measure. However, let's see if things work without it
'''
import pickle
from MasterConstants import Model
import numpy as np
import matplotlib.pyplot as plt
import math

result_saved_mod = "cat_results/base_dset_wt_no_inflect_RESULT/Split0/saved_models.p"
mods = pickle.load(open(result_saved_mod, 'rb'))
splines = [mod_spec for mod_spec in mods if (isinstance(mod_spec, Model)) and (len(mod_spec.Zs) == 2) and (mod_spec.oper in ["H", "S"])]
tst_model_spec = Model(oper='S', Zs=(6,6), orb='pp_sigma')
#These splines should be fifth order
trained_mod = mods[tst_model_spec]
# assert(trained_mod.pairwise_linear_model.deg == 5)
xgrid = np.linspace(1.04, 4.5, 500)
#print xgrid to ensure that all the knots are evenly spaced (should be the case)
print(xgrid)
# xgrid = np.linspace(trained_mod.pairwise_linear_model.xknots[0], trained_mod.pairwise_linear_model.xknots[-1], 10_000)

zero_der = trained_mod.pairwise_linear_model.linear_model(xgrid, 0) #len spline_dict['X'] = 5 here
first_der = trained_mod.pairwise_linear_model.linear_model(xgrid, 1) #len spline_dict['X'] = 1 here
second_der = trained_mod.pairwise_linear_model.linear_model(xgrid, 2) #len spline_dict['X'] = 2 here
third_der = trained_mod.pairwise_linear_model.linear_model(xgrid, 3) #len spline_dict['X'] - 3 here

assert(len(first_der) == 2)
assert(len(second_der) == 2)
assert(len(third_der) == 2)

c = trained_mod.get_variables().detach().numpy()

resulting_vals = []

inflect_val = trained_mod.get_inflection_pt()
rlow, rhigh = trained_mod.pairwise_linear_model.r_range()
first_term = (rhigh - rlow) / 2
const = math.pi / 2
second_term = (math.atan(inflect_val) / const) + 1
position = rlow + (first_term * second_term)

for pair in [(0, zero_der), (1, first_der), (2, second_der), (3, third_der)]:
    print(f"Evaluating {pair[0]}")
    derA, derb = pair[1]
    pred_der = np.dot(derA, c) + derb
    resulting_vals.append((pair[0], pred_der))
    fig, axs = plt.subplots()
    axs.plot(xgrid, pred_der)
    axs.axvline(position)
    axs.set_title(f"Predictions of derivative {pair[0]}")
    plt.show()

# fig, axs = plt.subplots()
# axs.plot(xgrid, resulting_vals[-1][1], label = 'third deriv')
# axs2 = axs.twinx()
# axs2.plot(xgrid, resulting_vals[-2][1], label = 'second_deriv')
# axs.legend()
# plt.show()

#%% Inflection point analysis
'''
Experimenting with the effect of the grid density on the inflection point
position.

Theoretically, if the inflection point gets caught between two knots, then
the convex penalty is not affected and the result is that the inflection 
point does not move.

Here will take a look at the evolution of the inflection point penalty on the 
(C|C) pp_sigma model as a function of the xgrid used to evaluate the penalties.
Using the experiments from EXP_SET_5. The inflection point was initialized, 
for this set of experiments, at 1/10 the range. The xgrid for evaluation was 
over just the knots
'''
import pickle
from MasterConstants import *
import math
import os
import numpy as np

def solve_for_inflect_var(rlow: float, rhigh: float, r_target: float) -> float:
    r"""Solves the inflection point equation for the variable x
    
    Arguments:
        rlow (float): lower bound of distance range
        rhigh (float): upper bound of distance range
        r_target (float): The target starting guess for the inflection point
    
    Returns:
        x_val (float): The value of the variable for computing the inflection point
    
    Notes: This function solves the following equation for x:
        r_target = rlow + ((rhigh - rlow) / 2) * ((atan(x)/pi/2) + 1)
    """
    first_term = r_target - rlow
    second_term = 2 / (rhigh - rlow)
    pi_const = math.pi / 2
    operand = ((first_term * second_term) - 1) * pi_const
    x_val = math.tan(operand)
    return x_val

def compute_inflection_point(rlow, rhigh, x_val):
    first_term = (rhigh - rlow) / 2
    const = math.pi / 2
    second_term = (math.atan(x_val) / const) + 1
    return rlow + (first_term * second_term)

#Ensure that the calculations are correct and that the two operations and 
#   interchangeable
rlow, rhigh = 1.04, 4.5 #for all C-C interactions
target = rlow + ((rhigh - rlow) / 10)
x_val_1 = solve_for_inflect_var(rlow, rhigh, target)
calculated_target = compute_inflection_point(rlow, rhigh, x_val_1)
assert(abs(calculated_target - target) < 1E-12)

target = rlow + ((rhigh - rlow) / 5)
x_val_1 = solve_for_inflect_var(rlow, rhigh, target)
calculated_target = compute_inflection_point(rlow, rhigh, x_val_1)
assert(abs(calculated_target - target) < 1E-12)

tst_spec = Model("S", (6,6), 'pp_sigma')

all_files = os.listdir("POSTPROCESSING_SCRATCH/KNOT_EXPERIMENTS_SET_5/")
result_directories = list(filter(lambda x : os.path.isdir(f"POSTPROCESSING_SCRATCH/KNOT_EXPERIMENTS_SET_5/{x}"), all_files))
print("Using the following results")
print(result_directories)

cc_directories = [elem for elem in result_directories if ('wt' not in elem) and ('cc' not in elem)]

model_dictionaries = []
for directory in cc_directories:
    print(f"Loading  POSTPROCESSING_SCRATCH/KNOT_EXPERIMENTS_SET_5/{directory}/Split0/saved_models.p")
    model_dict = pickle.load(open(f"POSTPROCESSING_SCRATCH/KNOT_EXPERIMENTS_SET_5/{directory}/Split0/saved_models.p", 'rb'))
    model_dictionaries.append(model_dict)

assert(len(model_dictionaries) == len(cc_directories))

tst_mods = [d[tst_spec] for d in model_dictionaries]
knot_numbers = [int(elem.split("_")[2]) for elem in cc_directories]
inflect_values = [mod.get_inflection_pt().detach()[0].item() for mod in tst_mods]
inflect_locations = [compute_inflection_point(1.04, 4.5, x) for x in inflect_values]
target = 1.04 + ((4.5 - 1.04) / 10)
differences = [abs(elem - target) for elem in inflect_locations]
import matplotlib.pyplot as plt
plt.scatter(knot_numbers, differences)
plt.show()

#Do some knot analysis
start_val = 1.04 + ((4.5 - 1.04) / 10)
for num_knots, final_x in zip(knot_numbers, inflect_locations):
    xgrid = np.linspace(rlow, rhigh, num_knots)
    fig, axs = plt.subplots()
    for xknot in xgrid:
        axs.axvline(xknot, color = 'blue')
    axs.axvline(start_val, color = 'red', linestyle = '-.')
    axs.axvline(final_x, color = 'green', linestyle = '-.')
    axs.set_title(f"For {num_knots} knots")
    plt.show()

#Do the 500 grid
mod_dict = pickle.load(open("cat_results/dense_grid_for_penalties/results/base_dset_RESULT/Split0/saved_models.p", 'rb'))
tst_mod_500 = mod_dict[tst_spec]
inflect_var = tst_mod_500.get_inflection_pt().detach()[0].item()
x_val_500 = compute_inflection_point(1.04, 4.5, inflect_var)
start_val = 1.04 + ((4.5 - 1.04) / 5)
xgrid_500 = np.linspace(1.04, 4.5, 500)
fig, axs = plt.subplots()
for elem in xgrid_500:
    axs.axvline(elem, color = 'blue', linewidth = 0.5)
knots = tst_mod_500.pairwise_linear_model.xknots
for k in knots:
    axs.axvline(k, color = 'brown', linewidth = 0.75)
axs.set_title("For 500 grid points")
axs.axvline(start_val, color = 'red', linewidth = 0.75, linestyle = '-.')
axs.axvline(x_val_500, color = 'green', linewidth = 0.75, linestyle = '-.')
plt.show()

#%% More inflection point analysis
'''
Performing another inflection point analysis on the results contained in 
EXP_SET_8 and EXP_SET_7
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PostProcessing import generate_master_table

EXP_SET_7_path = "D:DFTBTORCH_DATA_REPOSITORY/EXP_SET_7/analysis_dir/analysis_files"
EXP_SET_8_path = "D:DFTBTORCH_DATA_REPOSITORY/EXP_SET_8/analysis_dir/analysis_files"

generate_master_table(EXP_SET_7_path)
generate_master_table(EXP_SET_8_path)




