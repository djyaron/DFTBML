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

import pickle, os
from PlottingUtil import plot_skf_dist_overlay, read_skf_set
from DFTBrepulsive import SKFSet

mols0 = pickle.load(open("fold_molecs_test_8020/Fold0_molecs.p", "rb"))
mols1 = pickle.load(open("fold_molecs_test_8020/Fold1_molecs.p","rb"))
dset = mols0 + mols1

skf_path = "skf_full_pairwise_linear"
dest = skf_path + "/skf_plots"
if (not os.path.isdir(dest)):
    os.mkdir(dest)

skfset = read_skf_set(skf_path)

plot_skf_dist_overlay(skfset, dest = None, mode = 'plot', dset = dset)

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


            
    









    
        
            
        
    




