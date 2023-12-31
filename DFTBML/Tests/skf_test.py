# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:51:38 2021

@author: fhu14

Test suite for different functionalities present in SKF writer.
"""
#%% Imports, definitions

import os
import pickle
import shutil

import Auorg_1_1
import DFTBrepulsive  # SKF, SKFSet, SKFBlockCreator
from DFTBPlus import run_organics
from MasterConstants import Model, atom_masses, atom_nums
from SKF import determine_index, write_skfs

from .helpers import ani1_path, auorg_dir, test_data_dir

#%% Code behind


def test_determine_index():
    r"""Tests that the proper indices are used for the model values"""

    print("Testing get index...")

    dummy_mod_specs = [
        Model(oper="H", Zs=(6, 1), orb="ss"),
        Model(oper="H", Zs=(1, 6), orb="sp"),
        Model(oper="H", Zs=(7, 7), orb="pp_pi"),
        Model(oper="H", Zs=(7, 7), orb="pp_sigma"),
        Model(oper="S", Zs=(6, 1), orb="ss"),
        Model(oper="S", Zs=(1, 6), orb="sp"),
        Model(oper="S", Zs=(7, 7), orb="pp_pi"),
        Model(oper="S", Zs=(7, 7), orb="pp_sigma"),
    ]

    target_inds = [9, 8, 6, 5, 9, 8, 6, 5]

    for i, mod_spec in enumerate(dummy_mod_specs):
        assert determine_index(mod_spec) == target_inds[i]

    print("Testing index passed successfully")


def test_new_SKF_framework(clear_direc: bool):
    r"""Tests the SKF functionalities written by Francis in DFTBrepulsive.
    This is accomplished by writing out SKFs from past saved models
    (which should not fail) and then checking the value obtained by
    DFTB+ on those skfs against benchmarks.
    """
    print("Testing new SKF framework...")
    tol = 1e-6
    skf_8020_benchmark = 4.318332359222469
    refacted_benchmark = 4.290744984048831

    # Parameters for writing SKFs
    model_path = os.path.join(test_data_dir, "skf_8020_100knot/Split0/saved_models.p")
    models = pickle.load(open(model_path, "rb"))
    compute_S_block = True
    ref_direct = os.path.join(auorg_dir, "auorg-1-1")
    rep_mode = "old"
    dest = os.path.join(test_data_dir, "skf_8020_100knot_deriv_delete")
    spl_ngrid = 50

    # Write the SKFs using the new framework
    write_skfs(
        models, atom_nums, atom_masses, compute_S_block, ref_direct, rep_mode, dest, spl_ngrid
    )

    # Now run them through DFTB+ with the proper settings from run_organics
    max_config = 1
    maxheavy = 8
    allowed_Zs = [1, 6, 7, 8]
    target = "cc"

    skf_dir = os.path.join(test_data_dir, "skf_8020_100knot_deriv_delete")

    exec_path = "C:\\Users\\fhu14\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
    pardict = Auorg_1_1.ParDict()
    rep_setting = "old"

    err_Ha, err_kcal, _ = run_organics(
        ani1_path,
        max_config,
        maxheavy,
        allowed_Zs,
        target,
        skf_dir,
        exec_path,
        pardict,
        rep_setting,
    )

    print(f"Error in Hartrees: {err_Ha}")
    print(f"Error in kcal/mol: {err_kcal}")

    assert abs(err_kcal - skf_8020_benchmark) < tol

    # Do another SKF set
    model_path = os.path.join(test_data_dir, "refacted_joined_spline_run/Split0/saved_models.p")
    models = pickle.load(open(model_path, "rb"))
    compute_S_block = True
    ref_direct = os.path.join(auorg_dir, "auorg-1-1")
    rep_mode = "old"
    dest = os.path.join(test_data_dir, "refacted_joined_spline_run_deriv_delete")
    spl_ngrid = 50

    # Write the new SKFs using the new framework
    write_skfs(
        models, atom_nums, atom_masses, compute_S_block, ref_direct, rep_mode, dest, spl_ngrid
    )

    # Now run them throughn DFTB+ with the proper settings from run_organics
    skf_dir = os.path.join(test_data_dir, "refacted_joined_spline_run_deriv_delete")

    err_Ha, err_kcal, _ = run_organics(
        ani1_path,
        max_config,
        maxheavy,
        allowed_Zs,
        target,
        skf_dir,
        exec_path,
        pardict,
        rep_setting,
    )

    print(f"Error in Hartrees: {err_Ha}")
    print(f"Error in kcal/mol: {err_kcal}")

    assert abs(err_kcal - refacted_benchmark) < tol

    if clear_direc:
        print("Doing some cleanup...")
        shutil.rmtree(os.path.join(test_data_dir, "skf_8020_100knot_deriv_delete"))
        shutil.rmtree(os.path.join(test_data_dir, "refacted_joined_spline_run_deriv_delete"))

    print("New SKF framework tests passed")


def run_skf_tests(clear_direc: bool = True):
    test_determine_index()
    test_new_SKF_framework(clear_direc)


#%% Main block
if __name__ == "__main__":
    run_skf_tests(True)
