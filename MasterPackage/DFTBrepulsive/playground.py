import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from h5py import File
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from util import mpl_default_setting


def flatten(dataset):
    # Determine atom types
    Zs = set()
    for moldata in dataset.values():
        Zs.update(moldata['atomic_numbers'][()])
    Zs = tuple(sorted(Zs))
    ZtoName = {1: "nH",
               6: "nC",
               7: "nN",
               8: "nO",
               79: "nAu"}
    for Z in Zs:
        ZtoName.get(Z, f"Atom type {Z} is not included in ZtoName dictionary")
    # Generate flattened dataset
    df = {}
    for mol, moldata in dataset.items():
        nconfs = len(moldata['coordinates'])
        conf_arr = np.arange(nconfs)
        # Column of molecular formulas
        mol_col = [mol] * nconfs
        try:
            df['mol'].extend(mol_col)
        except KeyError:
            df['mol'] = []
            df['mol'].extend(mol_col)
        # Column of conformation indices
        conf_col = np.arange(nconfs, dtype='int')
        try:
            df['conf'].extend(conf_col)
        except KeyError:
            df['conf'] = []
            df['conf'].extend(conf_col)
        # Columns of atomic numbers
        for Z in Zs:
            nZ = list(moldata['atomic_numbers'][()]).count(Z)
            Z_col = [nZ] * nconfs
            try:
                df[ZtoName[Z]].extend(Z_col)
            except KeyError:
                df[ZtoName[Z]] = []
                df[ZtoName[Z]].extend(Z_col)
        # Column of data
        for entry, data in moldata.items():
            if entry in ('atomic_numbers', 'coordinates'):
                continue
            try:
                df[entry].extend(data)
            except KeyError:
                df[entry] = []
                df[entry].extend(data)

    return pd.DataFrame(df)


def compare(df, entry1, entry2, mols='all', show=False):
    if mols == 'all':
        moldata = df
    elif isinstance(mols, str):
        moldata = df.loc[df['mol'] == mols]
    else:
        mol_mask = np.logical_or.reduce([df['mol'] == mol for mol in mols])
        moldata = df.loc[mol_mask]
    e1 = moldata[entry1]
    e2 = moldata[entry2]
    names = ('nH', 'nC', 'nN', 'nO', 'nAu')
    Zs = [Z for Z in names if Z in df.columns]
    nZ = moldata[Zs]

    # Fit linear model (shifter)
    s1 = LinearRegression()
    s2 = LinearRegression()
    s1.fit(nZ, e1)
    s2.fit(nZ, e2)
    e1_shifted = e1 - s1.predict(nZ)
    e2_shifted = e2 - s2.predict(nZ)
    mae = mean_absolute_error(e1_shifted, e2_shifted) * HARTREE
    rms = mean_squared_error(e1_shifted, e2_shifted) ** 0.5 * HARTREE


    if show:
        print(f"=======================================================")
        print(f"{entry1} vs {entry2}")
        print(f"-------------------- Discrepancies --------------------")
        print(f"MAE: {mae:>15.3f} kcal/mol")
        print(f"RMSE: {rms:>14.3f} kcal/mol")
        print(f"=======================================================")

    return mae, rms


def all_deviations(df):
    mae_tmp = list()
    rmse_tmp = list()

    ENERGIES = [e for e in df.columns if 'energy' in e
                and 'disp' not in e
                and 'rep' not in e
                and 'elec' not in e]

    for e1 in ENERGIES:
        for e2 in ENERGIES:
            mae, rmse = compare(df, e1, e2)
            mae_tmp_df = pd.DataFrame(data={"e1": [e1], "e2": [e2], "MAE": [mae]})
            rmse_tmp_df = pd.DataFrame(data={"e1": [e1], "e2": [e2], "RMSE": [rmse]})
            mae_tmp.append(mae_tmp_df)
            rmse_tmp.append(rmse_tmp_df)

    mae_df_long = pd.concat(mae_tmp, ignore_index=True)
    rmse_df_long = pd.concat(rmse_tmp, ignore_index=True)

    mae_df_wide = mae_df_long.pivot("e1", "e2", "MAE").reindex(index=ENERGIES, columns=ENERGIES)
    rmse_df_wide = rmse_df_long.pivot("e1", "e2", "RMSE").reindex(index=ENERGIES, columns=ENERGIES)

    return mae_df_long, rmse_df_long, mae_df_wide, rmse_df_wide


mpl_default_setting()

# Combine datasets
# src_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean.h5'
# res_path = '/home/francishe/Documents/DFTBrepulsive/aec_dftb+.h5'
# des_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/Au_energy_clean_dispersion.h5'
#
# # with File(src_path, 'r') as src, File(res_path, 'r') as res, File(des_path, 'w') as des:
# #     for (mol, srcdata), resdata in zip(src.items(), res.values()):
# #         mlist = [np.isnan(data) for data in resdata.values()]
# #         mask = ~np.logical_or.reduce(mlist)
# #         if mask.sum() == 0:
# #             continue
# #
# #         g = des.create_group(mol)
# #         g.create_dataset('atomic_numbers', data=srcdata['atomic_numbers'])
# #         g.create_dataset('coordinates', data=srcdata['coordinates'][mask, :, :])
# #         g.create_dataset('fhi_aims_md.total_energy', data=srcdata['wb97x_dz.energy'][mask])
# #         g.create_dataset('dftb.elec_energy', data=srcdata['dftb_plus.elec_energy'][mask])
# #         g.create_dataset('dftb.rep_energy', data=srcdata['dftb_plus.rep_energy'][mask])
# #         g.create_dataset('dftb.energy', data=srcdata['dftb_plus.energy'][mask])
# #         for entry, data in resdata.items():
# #             if 'time' in entry:
# #                 continue
# #             g.create_dataset(entry, data=data[mask])
#
# # with File(des_path, 'r+') as des:
# #     for mol in des.keys():
# #         pf = des[mol]['dftb_plus.force_related_energy'][()]
# #         pr = des[mol]['dftb_plus.rep_energy'][()]
# #         pc = pf - pr
# #         des[mol].create_dataset('dftb_plus.rep_corrected_energy', data=pc)
#
# # Flatten datasets
# # Dataset(des_path).entries()
# # with File(des_path, 'r') as dataset:
# #     df = flatten(dataset)
# df_path = '/home/francishe/Documents/DFTBrepulsive/df.pkl'
# df  = pd.read_pickle(df_path)
#
# _, _, mae_df, rmse_df = all_deviations(df)
# # mae_df = mae_df.drop("dftb.energy").drop(columns=["dftb.energy"])
# # rmse_df = rmse_df.drop("dftb.energy").drop(columns=["dftb.energy"])
# mask = np.zeros_like(mae_df)
# mask[np.triu_indices_from(mask)] = True
# mask = (1 - mask).T
# error_selector = "mae"
# location = "right"
#
# if location in ("top", "bottom"):
#     figsize = (20, 24)
# elif location in ("left", "right"):
#     figsize = (24, 20)
# else:
#     raise ValueError("location")
#
# plt.figure(figsize=figsize)
# ax = sns.heatmap(eval(f"{error_selector}_df"), annot=True, annot_kws={'size': 20},
#                  fmt='.2f', mask=mask, cbar_kws=dict(use_gridspec=False, location=location))
# ax.set_xlabel(None)
# ax.set_ylabel(None)
# # ax.set_title(error_selector.upper(), fontsize=30)
# plt.xticks(rotation=30, ha='right', size=20)
# plt.yticks(rotation=30, ha='right', size=20)
# plt.show()
# plt.savefig(f"/Users/francishe/Downloads/{error_selector.upper()}_{location}.png", dpi=72)

# # Histogram of losses
# conv_path = '/home/francishe/Documents/DFTBrepulsive/Au_cv/Au_cv (lstsq, nknots=50, deg=5, rmax=au_short~au_full, ptype=None)/Au_cv_rmax.pkl'
# with open(conv_path, 'rb') as f:
#     conv_res = pkl.load(f)
#
# short_train = conv_res['all_losses']['sparse_train'][0][-1][0][-1] * HARTREE
# short_test = conv_res['all_losses']['sparse_test'][0][-1][0][-1] * HARTREE
#
# fig, ax = plt.subplots(figsize=(12, 10))
# plt.hist(short_train, bins=np.linspace(0, 0.1, 100) * HARTREE)
# plt.xlabel(r"$|E_{true} - E_{pred}|$  (kcal / mol)")
# plt.ylabel('Counts')
# plt.title('Distribution of training error (dense, short)')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(12, 10))
# plt.hist(short_test, bins=np.linspace(0, 0.1, 100) * HARTREE)
# plt.xlabel(r"$|E_{true} - E_{pred}|$  (kcal / mol)")
# plt.ylabel('Counts')
# plt.title('Distribution of test error (dense, short)')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(12, 10))
# plt.boxplot([short_train, short_test])
# plt.xticks(ticks=[1, 2], labels=['Train', 'Test'])
# plt.ylabel(r"$|E_{true} - E_{pred}|$  (kcal / mol)")
# plt.title('Distribution of error (dense, short)')
# plt.show()

from deprecated.model_v1 import get_predictions_from_sparse
from sklearn.metrics import mean_absolute_error as mae
from consts import HARTREE, ALIAS2TARGET
from os.path import exists
from fold import Fold

dset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
fset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.pkl'
tb_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/a1k_convex.h5'
res_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/a1k_res.h5'
rf_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/a1k_res.pkl'
g_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/gammas_50_5_full_a1k.h5'
cv_path = '/home/francishe/Documents/DFTBrepulsive/Au_cv/Au_cv (cvxopt, nknots=50, deg=3, rmax=au_short~au_short, ptype=convex)/Au_cv_rmax.pkl'

with open(cv_path, 'rb') as f:
    cv_res = pkl.load(f)
    shifter = cv_res['shifter'][0]
    sparse_coefs = cv_res['sparse_coefs_cv'][0][0][0]
    map_models = cv_res['map_models_cv'][0][0]
    sparse_loss = cv_res['all_losses']['sparse_train'][0][0][0][0]['mae'] * HARTREE
    sparse_xydata = cv_res['all_xydata']['sparse_xydata'][0][0][0]
    print(f"Train loss (aed): {sparse_loss:.3f} kcal/mol")

# IMPORTANT: energy shifter from fitted ref coefs
ref_shifter = LinearRegression()
ref_shifter.coef_ = sparse_coefs[-5:]
ref_shifter.intercept_ = sparse_coefs[-6]

# Test 1: can we reproduce model performance?
## Flatten a1K and save
if exists(fset_path):
    fset = pd.read_pickle(fset_path)
else:
    with File(dset_path, 'r') as dset:
        fset = flatten(dset)
        fset.to_pickle(fset_path)
## Model preds
with File(dset_path, 'r') as dset:
    fd = Fold.from_dataset(dset)
_preds = get_predictions_from_sparse(sparse_coefs, map_models, fd, g_path)
preds = np.concatenate([moldata for moldata in _preds.values()])
rep_mod = preds # will be used in later sections
## Shift targets
an = fset.loc[:, 'nH':'nAu']
targets = fset[ALIAS2TARGET['fm']] - fset[ALIAS2TARGET['pf']] + fset[ALIAS2TARGET['pr']] \
          - shifter[ALIAS2TARGET['fm']].predict(an) \
          + shifter[ALIAS2TARGET['pf']].predict(an) \
          - shifter[ALIAS2TARGET['pr']].predict(an)
## Compare targets and preds
loss = mae(targets, preds) * HARTREE
print(f"Test 1 loss (a1k): {loss:.3f} kcal/mol")

# Test 2: can DFTB+ reproduce model performance?
## Combine a1k and a1k_convex (DFTB+ results)
if exists(res_path):
    pass
else:
    with File(dset_path, 'r') as src, File(tb_path, 'r') as res, File(res_path, 'w') as des:
        for mol in src.keys():
            srcdata = src[mol]
            resdata = res[mol]
            desdata = des.create_group(mol)
            for name, data in srcdata.items():
                if 'dftb' not in name:
                    desdata.create_dataset(name, data=data)
            for name, data in resdata.items():
                if 'time' not in name:
                    desdata.create_dataset(name, data=data)
## Flatten combined data
if exists(rf_path):
    rset = pd.read_pickle(rf_path)
else:
    with File(res_path, 'r') as res:
        rset = flatten(res)
        rset.to_pickle(rf_path)
## Shift targets and preds
an = rset.loc[:, 'nH':'nAu']
targets = rset[ALIAS2TARGET['fm']] \
          - shifter[ALIAS2TARGET['fm']].predict(an) \
          + shifter[ALIAS2TARGET['pf']].predict(an) \
          - shifter[ALIAS2TARGET['pr']].predict(an)
preds = rset[ALIAS2TARGET['pf']] + ref_shifter.predict(an)
## Compare targets and preds
loss = mae(targets, preds) * HARTREE
print(f"Test 2 loss (a1k): {loss:.3f} kcal/mol")

# Test 3: can DFTB+ reproduce model predictions of rep energies?
rep_tb = rset[ALIAS2TARGET['pr']] + ref_shifter.predict(an)
loss = mae(rep_mod, rep_tb) * HARTREE
print(f"Test 3 loss (a1k): {loss:.3f} kcal/mol")

# Test 4: plot a repulsive spline using DFTB+ calculations
## Generate a simple dataset (N2)
n2_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/n2.h5'
n2r_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/n2_tb.h5'
n2c_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/n2_combined.h5'

from consts import CUTOFFS_AST
cutoff = CUTOFFS_AST[(7, 7)]
coords = np.zeros((1000, 2, 3))
coords[:, 1, 0] = np.linspace(cutoff[0], cutoff[-1], 1000)

# with File(n2_path, 'w') as n2:
#     g = n2.create_group('N2')
#     g.create_dataset('atomic_numbers', data=[7, 7])
#     g.create_dataset('coordinates', data=coords)

with File(n2_path, 'r') as src, File(n2r_path, 'r') as res, File(n2c_path, 'w') as des:
    for mol in src.keys():
        srcdata = src[mol]
        resdata = res[mol]
        desdata = des.create_group(mol)
        for name, data in srcdata.items():
            desdata.create_dataset(name, data=data)
        for name, data in resdata.items():
            if 'time' not in name:
                desdata.create_dataset(name, data=data)

with File(n2c_path, 'r') as des:
    n2f = flatten(des)

spl_mod = sparse_xydata[(7, 7)]
spl_skf = np.array([coords[:, 1, 0], n2f[ALIAS2TARGET['pr']]])
plt.plot(spl_mod[0], spl_mod[1] * HARTREE, 'r.')
plt.plot(spl_skf[0], spl_skf[1] * HARTREE, 'b.')
# plt.ylim(-0.5, 2)
from sklearn.metrics import mean_absolute_error as mae
print(f"Test 4 (n2): {mae(spl_mod[1], spl_skf[1]) * HARTREE:.3f} kcal/mol")
plt.show()
