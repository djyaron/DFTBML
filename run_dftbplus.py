#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:19:17 2021

@author: yaron
"""
import os
from subprocess import call
import numpy as np
from h5py import File
from collections import Counter
from matplotlib import pyplot as plt
from dftbplus import write_dftb_infile, read_dftb_out
from dftb_layer_splines_4 import get_ani1data

from time import time
from auorg_1_1 import ParDict
from dftb import DFTB
pardict = ParDict()

#load data

data_type = 'au' # or 'ani1'
max_config = 1

# for 'au' only
max_mol = 1000

# for 'ani1' only
allowed_Zs = [1, 6, 7, 8]
maxheavy = 2

fermi_temp = 300.0


if os.getenv("USER") == "yaron":
    dftb_exec = "/home/yaron/code/dftbplusexe/dftbplus-20.2.1/bin/dftb+"
    Au_data_path = "/home/yaron/code/dftbtorch/data/Au_energy_clean.h5"
    des_path = "/home/yaron/code/dftbtorch/data/aec_dftb.h5"
else:
    dftb_exec = "C:\\Users\\Frank\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
    Au_data_path = "/home/francishe/Downloads/Datasets/Au_energy_clean.h5"
    des_path = "/home/francishe/Downloads/Datasets/aec_dftb.h5"

#dftb_exec = "/home/francishe/dftbplus-20.2.1/bin/dftb+"
scratch_dir = "dftbscratch"

if data_type == 'ani1':
    target = {'dt': 'dt', 'dr': 'dr', 'pt': 'pt', 'pe': 'pe', 'pr': 'pr',
              'cc': 'cc', 'ht': 'ht',
              'dipole': 'wb97x_dz.dipole',
              'charges': 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']

    heavy_atoms = [x for x in range(1, maxheavy + 1)]
    skf_type = 'mio'
    skf_dir = os.path.join(skf_type)

    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)

elif data_type == 'au':
    skf_type = 'au'
    skf_dir = os.path.join(skf_type)
    dataset = []
    # targets starting with a refer to the value sent by Adam McSloy, generating
    # using or dftb code
    targets = {'ae': 'dftb_plus.elec_energy',
                'ar': 'dftb_plus.rep_energy',
                'at': 'dftb_plus.energy',
                'wd': 'wb97x_dz.energy'}
    
    with File(Au_data_path, 'r') as h5data:
        for mol_name in h5data.keys():
            Zs = h5data[mol_name]['atomic_numbers'][()]
            emp_formula = Counter(Zs)
            if emp_formula[79] <= 2:
                continue
            if len(dataset) >= max_mol:
                continue
            coords = h5data[mol_name]['coordinates'][()]
            nconfig = coords.shape[0]
            targets_nconfig = {k: h5data[mol_name][v][()] for k, v in targets.items()}
            for iconfig in range(min(max_config, nconfig)):
                curr = dict()
                curr['name'] = mol_name
                curr['atomic_numbers'] = Zs
                curr['coordinates'] = coords[iconfig, :, :]
                curr['targets'] = {k: targets_nconfig[k][iconfig] for k in targets}
                dataset.append(curr)
            # keys = [k for k in h5data[mol_name].keys()]

#%% run calcs

DFTBoptions = {'ShellResolvedSCC': True}

if fermi_temp is not None:
    DFTBoptions['FermiTemp'] = fermi_temp

do_our_dftb = True
do_dftbplus = True

for imol,mol in enumerate(dataset):
    Zs = mol['atomic_numbers']
    rcart = mol['coordinates']
    
    natom = len(Zs)
    cart = np.zeros([natom,4])
    cart[:,0] = Zs
    for ix in range(3):
        cart[:,ix+1] = rcart[:,ix]
    charge = 0
    mult = 1
    
    if do_our_dftb:
        start_time = time()
        try:
            if fermi_temp is None:
                dftb_us = DFTB(pardict, cart, charge, mult)
                mol['de'],_,_ = dftb_us.SCF()
                mol['dr'] = dftb_us.repulsion
                mol['dt'] = mol['de'] + mol['dr']
                mol['dconv'] = True
            else:
                smearing = {'scheme': 'fermi',
                            'width' : 3.16679e-6 * fermi_temp}
                dftb_us = DFTB(pardict, cart, charge, mult, smearing = None)
                mol['de'],_,_ = dftb_us.SCF()
                mol['dr'] = dftb_us.repulsion
                mol['dt'] = mol['de'] + mol['dr']
                mol['dconv'] = True
                
        except Exception:
            mol['dconv'] = False
        end_time = time()
        mol['dtime'] = end_time-start_time
    
    if do_dftbplus:
        dftb_infile = os.path.join(scratch_dir,'dftb_in.hsd')
        dftb_outfile = os.path.join(scratch_dir,'dftb.out')
        write_dftb_infile(Zs, rcart, dftb_infile, skf_dir,DFTBoptions)
        start_time = time()
        with open(dftb_outfile,'w') as f:
            try:
                res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
                dftb_res = read_dftb_out(dftb_outfile)
                mol['pt'] = dftb_res['Ehartree']
                mol['pconv'] = True
            except Exception:
                mol['pconv'] = False
        end_time = time()
        mol['ptime'] = end_time-start_time
    
    if 'dconv' in mol and 'pconv' in mol:
        if mol['dconv'] and mol['pconv']:
            print(f"{mol['name']} us elec {mol['de']:7.4f} rep {mol['dr']:7.4f} sum {mol['dt']:7.4f}" \
                  f" diff DFTB+ {np.abs(mol['dt']-mol['pt']):7.4e}") 
        elif mol['dconv']:
            print(f"{mol['name']} DFTB+ failed")
        elif mol['pconv']:
            print(f"{mol['name']} our dftb failed")
        else:
            print(f"{mol['name']} both our dftb and DFTB+ failed")
        #ts = mol['targets']
        #print(f"H5 elec {ts['pe']} rep {ts['pr']} sum {ts['pe'] +ts['pr']}" \
        #      f"tot {ts['pt']}  diff {ts['pt'] - ts['pe'] -ts['pr']} ")
        #print(f"{skf_type} on mol {imol} E(H) = {Ehartree:7.3f} " \
        #  #f" diff resolved(kcal/mol) {np.abs(Ehartree-mol['targets']['dt'])*627.0:7.3e}" \
        #  f" not resolved {np.abs(Ehartree-mol['targets']['pt'])*627.0:7.3e}" )

#%% print summary of Au results
conv = [x for x in dataset if x['dconv'] and x['pconv']]
diff = np.array([x['dt'] - x['pt'] for x in conv]) * 627.0
print(f"rms diff us and DFTB+ {np.sqrt(np.average(np.square(diff)))} kcal/mol")
print(f"max diff us and DFTB+ {np.max(np.abs(diff))} kcal/mol")

diff = np.array([x['dt'] - x['targets']['at'] for x in conv]) * 627.0
print(f"rms diff us and Adam {np.sqrt(np.average(np.square(diff)))} kcal/mol")
print(f"max diff us and Adam {np.max(np.abs(diff))} kcal/mol")

our_time = np.abs([x['dtime'] for x in dataset])
plus_time = np.abs([x['ptime'] for x in dataset])
print(f"our time {np.sum(our_time)}  plus time {np.sum(plus_time)}")

failed = dict()
failed['our dftb'] = [x['name'] for x in dataset if not x['dconv'] and x['pconv']]
failed['dftb+'] = [x['name'] for x in dataset if not x['pconv'] and x['dconv']]
failed['both'] = [x['name'] for x in dataset if not x['pconv'] and not x['dconv']]

for ftype,names in failed.items():
    print(f"{len(names)} molecules failed {ftype}")
    print(names)
         

#%% OLD, commented out, CODE FROM HERE DOWN

# allowed_Zs = [1,6,7,8]
# maxheavy = 2
# skf_test = 'ANI1rep1'
# pkl_file = os.path.join('dftbscratch',skf_test,'_heavy' + str(maxheavy) + '.pk')

# if not os.path.exists(pkl_file):    
#     heavy_atoms = [x for x in range(1,maxheavy+1)]
#     #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
#     max_config = 10 
#     # target = 'dt'
#     target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
#               'cc' : 'cc', 'ht': 'ht',
#                'dipole' : 'wb97x_dz.dipole',
#                'charges' : 'wb97x_dz.cm5_charges'}
#     exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    
    
#     dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    
#     #Proportion for training and validation
#     # prop_train = 0.8
#     # transfer_training = False
#     # transfer_train_params = {
#     #     'test_set' : 'pure',
#     #     'impure_ratio' : 0.2,
#     #     'lower_limit' : 4
#     #     }
#     # train_ener_per_heavy = False
    
#     # training_molecs, validation_molecs = dataset_sorting(dataset, prop_train, 
#     #             transfer_training, transfer_train_params, train_ener_per_heavy)
    
#     #mol = pkl.load(open('mtest.pk','rb'))
#     all_mol = dataset
#     print('generating data for', len(all_mol),'molecules')
    
#     for skf_type in ['mio',skf_test]:
#         copyskf(os.path.join(scratch_dir,skf_type), os.path.join(scratch_dir,'skf'))

#         for imol,mol in enumerate(all_mol):
#             Zs = mol['atomic_numbers']
#             rcart = mol['coordinates']
#             write_dftb_in_hsd(Zs, rcart, scratch_dir)
#             with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:       
#                 res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
#             Ehartree = read_dftb_out(scratch_dir)
#             print(skf_type, imol,Ehartree,np.abs(Ehartree-mol['targets']['dt']))
#             mol[skf_type] = Ehartree
    
#     pkl.dump(all_mol, open(pkl_file,'wb'))
# else:
#     all_mol = pkl.load(open(pkl_file,'rb'))

# #%%
# # fit reference energy
# iZ = {x:i for i,x in enumerate(allowed_Zs)}

# nmol = len(all_mol)
# XX = np.zeros([nmol,len(allowed_Zs)+1])
# mio = np.zeros([nmol])
# cc  = np.zeros([nmol])
# ht = np.zeros([nmol])
# ml  = np.zeros([nmol])
# pt  = np.zeros([nmol])
# for imol,mol in enumerate(all_mol):
#     Zc = collections.Counter(mol['atomic_numbers'])
#     for Z,count in Zc.items():
#         XX[imol, iZ[Z]] = count
#         XX[imol, len(allowed_Zs)] = 1.0
#     cc[imol] = mol['targets']['cc']
#     ht[imol] = mol['targets']['ht']
#     mio[imol] = mol['mio']
#     ml[imol] = mol[skf_test]
#     pt[imol] = mol['targets']['pt']

# yy = ht - mio
# lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# coefs = lsq_res[0]
# pred = mio + np.dot(XX,coefs)

# print(len(pred),'molecules')
# mae_mio = np.mean(np.abs(pred - ht)) * 627.509
# print('mio: mae of preds',mae_mio)

# yy = ht - pt
# lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# coefs = lsq_res[0]
# pred = pt + np.dot(XX,coefs)
# mae_pt = np.mean(np.abs(pred - ht)) * 627.509
# print('pt: mae of preds',mae_pt)

# yy = ht - ml
# lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# coefs = lsq_res[0]
# pred = ml + np.dot(XX,coefs)
# mae_ml = np.mean(np.abs(pred - ht)) * 627.509
# print('ml: mae of preds',mae_ml)

# #%%
# if False:
#     pardict = ParDict()
#     cart = np.hstack([Zs.reshape([-1,1]), rcart])
#     dftb = DFTB(pardict, cart)
#     Eelec, Fs, rhos = dftb.SCF()
#     Erep = dftb.repulsion
#     Eus = Eelec + Erep
    
#     diff = (Ehartree - Eus) * 627

#     print('E from our code Hartree', Eus, 'Ediff  kcal/mol', diff)


# dftb_infile = os.path.join(scratch_dir, 'dftb_in.hsd')
# dftb_outfile = os.path.join(scratch_dir, 'dftb.out')
# DFTBoptions = {'ShellResolvedSCC': True}
# pardict = ParDict()
# charge = 0
# mult = 1


# with File(Au_data_path, 'r') as src, File(des_path, 'w') as des,\
#      open(dftb_outfile, 'w') as f:

#     for mol, moldata in src.items():
#         Zs = moldata['atomic_numbers'][()]
#         coords = moldata['coordinates'][()]
#         des_mol = {'dftb.elec_energy': [],
#                    'dftb.rep_energy': [],
#                    'dftb.energy': [],
#                    'dconv': [],
#                    'dftb_plus.energy': [],
#                    'pconv': []}

#         for iconf, coord in enumerate(coords):
#             # DFTB calculation
#             cart = np.block([Zs.reshape(-1, 1), coord])
#             # noinspection PyBroadException
#             try:
#                 dftb = DFTB(pardict, cart, charge, mult)
#                 de = dftb.SCF()[0]
#                 dr = dftb.repulsion
#                 dt = de + dr
#                 des_mol['dftb.elec_energy'].append(de)
#                 des_mol['dftb.rep_energy'].append(dr)
#                 des_mol['dftb.energy'].append(dt)
#                 des_mol['dconv'].append(True)
#             except Exception:
#                 des_mol['dftb.elec_energy'].append(np.nan)
#                 des_mol['dftb.rep_energy'].append(np.nan)
#                 des_mol['dftb.energy'].append(np.nan)
#                 des_mol['dconv'].append(False)
#                 print(f"DFTB divergence: {mol}, conf #{iconf}")

#             # DFTB+ calculation
#             write_dftb_infile(Zs, coord, dftb_infile, skf_dir, DFTBoptions)
#             # noinspection PyBroadException
#             try:
#                 call(dftb_exec, cwd=scratch_dir, stdout=f, shell=False)
#                 dftb_res = read_dftb_out(dftb_outfile)
#                 pe = dftb_res['Ehartree']
#                 des_mol['dftb_plus.energy'].append(pe)
#                 des_mol['pconv'].append(True)
#             except Exception:
#                 des_mol['dftb_plus.energy'].append(np.nan)
#                 des_mol['pconv'].append(False)
#                 print(f"DFTB+ divergence: {mol}, conf #{iconf}")

#         # Save results to des
#         g = des.create_group(mol)
#         g.create_dataset('atomic_numbers', data=Zs)
#         g.create_dataset('coordinates', data=coords)
#         for entry, data in des_mol.items():
#             g.create_dataset(name=entry, data=data)

# for imol, mol in enumerate(dataset):
#     Zs = mol['atomic_numbers']
#     rcart = mol['coordinates']
#
#     natom = len(Zs)
#     cart = np.zeros([natom, 4])
#     cart[:, 0] = Zs
#     for ix in range(3):
#         cart[:, ix + 1] = rcart[:, ix]
#     charge = 0
#     mult = 1
#     try:
#         dftb_us = DFTB(pardict, cart, charge, mult)
#         mol['de'], _, _ = dftb_us.SCF()
#         mol['dr'] = dftb_us.repulsion
#         mol['dt'] = mol['de'] + mol['dr']
#         mol['dconv'] = True
#     except Exception:
#         mol['dconv'] = False
#
#     dftb_infile = os.path.join(scratch_dir, 'dftb_in.hsd')
#     dftb_outfile = os.path.join(scratch_dir, 'dftb.out')
#     write_dftb_infile(Zs, rcart, dftb_infile, skf_dir, DFTBoptions)
#     with open(dftb_outfile, 'w') as f:
#         try:
#             res = call(dftb_exec, cwd=scratch_dir, stdout=f, shell=False)
#             dftb_res = read_dftb_out(dftb_outfile)
#             mol['pt'] = dftb_res['Ehartree']
#             mol['pconv'] = True
#         except Exception:
#             mol['pconv'] = False
#
#     if mol['dconv'] and mol['pconv']:
#         print(f"{mol['name']} us elec {mol['de']:7.4f} rep {mol['dr']:7.4f} sum {mol['dt']:7.4f}" \
#               f" diff DFTB+ {np.abs(mol['dt'] - mol['pt']):7.4e}")
#     elif mol['dconv']:
#         print(f"{mol['name']} DFTB+ failed")
#     elif mol['pconv']:
#         print(f"{mol['name']} our dftb failed")
#     else:
#         print(f"{mol['name']} both our dftb and DFTB+ failed")
#     # ts = mol['targets']
#     # print(f"H5 elec {ts['pe']} rep {ts['pr']} sum {ts['pe'] +ts['pr']}" \
#     #      f"tot {ts['pt']}  diff {ts['pt'] - ts['pe'] -ts['pr']} ")
#     # print(f"{skf_type} on mol {imol} E(H) = {Ehartree:7.3f} " \
#     #  #f" diff resolved(kcal/mol) {np.abs(Ehartree-mol['targets']['dt'])*627.0:7.3e}" \
#     #  f" not resolved {np.abs(Ehartree-mol['targets']['pt'])*627.0:7.3e}" )
#
# # print summary of Au results
# failed = dict()
# failed['our dftb'] = [x['name'] for x in dataset if not x['dconv'] and x['pconv']]
# failed['dftb+'] = [x['name'] for x in dataset if not x['pconv'] and x['dconv']]
# failed['both'] = [x['name'] for x in dataset if not x['pconv'] and not x['dconv']]
#
# for ftype, names in failed.items():
#     print(f"{len(names)} molecules failed {ftype}")
#     print(names)
#
# # allowed_Zs = [1,6,7,8]
# # maxheavy = 2
# # skf_test = 'ANI1rep1'
# # pkl_file = os.path.join('dftbscratch',skf_test,'_heavy' + str(maxheavy) + '.pk')
#
# # if not os.path.exists(pkl_file):
# #     heavy_atoms = [x for x in range(1,maxheavy+1)]
# #     #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
# #     max_config = 10
# #     # target = 'dt'
# #     target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
# #               'cc' : 'cc', 'ht': 'ht',
# #                'dipole' : 'wb97x_dz.dipole',
# #                'charges' : 'wb97x_dz.cm5_charges'}
# #     exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
#
#
# #     dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
#
# #     #Proportion for training and validation
# #     # prop_train = 0.8
# #     # transfer_training = False
# #     # transfer_train_params = {
# #     #     'test_set' : 'pure',
# #     #     'impure_ratio' : 0.2,
# #     #     'lower_limit' : 4
# #     #     }
# #     # train_ener_per_heavy = False
#
# #     # training_molecs, validation_molecs = dataset_sorting(dataset, prop_train,
# #     #             transfer_training, transfer_train_params, train_ener_per_heavy)
#
# #     #mol = pkl.load(open('mtest.pk','rb'))
# #     all_mol = dataset
# #     print('generating data for', len(all_mol),'molecules')
#
# #     for skf_type in ['mio',skf_test]:
# #         copyskf(os.path.join(scratch_dir,skf_type), os.path.join(scratch_dir,'skf'))
#
# #         for imol,mol in enumerate(all_mol):
# #             Zs = mol['atomic_numbers']
# #             rcart = mol['coordinates']
# #             write_dftb_in_hsd(Zs, rcart, scratch_dir)
# #             with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:
# #                 res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
# #             Ehartree = read_dftb_out(scratch_dir)
# #             print(skf_type, imol,Ehartree,np.abs(Ehartree-mol['targets']['dt']))
# #             mol[skf_type] = Ehartree
#
# #     pkl.dump(all_mol, open(pkl_file,'wb'))
# # else:
# #     all_mol = pkl.load(open(pkl_file,'rb'))
#
# # #%%
# # # fit reference energy
# # iZ = {x:i for i,x in enumerate(allowed_Zs)}
#
# # nmol = len(all_mol)
# # XX = np.zeros([nmol,len(allowed_Zs)+1])
# # mio = np.zeros([nmol])
# # cc  = np.zeros([nmol])
# # ht = np.zeros([nmol])
# # ml  = np.zeros([nmol])
# # pt  = np.zeros([nmol])
# # for imol,mol in enumerate(all_mol):
# #     Zc = collections.Counter(mol['atomic_numbers'])
# #     for Z,count in Zc.items():
# #         XX[imol, iZ[Z]] = count
# #         XX[imol, len(allowed_Zs)] = 1.0
# #     cc[imol] = mol['targets']['cc']
# #     ht[imol] = mol['targets']['ht']
# #     mio[imol] = mol['mio']
# #     ml[imol] = mol[skf_test]
# #     pt[imol] = mol['targets']['pt']
#
# # yy = ht - mio
# # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# # coefs = lsq_res[0]
# # pred = mio + np.dot(XX,coefs)
#
# # print(len(pred),'molecules')
# # mae_mio = np.mean(np.abs(pred - ht)) * 627.509
# # print('mio: mae of preds',mae_mio)
#
# # yy = ht - pt
# # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# # coefs = lsq_res[0]
# # pred = pt + np.dot(XX,coefs)
# # mae_pt = np.mean(np.abs(pred - ht)) * 627.509
# # print('pt: mae of preds',mae_pt)
#
# # yy = ht - ml
# # lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
# # coefs = lsq_res[0]
# # pred = ml + np.dot(XX,coefs)
# # mae_ml = np.mean(np.abs(pred - ht)) * 627.509
# # print('ml: mae of preds',mae_ml)
#
# # #%%
# # if False:
# #     pardict = ParDict()
# #     cart = np.hstack([Zs.reshape([-1,1]), rcart])
# #     dftb = DFTB(pardict, cart)
# #     Eelec, Fs, rhos = dftb.SCF()
# #     Erep = dftb.repulsion
# #     Eus = Eelec + Erep
#
# #     diff = (Ehartree - Eus) * 627
#
# #     print('E from our code Hartree', Eus, 'Ediff  kcal/mol', diff)
