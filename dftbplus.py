#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:39:16 2021

@author: yaron
"""

from dftb_layer_splines_4 import get_ani1data

from skfwriter import main
from dftb import ANGSTROM2BOHR
from model_ranges import plot_skf_values
from pathlib import Path
import glob
import os, shutil
import errno
from subprocess import call
import re
from elements import ELEMENTS
import pickle as pkl
from mio_0_1 import ParDict
import numpy as np
import collections
from h5py import File
#from trainedskf import ParDict #Comment this line if using auorg-1-1
from numbers import Real
from typing import Union, List, Optional, Dict, Any, Literal
Array = np.ndarray

def named_re(name: str,  respec: str, 
             before: str = 'none', after: str='require') -> str:
    r"""
    Wrap a regular expression in a block that gives it a name.
    
    Arguments:
        respec (string): regexp to be wrapped
        name (string): name to be assigned to the parsed results:
                (?P<name> respec)
        before (string): whitespace requirements before respec with choices
            'none'   : do not allow any whitespace
            'allow' : accept but do not require whitespace
            'require' : require whitespace
        after(string): whitespace requirements after respec with same choices
            as for the after variable

    Returns:            
        final_regexp: wrapped version of the respec
    """
    ws = {'none'    : r'',
          'allow'   : r'\s*',
          'require' : r'\s+'}
    res = ws[before] + "(?P<" + name + ">" + respec + ")" + ws[after]
    
    return res

# def copyskf(src: str, dst: str):
#     r"""
#     Copy skf files from src to dist path.
    
#     Arguments: 
#         src: path
    
#     # (1) makes dst or deletes dst/*.skf and (2) copies src/*.skf to dst
    

#     """
#     # from: http://stackoverflow.com/questions/273192
#     #    prevents raise conditions versus os.path.exists()
#     try:
#         os.makedirs(dst)
#     except OSError as exception:
#         if exception.errno != errno.EEXIST:
#             raise
#     for item in glob.glob(os.path.join(dst,'*.skf')):
#         os.remove(item)
    
#     for item_src in glob.glob(os.path.join(src,'*.skf')):
#         _, filename = os.path.split(item_src)
#         item_dest = os.path.join(dst, filename)
#         shutil.copyfile(item_src, item_dest)

def write_dftb_infile(Zs: List[int], rcart_angstroms: Array, 
                      file_path: str, skf_dir: str,
                      DFTBparams_overrides: dict = {}):
    r"""
    Write DFTB HSD input file (dftb_hsd.in) for single point calculation.
    
    Arguments:
        Zs (List[int]): element numbers for atoms in the molecule
        rcart_angstroms (array[]): [natom,3] array with cartesian coordinates
          in anstroms
        file_path (str): path to the output file (e.g. 'scratch/dftb_in.hsd')
        skf_dir (str): directory with the SKF files (should not end in / or \)
        DFTBparams_overrides (dict): dict to override these default params
           'ShellResolvedSCC' : True 
                 If True, s,p,d.. have separate Hubbard parameters
    
    Raises:
        ArgumentError: If keys in DFTBparams_overrides are not supported
                 
    """
    # Default DFTB parameters
    DFTBparams = {'ShellResolvedSCC' : True}
    # Check and implement requested overides to DFTB parameters
    if any([x not in DFTBparams for x in DFTBparams_overrides]):
        unsupported = [x for x in DFTBparams_overrides if x not in DFTBparams]
        raise Exception('Unsupported DFTB parameters '+ 
                            ' '.join(unsupported))
    DFTBparams.update(DFTBparams_overrides)
    # For convenience. The angstroms is attached to variable name to prevent
    # use of a.u. (DFTB+ uses a.u. for most quantities, so use of a.u. here may
    # be a common error.)
    rcart = rcart_angstroms
    natom = len(Zs)
    #HSD input requires list of element types, and their max ang momentum, only
    #for elements in this molecule.
    Ztypes = np.unique(Zs)
    # map from Z to the "type" given in the TypeNames argument of the HSD file
    ZtoType = {Z:(i+1) for i,Z in enumerate(Ztypes)}
    with open(file_path,'w') as dftbfile:
        dftbfile.write(r'Geometry = {' + '\n')
        line = r'  TypeNames = {'
        for Z in Ztypes:
            line += r' "' + ELEMENTS[Z].symbol + r'" '
        line += r'}'
        dftbfile.write(line + '\n')
        dftbfile.write(r'   TypesAndCoordinates [Angstrom] = {' + '\n')
        for iatom,Z in enumerate(Zs):
            line = r'      ' + str(ZtoType[Z])
            line += ' %.8f' % rcart[iatom,0] + ' %.8f' % rcart[iatom,1] \
                + ' %.8f' % rcart[iatom,2]
            dftbfile.write(line+ '\n')
        dftbfile.write(r'   }' + '\n')
        dftbfile.write(r'}' + '\n')
        dftbfile.write(
            r'Driver = {}' + '\n' +
            r'Hamiltonian = DFTB {' + '\n' +
            r'   Scc = Yes' + '\n')
        if DFTBparams['ShellResolvedSCC']:
            # Using OrbitalResolved for backward compatibility
            # This is why the version number below is 5
            dftbfile.write(r'   OrbitalResolvedSCC = Yes' + '\n')
        else:
            dftbfile.write(r'   OrbitalResolvedSCC = No' + '\n')
        dftbfile.write(
            r'   SlaterKosterFiles = Type2FileNames {' + '\n' +
            r'      Prefix = "' + skf_dir + r'/"' + '\n' +
            r'      Separator = "-"' + '\n' +
            r'      Suffix = ".skf"' + '\n' +
            r'   }' + '\n' +
            r'   MaxAngularMomentum {' + '\n'
            )
        # required because DFTB+ wants ang momentum listed only for elements
        # actually in the molecule. The block field of ELEMENT indicates
        # the s,p,d,f label of the valence orbitals, which works for elements
        # currently being studied.
        # TODO: How does DFTB+ handle an element like Ca, for which block = 's' 
        for Z in Ztypes:
            ele = ELEMENTS[Z]
            dftbfile.write(r'      ' + ele.symbol 
                           + r' = "' + ele.block + '"\n')
        dftbfile.write(
            r'   }' + '\n' +
            r'}'+ '\n' +
            r'Options {}'  + '\n' +
            r'Analysis {' + '\n' +
            r'   CalculateForces = No' + '\n' +
            r'}' + '\n' )
        # A windows executable is only available for version 17.1
        #  https://sites.google.com/view/djmolplatform/get-dftb-17-1-windows
        # and this version uses the OrbitalResolvedSCC keywork, instead of
        # ShellResolvedSCC. We use parserversion = 5, so that more recent 
        # versions of DFTB+ will use OrbitalResolvedSCC
        dftbfile.write(
            r'ParserOptions {' + '\n' 
            r'   Parserversion = 5' + '\n' 
            r'}'
            )

def read_dftb_out(file_path: str) -> dict:
    r"""
    Parse a dftb.out file.

    Args:
        file_path (str): path to dftb.out file

    Returns:
        results(dict): parsed results. Currently includes:
            Eev: total energy in eV
            Ehartree: total energy in Hartree
    
    Raises:
        Exception('DFTB calculation failed') 
    
    """
    # From: https://stackoverflow.com/questions/12643009/
    anyfloat = r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)"
    # Parsing line of this type:
    #Total Energy:                      -33.6613043365 H         -915.9707 eV
    energy_string = r'Total\sEnergy:' \
        + named_re('Ehartree', anyfloat, 'require', 'allow') + r'H'\
        + named_re('Eev',anyfloat,'allow','allow') + r'eV'
    energy_re = re.compile(energy_string)
    
    with open(file_path,'rb') as file_in:
        lines = file_in.readlines()
    lines = [line.decode("utf-8").strip() for line in lines]
    
    results = dict()
    found = False
    for line in lines:
        if energy_re.match(line):
            energy_dict = energy_re.match(line).groupdict()
            results['Ehartree'] = float(energy_dict['Ehartree'])
            results['Eev'] = float(energy_dict['Eev'])
            found = True
    if not found:
        results['Ehartree']= np.nan
        results['Eev'] = np.nan
        raise Exception('DFTB Calculation Failed')
    return results


#%% load data
if os.getenv("USER") == "yaron":
    dftb_exec = "/home/yaron/code/dftbplusexe/dftbplus-20.2.1/bin/dftb+"
else:
    dftb_exec = "C:\\Users\\Frank\\Desktop\\DFTB17.1Windows\\DFTB17.1Windows-CygWin\\dftb+"
scratch_dir = "dftbscratch"

if False:
    target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
              'cc' : 'cc', 'ht': 'ht',
               'dipole' : 'wb97x_dz.dipole',
               'charges' : 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
       
    allowed_Zs = [1,6,7,8]
    maxheavy = 2
    heavy_atoms = [x for x in range(1,maxheavy+1)]
    skf_type = 'mio'
    skf_dir = os.path.join(skf_type)
    max_config = 3
    
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)

Au_data_path = os.path.join('data', 'Au_energy_clean.h5')
maxconfig = 1
skf_type = 'au'
skf_dir = os.path.join(skf_type)
dataset = []
# targets starting with a refer to the value sent by Adam McSloy, generating
# using or dftb code 
targets = {'ae': 'dftb_plus.elec_energy',
           'ar': 'dftb_plus.rep_energy',
           'at': 'dftb_plus.energy',
           'wd': 'wb97x_dz.energy'}
with File(Au_data_path,'r') as h5data:
    for mol_name in h5data.keys():
        Zs = h5data[mol_name]['atomic_numbers'][()]
        coords= h5data[mol_name]['coordinates'][()]
        nconfig = coords.shape[0]
        targets_nconfig = {k:h5data[mol_name][v][()] for k,v in targets.items()}
        for iconfig in range(min(maxconfig,nconfig)):
            curr = dict()
            curr['name'] = mol_name
            curr['atomic_numbers'] = Zs
            curr['coordinates'] = coords[iconfig,:,:]
            curr['targets'] = {k:targets_nconfig[k][iconfig] for k in targets}
            dataset.append(curr)
        #keys = [k for k in h5data[mol_name].keys()]


#%% run calcs
from auorg_1_1 import ParDict
from dftb import DFTB
pardict = ParDict()
DFTBoptions = {'ShellResolvedSCC': True}
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
    try:
        dftb_us = DFTB(pardict, cart, charge, mult)
        mol['de'],_,_ = dftb_us.SCF()
        mol['dr'] = dftb_us.repulsion
        mol['dt'] = mol['de'] + mol['dr']
        mol['dconv'] = True
    except Exception:
        mol['dconv'] = False    
    
    dftb_infile = os.path.join(scratch_dir,'dftb_in.hsd')
    dftb_outfile = os.path.join(scratch_dir,'dftb.out')
    write_dftb_infile(Zs, rcart, dftb_infile, skf_dir,DFTBoptions)
    with open(dftb_outfile,'w') as f:
        try:
            res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
            dftb_res = read_dftb_out(dftb_outfile)
            mol['pt'] = dftb_res['Ehartree']
            mol['pconv'] = True
        except Exception:
            mol['pconv'] = False

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

# print summary of Au results
failed = dict()
failed['our dftb'] = [x['name'] for x in dataset if not x['dconv'] and x['pconv']]
failed['dftb+'] = [x['name'] for x in dataset if not x['pconv'] and x['dconv']]
failed['both'] = [x['name'] for x in dataset if not x['pconv'] and not x['dconv']]

for ftype,names in failed.items():
    print(f"{len(names)} molecules failed {ftype}")
    print(names)
    
        



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

