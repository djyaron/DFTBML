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
#from trainedskf import ParDict #Comment this line if using auorg-1-1


def named_re(name, respec, before = 'none', after='require'):
    '''
      wraps a regular expression in a block that gives it a name:
          (?P<name> respec)
      before and after are whitespace requirements before and after the match
         'none'   : do not allow any whitespace
         'allow' : accept but do not require whitespace
         'require' : require whitespace
    '''
    ws = {'none'    : r'',
          'allow'   : r'\s*',
          'require' : r'\s+'}
    res = ws[before] + "(?P<" + name + ">" + respec + ")" + ws[after]
    
    return res

def copyskf(src, dst):
    # (1) makes dst or deletes dst/*.skf and (2) copies src/*.skf to dst
    
    # from: http://stackoverflow.com/questions/273192
    #    prevents raise conditions versus os.path.exists()
    try:
        os.makedirs(dst)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    for item in glob.glob(os.path.join(dst,'*.skf')):
        os.remove(item)
    
    for item_src in glob.glob(os.path.join(src,'*.skf')):
        _, filename = os.path.split(item_src)
        item_dest = os.path.join(dst, filename)
        shutil.copyfile(item_src, item_dest)

def readlines(file):
    f = open(file,'rb')
    lines = f.readlines()
    lines = [line.decode("utf-8").strip() for line in lines]
    return lines

def write_dftb_in_hsd(Zs, rcart_angstroms, scratch_dir):
    rcart = rcart_angstroms
    natom = len(Zs)
    max_angular_momentum = {1: 'H = "s"', 6: 'C = "p"', 7: 'N = "p"', 8: 'O = "p"'}
    
    with open(os.path.join(scratch_dir,'dftb_in.hsd'),'w') as dftbfile:
        dftbfile.write(r'Geometry = xyzFormat {' + '\n')
        dftbfile.write(str(natom)+ '\n')
        dftbfile.write(r'junk' + '\n')
        for iatom,Z in enumerate(Zs):
            line = ELEMENTS[Z].symbol 
            line += ' %.8f' % rcart[iatom,0] + ' %.8f' % rcart[iatom,1] \
                + ' %.8f' % rcart[iatom,2] + ' 0.0'
            dftbfile.write(line+ '\n')
        dftbfile.write(r'}' + '\n')
        dftbfile.write(
            r'Driver = {}' + '\n' +
            r'Hamiltonian = DFTB {' + '\n' +
            r'Scc = Yes' + '\n' +
            r'ShellResolvedSCC = Yes' + '\n' +
            r'SlaterKosterFiles = Type2FileNames {' + '\n' +
            r'Prefix = "skf/"' + '\n' +
            r'Separator = "-"' + '\n' +
            r'Suffix = ".skf"' + '\n' +
            r'}' + '\n' +
            r'MaxAngularMomentum {' + '\n'
            )
        # required because DFTB+ wants ang momentum listed only for elements
        # actually in the molecule        
        for Z, ang_str in max_angular_momentum.items():
            if Z in Zs:
                dftbfile.write(ang_str + '\n')
        dftbfile.write(
            r'}' + '\n' +
            r'}'+ '\n' +
            r'Options {}'  + '\n' +
            r'Analysis {' + '\n' +
            r'CalculateForces = No' + '\n' +
            r'}'+ '\n' 
            )

def read_dftb_out(scratch_dir):
    # From: https://stackoverflow.com/questions/12643009/
    anyfloat = r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)"
    # Parsing line of this type:
    #Total Energy:                      -33.6613043365 H         -915.9707 eV
    energy_string = r'Total\sEnergy:' \
        + named_re('Ehartree', anyfloat, 'require', 'allow') + r'H'\
        + named_re('Eev',anyfloat,'allow','allow') + r'eV'
    energy_re = re.compile(energy_string)
    
    lines = readlines(os.path.join(scratch_dir,'dftb.out'))
    found = False
    for line in lines:
        if energy_re.match(line):
            energy_dict = energy_re.match(line).groupdict()
            Ehartree = float(energy_dict['Ehartree'])
            Eev = float(energy_dict['Eev'])
            found = True
    if not found:
        Ehartree = np.nan
        Eev = np.nan
        print("WARNING: RESULTS NOT FOUND")
    return Ehartree
        

dftb_exec = "/home/yaron//code/dftbplusexe/dftbplus-20.2.1/bin/dftb+"
scratch_dir = "/home/yaron/code/dftbtorch/dftbscratch"


allowed_Zs = [1,6,7,8]
maxheavy = 8
pkl_file = 'heavy' + str(maxheavy) + '.pk'

if not os.path.exists(pkl_file):    
    heavy_atoms = [x for x in range(1,maxheavy+1)]
    #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
    max_config = 10 
    # target = 'dt'
    target = {'dt' : 'dt', 'dr': 'dr', 'pt' : 'pt', 'pe' : 'pe', 'pr' : 'pr',
              'cc' : 'cc',
               'dipole' : 'wb97x_dz.dipole',
               'charges' : 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
    
    
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    
    #Proportion for training and validation
    # prop_train = 0.8
    # transfer_training = False
    # transfer_train_params = {
    #     'test_set' : 'pure',
    #     'impure_ratio' : 0.2,
    #     'lower_limit' : 4
    #     }
    # train_ener_per_heavy = False
    
    # training_molecs, validation_molecs = dataset_sorting(dataset, prop_train, 
    #             transfer_training, transfer_train_params, train_ener_per_heavy)
    
    #mol = pkl.load(open('mtest.pk','rb'))
    all_mol = [x[0] for x in dataset]
    print('generating data for', len(all_mol),'molecules')
    
    for skf_type in ['mio','ml']:
        copyskf(os.path.join(scratch_dir,skf_type), os.path.join(scratch_dir,'skf'))

        for imol,mol in enumerate(all_mol):
            Zs = mol['atomic_numbers']
            rcart = mol['coordinates']
            write_dftb_in_hsd(Zs, rcart, scratch_dir)
            with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:       
                res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
            Ehartree = read_dftb_out(scratch_dir)
            print(skf_type, imol,Ehartree,np.abs(Ehartree-mol['targets']['dt']))
            mol[skf_type] = Ehartree
    
    pkl.dump(all_mol, open(pkl_file,'wb'))
else:
    all_mol = pkl.load(open(pkl_file,'rb'))

#%%
# fit reference energy
iZ = {x:i for i,x in enumerate(allowed_Zs)}

nmol = len(all_mol)
XX = np.zeros([nmol,len(allowed_Zs)+1])
mio = np.zeros([nmol])
cc  = np.zeros([nmol])
ml  = np.zeros([nmol])
pt  = np.zeros([nmol])
for imol,mol in enumerate(all_mol):
    Zc = collections.Counter(mol['atomic_numbers'])
    for Z,count in Zc.items():
        XX[imol, iZ[Z]] = count
        XX[imol, len(allowed_Zs)] = 1.0
    cc[imol] = mol['targets']['cc']
    mio[imol] = mol['mio']
    ml[imol] = mol['ml']
    pt[imol] = mol['targets']['pt']

yy = cc - mio
lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
coefs = lsq_res[0]
pred = mio + np.dot(XX,coefs)

print(len(pred),'molecules')
mae_mio = np.mean(np.abs(pred - cc)) * 627.509
print('mio: mae of preds',mae_mio)

yy = cc - pt
lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
coefs = lsq_res[0]
pred = pt + np.dot(XX,coefs)
mae_pt = np.mean(np.abs(pred - cc)) * 627.509
print('pt: mae of preds',mae_pt)

yy = cc - ml
lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
coefs = lsq_res[0]
pred = ml + np.dot(XX,coefs)
mae_ml = np.mean(np.abs(pred - cc)) * 627.509
print('ml: mae of preds',mae_ml)

#%%
if False:
    pardict = ParDict()
    cart = np.hstack([Zs.reshape([-1,1]), rcart])
    dftb = DFTB(pardict, cart)
    Eelec, Fs, rhos = dftb.SCF()
    Erep = dftb.repulsion
    Eus = Eelec + Erep
    
    diff = (Ehartree - Eus) * 627

    print('E from our code Hartree', Eus, 'Ediff  kcal/mol', diff)

