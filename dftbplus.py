#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:39:16 2021

@author: yaron
"""

from dftb_layer_splines_4 import *
from trainedskf import ParDict #Comment this line if using auorg-1-1
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
ANGSTROM2BOHR = 1.889725989

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



allowed_Zs = [1,6,7,8]
heavy_atoms = [1,2,3,4,5]
#Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
max_config = 10
# target = 'dt'
target = {'Etot' : 'pt',
           'dipole' : 'wb97x_dz.dipole',
           'charges' : 'wb97x_dz.cm5_charges'}
exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']


dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)

#Proportion for training and validation
prop_train = 0.8
transfer_training = False
transfer_train_params = {
    'test_set' : 'pure',
    'impure_ratio' : 0.2,
    'lower_limit' : 4
    }
train_ener_per_heavy = True

training_molecs, validation_molecs = dataset_sorting(dataset, prop_train, 
            transfer_training, transfer_train_params, train_ener_per_heavy)


#%%def run_dftb_in_scratch_dir():
dftb_exec = "/home/yaron//code/dftbplusexe/dftbplus-20.2.1/bin/dftb+"
scratch_dir = "/home/yaron/code/dftbtorch/dftbscratch"
copyskf(os.path.join(scratch_dir,'mio'), os.path.join(scratch_dir,'skf'))

#%%
mol = training_molecs[0]
Zs = mol['atomic_numbers']
rcart = mol['coordinates'] / ANGSTROM2BOHR
natom = len(Zs)
with open(os.path.join(scratch_dir,'mol.xyz'),'w') as xyzfile:
    xyzfile.write(str(natom)+ '\n')
    xyzfile.write('junk \n')
    for iatom,Z in enumerate(Zs):
        line = ELEMENTS[Z].symbol 
        line += ' %.8f' % rcart[iatom,0] + ' %.8f' % rcart[iatom,1] \
            + ' %.8f' % rcart[iatom,2] + ' 0.0'
        xyzfile.write(line+ '\n')
# required because DFTB+ wants ang momentum listed only for elements
# actually in the molecule        
max_angular_momentum = {1: 'H = "s"', 6: 'C = "p"', 7: 'N = "p"', 8: 'O="p"'}
with open(os.path.join(scratch_dir,'ang.txt'),'w') as angfile:
    for Z, ang_str in max_angular_momentum.items():
        if Z in Zs:
            angfile.write(ang_str + '\n')
        
with open(os.path.join(scratch_dir,'dftb.out'),'w') as f:
   res = call(dftb_exec,cwd=scratch_dir,stdout = f, shell=False)
   
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
    print("WARNING: RESULTS NOT FOUND")
else:
    print('Ehartree',Ehartree,'Eev',Eev)





