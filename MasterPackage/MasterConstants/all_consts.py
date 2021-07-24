# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:26:41 2021

@author: fhu14

Contains all the constant namedtuple to ensure that all the code that's built
on top of this uses the same objects

I hate relative import systems in Python!
"""

from collections import namedtuple

#%% Model constants
RawData = namedtuple('RawData',['index','glabel','Zs','atoms','oper','orb','dftb','rdist'])
RotData = namedtuple('RotData',['raw_indices','rot_indices','rot'] )

Model = namedtuple('Model',['oper', 'Zs', 'orb'])
Bcond = namedtuple('Bcond',['ix','der','val'])

#%% Physical constants
atom_nums = {
    6 : 'C',
    1 : 'H',
    8 : 'O',
    7 : 'N',
    79 : 'Au'
    }

atom_masses = {
    6 : 12.01,
    1 : 1.008,
    8 : 15.999,
    7 : 14.007,
    79 : 196.967
    }

ANGSTROM2BOHR = 1.889725989

#%% SKF backend constants

#NOTE: These constants are useful in interfacing with the DFTBrepulsive SKF code.
#   currently, only interested in the simple format.

H_entries = {'Hdd0': 0,
 'Hdd1': 1,
 'Hdd2': 2,
 'Hpd0': 3,
 'Hpd1': 4,
 'Hpp0': 5,
 'Hpp1': 6,
 'Hsd0': 7,
 'Hsp0': 8,
 'Hss0': 9}

S_entries = {'Sdd0': 0,
 'Sdd1': 1,
 'Sdd2': 2,
 'Spd0': 3,
 'Spd1': 4,
 'Spp0': 5,
 'Spp1': 6,
 'Ssd0': 7,
 'Ssp0': 8,
 'Sss0': 9}

atom_header_entries = ("Ed", "Ep", "Es", "SPE", "Ud", "Up", "Us", "fd", "fp", "fs")

