# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:26:41 2021

@author: fhu14

Contains all the constant namedtuple to ensure that all the code that's built
on top of this uses the same objects

I hate relative import systems in Python!
"""

from collections import namedtuple

RawData = namedtuple('RawData',['index','glabel','Zs','atoms','oper','orb','dftb','rdist'])
RotData = namedtuple('RotData',['raw_indices','rot_indices','rot'] )

Model = namedtuple('Model',['oper', 'Zs', 'orb'])
Bcond = namedtuple('Bcond',['ix','der','val'])

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
