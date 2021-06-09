# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:53:44 2021

@author: fhu14
"""

from collections import namedtuple
Model = namedtuple('Model',['oper', 'Zs', 'orb'])

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