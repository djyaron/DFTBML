# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:58:23 2020

@author: Frank

New ParDict() method for skf files generated from the model
"""
from skfinfo import SkfInfo

PARAMPATH = './newskf'

def ParDict():
    elements = ['H', 'C', 'N', 'O']
    parDict = {el1 + '-' + el2: SkfInfo(el1 + '-' + el2, paramPath=PARAMPATH)
               for el1 in elements for el2 in elements}
    # Copying over spin constants for homonuclear interactions. This set will not include
    # Au-Au or S-S, only C, N, O, H
    # H:
    parDict['H-H'].SetAtomProp('Wss', -0.07174)
    # C:
    parDict['C-C'].SetAtomProp('Wss', -0.03062)
    parDict['C-C'].SetAtomProp('Wsp', -0.02505)
    parDict['C-C'].SetAtomProp('Wpp', -0.02265)
    # N:
    parDict['N-N'].SetAtomProp('Wss', -0.03318)
    parDict['N-N'].SetAtomProp('Wsp', -0.02755)
    parDict['N-N'].SetAtomProp('Wpp', -0.02545)
    # O: 
    parDict['O-O'].SetAtomProp('Wss', -0.03524)
    parDict['O-O'].SetAtomProp('Wsp', -0.02956)
    parDict['O-O'].SetAtomProp('Wpp', -0.02785)
    
    return parDict