# General parameter interface information:
# Dictionary: par = parDict['C-N']
# Constructor: par = SkfInfo(name='C-N', paramPath='./mio-0-1')
# Getters:
#   par.GetSkInt(key='Hss0', dist=1.5)
#   par.GetAtomProp(key='Es')
#   par.GetRep(dist=1.5)

# SKF parameter interface:
import os
from skfinfo import SkfInfo

def ParDict(paramPath : str) -> dict:
    elements = ['H', 'C', 'N', 'O', 'S']
    parDict = {el1 + '-' + el2: SkfInfo(el1 + '-' + el2, paramPath=paramPath)
               for el1 in elements for el2 in elements}
    # Spin constants (parameter W, double)
    # H:
    #      -0.07174
    parDict['H-H'].SetAtomProp('Wss', 0.0)
    # C:
    #      -0.03062     -0.02505
    #      -0.02505     -0.02265
    parDict['C-C'].SetAtomProp('Wss', 0.0)
    parDict['C-C'].SetAtomProp('Wsp', 0.0)
    parDict['C-C'].SetAtomProp('Wpp', 0.0)
    # N:
    #      -0.03318     -0.02755
    #      -0.02755     -0.02545
    parDict['N-N'].SetAtomProp('Wss', 0.0)
    parDict['N-N'].SetAtomProp('Wsp', 0.0)
    parDict['N-N'].SetAtomProp('Wpp', 0.0)
    # O:
    #      -0.03524     -0.02956
    #      -0.02956     -0.02785
    parDict['O-O'].SetAtomProp('Wss', 0.0)
    parDict['O-O'].SetAtomProp('Wsp', 0.0)
    parDict['O-O'].SetAtomProp('Wpp', 0.0)
    # S:
    #      -0.02137     -0.01699     -0.01699
    #      -0.01699     -0.01549     -0.01549
    #      -0.01699     -0.01549     -0.01549
    parDict['S-S'].SetAtomProp('Wss', 0.0)
    parDict['S-S'].SetAtomProp('Wsp', 0.0)
    parDict['S-S'].SetAtomProp('Wsd', 0.0)
    parDict['S-S'].SetAtomProp('Wpp', 0.0)
    parDict['S-S'].SetAtomProp('Wpd', 0.0)
    parDict['S-S'].SetAtomProp('Wdd', 0.0)
    return parDict
