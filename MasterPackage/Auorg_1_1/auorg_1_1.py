# General parameter interface information:
# Dictionary: par = parDict['C-N']
# Constructor: par = SkfInfo(name='C-N', paramPath='./mio-0-1')
# Getters:
#   par.GetSkInt(key='Hss0', dist=1.5)
#   par.GetAtomProp(key='Es')
#   par.GetRep(dist=1.5)

# SKF parameter interface:
from skfinfo import SkfInfo

PARAMPATH = './auorg-1-1'

def ParDict():
    elements = ['H', 'C', 'N', 'O', 'S', 'Au']
    parDict = {el1 + '-' + el2: SkfInfo(el1 + '-' + el2, paramPath=PARAMPATH)
               for el1 in elements for el2 in elements}
    # Spin constants (parameter W, doublju)
    # H:
    #      -0.07174
    parDict['H-H'].SetAtomProp('Wss', -0.07174)
    # C:
    #      -0.03062     -0.02505
    #      -0.02505     -0.02265
    parDict['C-C'].SetAtomProp('Wss', -0.03062)
    parDict['C-C'].SetAtomProp('Wsp', -0.02505)
    parDict['C-C'].SetAtomProp('Wpp', -0.02265)
    # N:
    #      -0.03318     -0.02755
    #      -0.02755     -0.02545
    parDict['N-N'].SetAtomProp('Wss', -0.03318)
    parDict['N-N'].SetAtomProp('Wsp', -0.02755)
    parDict['N-N'].SetAtomProp('Wpp', -0.02545)
    # O:
    #      -0.03524     -0.02956
    #      -0.02956     -0.02785
    parDict['O-O'].SetAtomProp('Wss', -0.03524)
    parDict['O-O'].SetAtomProp('Wsp', -0.02956)
    parDict['O-O'].SetAtomProp('Wpp', -0.02785)
    # S:
    #      -0.02137     -0.01699     -0.01699
    #      -0.01699     -0.01549     -0.01549
    #      -0.01699     -0.01549     -0.01549
    parDict['S-S'].SetAtomProp('Wss', -0.02137)
    parDict['S-S'].SetAtomProp('Wsp', -0.01699)
    parDict['S-S'].SetAtomProp('Wsd', -0.01699)
    parDict['S-S'].SetAtomProp('Wpp', -0.01549)
    parDict['S-S'].SetAtomProp('Wpd', -0.01549)
    parDict['S-S'].SetAtomProp('Wdd', -0.01549)
    # Au:
    #      -0.01304     -0.01304   -0.00525
    #      -0.01304     -0.01304   -0.00525
    #      -0.00525     -0.00525   -0.01082
    parDict['Au-Au'].SetAtomProp('Wss', -0.01304)
    parDict['Au-Au'].SetAtomProp('Wsp', -0.01304)
    parDict['Au-Au'].SetAtomProp('Wsd', -0.00525)
    parDict['Au-Au'].SetAtomProp('Wpp', -0.01304)
    parDict['Au-Au'].SetAtomProp('Wpd', -0.00525)
    parDict['Au-Au'].SetAtomProp('Wdd', -0.01082)
    return parDict
    
