
ELEMENTS    =   {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 79: 'Au'}

GROUP_S     =   [1]
GROUP_SP    =   [6, 7, 8]
GROUP_SPD   =   [16, 79]

NUMSHELL    =   {'s': 1, 'sp': 2, 'spd': 3, 'spdf': 4}
NUMBASIS    =   {'s': 1, 'sp': 4, 'spd': 9, 'spdf': 16}

EXCEPTION   =   'Atomic number %d is not supported.'


class AtomEntry(object):
    
    def __init__(self, cartEntry):
        self.coord = cartEntry[1:]
        self.elemNum = int(cartEntry[0])
        self.elem = ELEMENTS[self.elemNum]
        self.orbType = _OrbType(self.elemNum)
        self.numShell = NUMSHELL[self.orbType]
        self.numBasis = NUMBASIS[self.orbType]

def _OrbType(elemNum):
    if elemNum in GROUP_S:
        return 's'
    elif elemNum in GROUP_SP:
        return 'sp'
    elif elemNum in GROUP_SPD:
        return 'spd'
    else:
        raise Exception(EXCEPTION % elemNum)

