
import numpy as np
from scipy.spatial.distance import pdist, squareform
from util import TriuToSymm

class Repulsion(object):
    
    def __init__(self, atomList, parDict):
        self._atomList = atomList
        self.__parDict = parDict
    
    def Energy(self):
        return self.__TriuMatrix(getRepFunc=_GetRepEnergy).sum()
        
    def rep_matrix(self):
        return self.__TriuMatrix(getRepFunc=_GetRepEnergy)        
    
    def XYZDeriv1(self):
        numAtom = len(self._atomList)
        atomCoord = np.array([atom.coord for atom in self._atomList])
        deriv1Mat = np.zeros((numAtom, 3))
        distMat = squareform(pdist(atomCoord, 'euclidean', p=2))
        triuMat = TriuToSymm(self.__TriuMatrix(getRepFunc=_GetRepDeriv1))
        zipList = list(zip(self._atomList, triuMat, distMat))
        for ind, (atom, distDeriv1, distVec) in enumerate(zipList):
            offInd = list(range(0, ind)) + list(range(ind + 1, numAtom))
            rOff = distVec[offInd][:, np.newaxis]
            deriv1dir = (atom.coord - atomCoord[offInd, :]) / rOff
            deriv1Mat[ind, :] = distDeriv1[offInd].dot(deriv1dir)
        return deriv1Mat
    
    def __TriuMatrix(self, getRepFunc):
        numAtom = len(self._atomList)
        repTriuMat = np.zeros((numAtom, numAtom))
        for ind1, atom1 in enumerate(self._atomList):
            for ind2, atom2 in enumerate(self._atomList[(ind1 + 1):]):
                dist = np.linalg.norm(atom2.coord - atom1.coord)
                parDict = self.__parDict[atom1.elem + '-' + atom2.elem]
                repTriuMat[ind1, ind1 + ind2 + 1] = getRepFunc(parDict, dist)
        return repTriuMat

    def add_to_batch(self, glabel, batch):
        numAtom = len(self._atomList)
        nvals = int((numAtom**2 - numAtom)/2)
        indices = np.zeros([nvals], dtype = np.int)
        icount = 0
        for ind1, atom1 in enumerate(self._atomList):
            for ind2, atom2 in enumerate(self._atomList):
                if ind2 <= ind1:
                    continue
                dist = np.linalg.norm(atom2.coord - atom1.coord)
                parDict = self.__parDict[atom1.elem + '-' + atom2.elem]
                dftb_val = _GetRepEnergy(parDict, dist)
                Zs = [atom1.elemNum,atom2.elemNum]
                atoms = [ind1, ind2]
                indices[icount] = batch.add_raw_rep(glabel,Zs,atoms,dftb_val,dist)
                icount += 1
        batch.set_rep_oper(glabel,indices)
def _GetRepEnergy(parDict, dist):
    return parDict.GetRep(dist)

def _GetRepDeriv1(parDict, dist):
    return parDict.GetRepDeriv1(dist)


