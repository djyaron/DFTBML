import numpy as np
from collections import deque

class CDIIS:

    def __init__(self, overlap, maxNumFock=20):
        self._overlap = overlap
        self._fockQue = deque(maxlen=maxNumFock)
        self._commQue = deque(maxlen=maxNumFock)

    def NewFock(self, fockList, comm):
        self._fockQue.append(fockList)
        self._commQue.append(comm)
        coeff = np.zeros(len(self._commQue))
        coeff[-1] = 1.0
        for numUse in range(len(self._commQue), 1, -1):
            commList = [x for x in self._commQue]
            commMat = np.array(commList[-numUse:])
            ones = np.ones((numUse, 1))
            cdiisMat = np.bmat([[commMat.dot(commMat.T),   ones            ],
                                [ones.T                ,   np.zeros((1, 1))]])
            rightSide = np.zeros(numUse + 1)
            rightSide[-1] = 1.0
            try:
                coeffUse = np.linalg.solve(cdiisMat, rightSide)[0:-1]
                coeff = np.zeros(len(self._commQue))
                coeff[-numUse:] = coeffUse
                break
            except:
                continue
        sh = self._fockQue[0][0].shape
        return [coeff.dot([fl[sp].ravel() for fl in self._fockQue]).reshape(sh)
                for sp in range(len(fockList))]



