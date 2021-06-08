
import numpy as np
import copy
from .util import TriuToSymm, BMatArr

FINITEDIFF = 1e-8

class _MatElem(object): # abstract
    # Returns full matrix
    def Matrix(self):
        matrix = []
        for ind1, atom1 in enumerate(self._atomList):
            diagBlock = self._diag.Block(atom1)
            zeroBlockWidth = sum([rowBlk[0].shape[0] for rowBlk in matrix])
            zeroBlock = np.zeros((diagBlock.shape[1], zeroBlockWidth))
            blocks = [BMatArr([zeroBlock, diagBlock])]
            for i2, atom2 in enumerate( self._atomList[(ind1 + 1):]):
                blocks += [self._offDiag.Block(atom1, atom2)]
            matrix += [[BMatArr(blocks)]]
        res = TriuToSymm(BMatArr(matrix))
        if self._batch is not None:
            self.Matrix_batch()
        return res
        
    def Matrix_batch(self):
        batch = self._batch
        self._diag._compute_batch = True
        self._offDiag._compute_batch = True
        matrix = []
        for ind1, atom1 in enumerate(self._atomList):
            batch.set_atoms((ind1,ind1))
            diagBlock = self._diag.Block(atom1)
            zeroBlockWidth = sum([rowBlk[0].shape[0] for rowBlk in matrix])
            zeroBlock = np.zeros((diagBlock.shape[1], zeroBlockWidth))
            blocks = [BMatArr([zeroBlock, diagBlock])]
            for i2, atom2 in enumerate( self._atomList[(ind1 + 1):]):
                self._batch.set_atoms((ind1,ind1+1+i2))
                blocks += [self._offDiag.Block(atom1, atom2)]
            matrix += [[BMatArr(blocks)]]
        batch.set_mol_oper( TriuToSymm(BMatArr(matrix)) )
        self._diag._compute_batch = False
        self._offDiag._compute_batch = False
    
    # Returns a list 'deriv1List' of length numAtom
    # Each element is a dictionary with keys 'basisInd' and 'xyz'
    # deriv1List[0]['basisInd'] is a list of basis indices
    # deriv1List[0]['xyz'] is a length 3 list of x, y, z partial derivatives
    def Deriv1List(self):
        deriv1List = []
        for ind, atom in enumerate(self._atomList):
            start = sum([der['xyz'][0].shape[0] for der in deriv1List])
            basisInd = list(range(start, start + atom.numBasis))
            blockRef = self.__Deriv1RowBlock(ind, atom)
            xyz = [self.__FinDiff(ind, atom, ci, blockRef) for ci in range(3)]
            deriv1List += [{'basisInd': basisInd, 'xyz': xyz}]
        return deriv1List
    
    def __FinDiff(self, ind, atom, coordInd, blockRef):
        atomCopy = copy.deepcopy(atom)
        atomCopy.coord[coordInd] += FINITEDIFF
        return (self.__Deriv1RowBlock(ind, atomCopy) - blockRef) / FINITEDIFF
    
    def __Deriv1RowBlock(self, ind1, atom1):
        offBlk = self._offDiag.Block
        block = [offBlk(atom1, atom2) for atom2 in self._atomList[:ind1]]
        block += [np.zeros((atom1.numBasis, atom1.numBasis))]
        block += [offBlk(atom1, atom2) for atom2 in self._atomList[(ind1 + 1):]]
        return BMatArr(block)

class Overlap(_MatElem):
    
    def __init__(self, atomList, parDict, batch = None):
        self._atomList = atomList
        self._diag = _SDiag()
        self._offDiag = _SOffDiag(parDict,batch)
        self._batch = batch
        if batch is not None:
            self._batch.set_oper('S')

class CoreH(_MatElem):
    
    def __init__(self, atomList, parDict, batch = None):
        self._atomList = atomList
        self._diag = _HDiag(parDict, batch)
        self._offDiag = _HOffDiag(parDict, batch)
        self._batch = batch
        if batch is not None:
            self._batch.set_oper('H')

class _SDiag(object):
    def __init__(self, batch = None):
        self._batch = batch
        self._compute_batch = False
    
    def Block(self, atom):
        if self._compute_batch:
            return np.zeros([atom.numBasis,atom.numBasis])
        else:
            return np.eye(atom.numBasis)

class _HDiag(object):
    
    def __init__(self, parDict, batch = None):
        self.__parDict = parDict
        self._batch = batch
        self._compute_batch = False
        
    def Block(self, atom):
        par = self.__parDict[atom.elem + '-' + atom.elem]
        sDiag = [par.GetAtomProp('Es')]
        pDiag = [par.GetAtomProp('Ep')] * 3 if 'p' in atom.orbType else []
        dDiag = [par.GetAtomProp('Ed')] * 5 if 'd' in atom.orbType else []
        
        if self._compute_batch:
            batch = self._batch
            batch.set_Zs( (atom.elemNum,) )
            sRaw = batch.add_raw('s', sDiag[0])
            sInd = batch.add_rot([sRaw], np.array([1]))
            if 'p' in atom.orbType:
                pRaw = batch.add_raw('p', pDiag[0])
                pInd = batch.add_rot([pRaw], np.ones([3,1]) )
            else:
                pInd = []
            if 'd' in atom.orbType: 
                dRaw = batch.add_raw('d', dDiag[0])
                dInd = batch.add_rot([dRaw], np.ones([5,1]) )
            else:
                dInd = []
            return np.diag(sInd + pInd + dInd)

        return np.diag(sDiag + pDiag + dDiag)


class _OffDiag(object): # abstract
    
    def __init__(self, parDict, batch):
        self.__parDict = parDict
        self._batch = batch
        self._compute_batch = False
    
    def Block(self, atom1, atom2):
        dist = np.linalg.norm(atom1.coord - atom2.coord)
        if dist < np.finfo(float).eps:
            raise Exception('Atoms clash; %s %s and %s %s are too close.'
                        % (atom1.elem, atom1.coord, atom2.elem, atom2.coord))
 
        if self._compute_batch:
            self._batch.set_Zs( (atom1.elemNum, atom2.elemNum) )

        xyz = (atom2.coord - atom1.coord) / dist
        par = self.__parDict[atom1.elem + '-' + atom2.elem]
        par21 = self.__parDict[atom2.elem + '-' + atom1.elem]
        pOn1, pOn2 = 'p' in atom1.orbType, 'p' in atom2.orbType
        dOn1, dOn2 = 'd' in atom1.orbType, 'd' in atom2.orbType
        sDum = np.zeros((1, 0))
        ssBlk = self._SsBlk(par, dist)
        spBlk = self._SpBlk(par, xyz, dist) if pOn2 else sDum
        psBlk = self._SpBlk(par21, -xyz, dist,True).T if pOn1 else sDum.T
        ppDum = np.zeros((psBlk.shape[0], spBlk.shape[1]))
        ppBlk = self._PpBlk(par, xyz, dist) if pOn1 and pOn2 else ppDum
        sdBlk = self._SdBlk(par, xyz, dist) if dOn2 else sDum
        dsBlk = self._SdBlk(par21, -xyz, dist,True).T if dOn1 else sDum.T
        pdDum = np.zeros((psBlk.shape[0], sdBlk.shape[1]))
        pdBlk = self._PdBlk(par, xyz, dist) if pOn1 and dOn2 else pdDum
        dpDum = np.zeros((dsBlk.shape[0], spBlk.shape[1]))
        dpBlk = self._PdBlk(par21, -xyz, dist,True).T if dOn1 and pOn2 else dpDum
        ddDum = np.zeros((dsBlk.shape[0], sdBlk.shape[1]))
        ddBlk = self._DdBlk(par, xyz, dist) if dOn1 and dOn2 else ddDum
        return BMatArr([[ssBlk, spBlk, sdBlk],
                        [psBlk, ppBlk, pdBlk],
                        [dsBlk, dpBlk, ddBlk]])

    def _SsBlk(self, par, dist):
        res = np.array([[par.GetSkInt(self._ss0, dist)]])
        
        if self._compute_batch:
            batch = self._batch
            raw = batch.add_raw('ss', res[0][0], dist)
            ind = batch.add_rot([raw], np.array([1]) )
            return np.array(ind).reshape([1,1])          

        return res
        
    def _SpBlk(self, par, xyzVec, dist,flipped=False):
        #return np.array([xyzVec]) * par.GetSkInt(self._sp0, dist)
        integral = par.GetSkInt(self._sp0, dist)
        rot = np.array([xyzVec])
        res = rot * integral
        
        if self._compute_batch:
            batch = self._batch
            raw = batch.add_raw('sp', integral, dist, flipped)
            ind = batch.add_rot([raw], rot.T)
            return np.array(ind).reshape([1,3])
            
        return res
    
    def _PpBlk(self, par, xyzVec, dist, flipped = False):
        x, y, z = xyzVec
        xsq, ysq, zsq = xyzVec**2
        xy, yz, zx = x * y, y * z, z * x
        rot = [[xsq, 1 - xsq],    # px-px
               [xy, -xy],         # px-py
               [zx, -zx],         # px-pz
               #
               [0.0, 0.0],        # py-px
               [ysq, 1 - ysq],    # py-py
               [yz, -yz],         # py-pz
               #
               [0.0, 0.0],        # pz-px
               [0.0, 0.0],        # pz-py
               [zsq, 1 - zsq]]    # pz-pz

        integral = [[par.GetSkInt(self._pp0, dist)],
                    [par.GetSkInt(self._pp1, dist)]]
        res = TriuToSymm(np.array(rot).dot(integral).reshape(3, 3))
        
        if self._compute_batch:
            rot = [[xsq, 1 - xsq],    # px-px
                   [xy, -xy],         # px-py
                   [zx, -zx],         # px-pz
                   #
                   [xy, -xy],        # py-px
                   [ysq, 1 - ysq],    # py-py
                   [yz, -yz],         # py-pz
                   #
                   [zx, -zx],        # pz-px
                   [yz, -yz],        # pz-py
                   [zsq, 1 - zsq]]    # pz-pz            
            batch = self._batch
            raw_pp0 = batch.add_raw('pp_sigma',integral[0][0], dist, flipped)
            raw_pp1 = batch.add_raw('pp_pi',integral[1][0], dist, flipped)
            ind = batch.add_rot([raw_pp0,raw_pp1],np.array(rot))
            return np.array(ind).reshape(3,3)            
            
        return res
    
    def _SdBlk(self, par, xyzVec, dist, flipped=False):
        rt3 = np.sqrt(3.0)
        x, y, z = xyzVec
        xsq, ysq, zsq = xyzVec**2
        rot = [rt3 * x * y,
               rt3 * y * z,
               rt3 * x * z,
               0.5 * rt3 * (xsq - ysq),
               zsq - 0.5 * (xsq + ysq)]
        integral = par.GetSkInt(self._sd0, dist)
        res= np.array([rot]) * integral
        
        if self._compute_batch:
            batch = self._batch
            raw = batch.add_raw('sd', integral, dist, flipped)
            rot_array = np.array(rot).reshape(-1,1)
            ind = batch.add_rot([raw], rot_array)
            return np.array(ind).reshape(1,5)
            
        return res
    
    def _PdBlk(self, par, xyzVec, dist, flipped=False):
        rt3 = np.sqrt(3.0)
        hRt3 = 0.5 * rt3
        x, y, z = xyzVec
        xsq, ysq, zsq = xyzVec**2
        xyz = x * y * z
        alpha, beta = xsq + ysq, xsq - ysq
        zsqMHA = zsq - 0.5 * alpha
        rot = [[rt3 * xsq * y, y * (1 - 2 * xsq)],      # px-dxy
               [rt3 * xyz, -2 * xyz],                   # px-dyz
               [rt3 * xsq * z, z * (1 - 2 * xsq)],      # px-dzx
               [hRt3 * x * beta, x * (1 - beta)],       # px-d(x2-y2)
               [x * zsqMHA, -rt3 * x * zsq],            # px-d(3z2-r2)
               #
               [rt3 * ysq * x, x * (1 - 2 * ysq)],      # py-dxy
               [rt3 * ysq * z, z * (1 - 2 * ysq)],      # py-dyz
               [rt3 * xyz, -2 * xyz],                   # py-dzx
               [hRt3 * y * beta, -y * (1 + beta)],      # py-d(x2-y2)
               [y * zsqMHA, -rt3 * y * zsq],            # py-d(3z2-r2)
               #
               [rt3 * xyz, -2 * xyz],                   # pz-dxy
               [rt3 * zsq * y, y * (1 - 2 * zsq)],      # pz-dyz
               [rt3 * zsq * x, x * (1 - 2 * zsq)],      # pz-dzx
               [hRt3 * z * beta, -z * beta],            # pz-d(x2-y2)
               [z * zsqMHA, rt3 * z * alpha]]           # pz-d(3z2-r2)
        integral = [[par.GetSkInt(self._pd0, dist)],
                    [par.GetSkInt(self._pd1, dist)]]
                    
        if self._compute_batch:
            batch = self._batch
            raw_pd0 = batch.add_raw('pd_sigma',integral[0][0], dist, flipped)
            raw_pd1 = batch.add_raw('pd_pi',integral[1][0], dist, flipped)
            ind = batch.add_rot([raw_pd0,raw_pd1],np.array(rot))
            return np.array(ind).reshape(3,5)            

        return np.array(rot).dot(integral).reshape(3, 5)
    
    def _DdBlk(self, par, xyzVec, dist, flipped= False):
        rt3 = np.sqrt(3.0)
        hRt3 = 0.5 * rt3
        x, y, z = xyzVec
        xsq, ysq, zsq = xyzVec**2
        xy, yz, zx, xyz = x * y, y * z, z * x, x * y * z
        xsqysq, ysqzsq, zsqxsq = xsq * ysq, ysq * zsq, zsq * xsq
        alpha, beta = xsq + ysq, xsq - ysq
        zsqMHA, aMZsq, betasq = zsq - 0.5 * alpha, alpha - zsq, beta**2
        rt3Be = rt3 * beta
        rot = [[3 * xsqysq, alpha - 4 * xsqysq, zsq + xsqysq],                  # dxy-dxy
               [3 * xyz * y, zx * (1 - 4 * ysq), zx * (ysq - 1)],               # dxy-dyz
               [3 * xyz * x, yz * (1 - 4 * xsq), yz * (xsq - 1)],               # dxy-dzx
               [1.5 * xy * beta, -2 * xy * beta, 0.5 * xy * beta],              # dxy-d(x2-y2)
               [rt3 * xy * zsqMHA, -2 * rt3 * xyz * z, hRt3 * xy * (1 + zsq)],  # dxy-d(3z2-r2)
               #
               [0.0, 0.0, 0.0],                                                 # dyz-dxy
               [3 * ysqzsq, ysq + zsq - 4 * ysqzsq, xsq + ysqzsq],              # dyz-dyz
               [3 * xyz * z, xy * (1 - 4 * zsq), xy * (zsq - 1)],               # dyz-dzx
               [1.5 * yz * beta, -yz * (1 + 2 * beta), yz * (1 + 0.5 * beta)],  # dyz-d(x2-y2)
               [rt3 * yz * zsqMHA, rt3 * yz * aMZsq, -hRt3 * yz * alpha],       # dyz-d(3z2 - r2)
               #
               [0.0, 0.0, 0.0],                                                 # dzx-dxy
               [0.0, 0.0, 0.0],                                                 # dzx-dyz
               [3 * zsqxsq, zsq + xsq - 4 * zsqxsq, ysq + zsqxsq],              # dzx-dzx
               [1.5 * zx * beta, zx * (1 - 2 * beta), -zx * (1 - 0.5 * beta)],  # dzx-d(x2-y2)
               [rt3 * zx * zsqMHA, rt3 * zx * aMZsq, -hRt3 * zx * alpha],       # dzx-d(3z2-r2)
               #
               [0.0, 0.0, 0.0],                                                 # d(x2-y2)-dxy
               [0.0, 0.0, 0.0],                                                 # d(x2-y2)-dyz
               [0.0, 0.0, 0.0],                                                 # d(x2-y2)-dzx
               [0.75 * betasq, alpha - betasq, zsq + 0.25 * betasq],            # d(x2-y2)-d(x2-y2)
               [0.5 * rt3Be * zsqMHA, -zsq * rt3Be, 0.25 * rt3Be * (1 + zsq)],  # d(x2-y2)-d(3z2-r2)
               #
               [0.0, 0.0, 0.0],                                                 # d(3z2-r2)-dxy
               [0.0, 0.0, 0.0],                                                 # d(3z2-r2)-dyz
               [0.0, 0.0, 0.0],                                                 # d(3z2-r2)-dzx
               [0.0, 0.0, 0.0],                                                 # d(3z2-r2)-d(x2-y2)
               [zsqMHA**2, 3 * zsq * alpha, 0.75 * alpha**2]]                   # d(3z2-r2)-d(3z2-r2)
        integral = [[par.GetSkInt(self._dd0, dist)],
                    [par.GetSkInt(self._dd1, dist)],
                    [par.GetSkInt(self._dd2, dist)]]
                    
        if self._compute_batch:
            # contains only the upper triangle values 
            rot_array_triu = np.array(rot)
            batch = self._batch
            raw0 = batch.add_raw('dd_sigma',integral[0][0], dist, flipped)
            raw1 = batch.add_raw('dd_pi',   integral[1][0], dist, flipped)
            raw2 = batch.add_raw('dd_delta',integral[2][0], dist, flipped)
            rot_array = np.zeros([25,3])
            for j in range(3):
                triu = rot_array_triu[:,j].reshape(5,5)
                full = TriuToSymm(triu)
                rot_array[:,j] = full.flatten()
            ind = batch.add_rot([raw0,raw1,raw2],rot_array)
            return np.array(ind).reshape(5,5)            

        return TriuToSymm(np.array(rot).dot(integral).reshape(5, 5))

class _SOffDiag(_OffDiag):
    _ss0, _sp0, _pp0, _pp1  =   'Sss0', 'Ssp0', 'Spp0', 'Spp1'
    _sd0, _pd0, _pd1        =   'Ssd0', 'Spd0', 'Spd1'
    _dd0, _dd1, _dd2        =   'Sdd0', 'Sdd1', 'Sdd2'

class _HOffDiag(_OffDiag):
    _ss0, _sp0, _pp0, _pp1  =   'Hss0', 'Hsp0', 'Hpp0', 'Hpp1'
    _sd0, _pd0, _pd1        =   'Hsd0', 'Hpd0', 'Hpd1'
    _dd0, _dd1, _dd2        =   'Hdd0', 'Hdd1', 'Hdd2'

