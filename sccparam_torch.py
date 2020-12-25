
import numpy as np
import scipy.linalg
from util import TriuToSymm, BMatArr

'''
Switching over to a torch tensor version of the sccparam code so that
pytorch gradients can be retained throughout the calculation
'''

TOLSAMEDIST = 1.0e-5
MINHUBDIFF  = 0.3125e-5

# Helpers of (regular) gamma
def _Gamma12(r12, hub1, hub2):
    tau1, tau2, smallHubDiff = _Tau1Tau2SmallHubDiff(hub1, hub2)
    if r12 < TOLSAMEDIST:
        return _GammaDiag(hub1, hub2)
    elif smallHubDiff:
        _, termExp, term1, term2 = _TermsSmallHubDiff(r12, tau1, tau2)
        expr = termExp * (term1 + term2)
    else:
        expr = _Expr(r12, tau1, tau2) + _Expr(r12, tau2, tau1)
    return 1.0 / r12 - expr

def _Tau1Tau2SmallHubDiff(hub1, hub2):
    return 3.2 * hub1, 3.2 * hub2, abs(hub1 - hub2) < MINHUBDIFF

def _GammaDiag(hub1, hub2):
    tau1, tau2, smallHubDiff = _Tau1Tau2SmallHubDiff(hub1, hub2)
    p12, s12 = tau1 * tau2, tau1 + tau2
    pOverS = p12 / s12
    return 0.5 * (hub1 + hub2 if smallHubDiff else pOverS + pOverS**2 / s12)

def _Expr(r12, tau1, tau2):
    termExp, term1, term2, term3 = _TermsExpr(r12, tau1, tau2)
    return termExp * (term1 - term2 / term3)

def _TermsSmallHubDiff(r12, tau1, tau2):
    tauMean = 0.5 * (tau1 + tau2)
    termExp = np.exp(-tauMean * r12)
    term1 = 1.0 / r12 + 0.6875 * tauMean + 0.1875 * r12 * tauMean**2
    term2 = 0.02083333333333333333 * r12**2 * tauMean**3
    return tauMean, termExp, term1, term2

def _TermsExpr(r12, tau1, tau2):
    sq1, sq2 = tau1**2, tau2**2
    sq1msq2 = sq1 - sq2
    quad2 = sq2**2
    termExp = np.exp(-tau1 * r12)
    term1 = 0.5 * quad2 * tau1 / sq1msq2**2
    term2 = sq2**3 - 3.0 * quad2 * sq1
    term3 = r12 * sq1msq2**3
    return termExp, term1, term2, term3

# Helpers of gamma derivativs (wrt distance r12)
def _AtomShell(atomList):
    atomShell = []
    for atom in atomList:
        curInd = sum([len(shell) for shell in atomShell])
        atomShell += [list(range(curInd, curInd + atom.numShell))]
    return atomShell

def _Gamma12Deriv(r12, hub1, hub2):
    tau1, tau2, smallHubDiff = _Tau1Tau2SmallHubDiff(hub1, hub2)
    if r12 < TOLSAMEDIST:
        return 0.0
    elif smallHubDiff:
        tauMean, termExp, term1, term2 = _TermsSmallHubDiff(r12, tau1, tau2)
        term3 = -1.0 / r12**2 + 0.1875 * tauMean**2 + 2.0 * term2 / r12
        expr = termExp * (-tauMean * (term1 + term2) + term3)
    else:
        expr = _ExprDeriv(r12, tau1, tau2) + _ExprDeriv(r12, tau2, tau1)
    return -1.0 / r12**2 - expr

def _ExprDeriv(r12, tau1, tau2):
    termExp, term1, term2, term3 = _TermsExpr(r12, tau1, tau2)
    term4 = r12 * term3
    return -tau1 * termExp * (term1 - term2 / term3) + termExp * term2 / term4

# The gamma matrix in regular scc-dftb
class Gamma(object):
    
    def __init__(self, atomList, parDict, dftb, batch = None):
        self.__atomList = atomList
        self.__parDict = parDict
        self._batch = batch
        self._dftb = dftb
        
    
    def Matrix(self, gammaFunc=_Gamma12):
        shellList = []
        for atom in self.__atomList:
            par = self.__parDict[atom.elem + '-' + atom.elem]
            orbType = atom.orbType
            sShell = [(atom, par.GetAtomProp('Us'))]
            pShell = [(atom, par.GetAtomProp('Up'))] if 'p' in orbType else []
            dShell = [(atom, par.GetAtomProp('Ud'))] if 'd' in orbType else []
            shellList += sShell + pShell + dShell
        numShell = len(shellList)
        mat = np.zeros((numShell, numShell))
        for ind1, (atom1, hub1) in enumerate(shellList):
            for ind2, (atom2, hub2) in enumerate(shellList[ind1:]):
                r12 = np.linalg.norm(atom2.coord - atom1.coord)
                mat[ind1, ind1 + ind2] = gammaFunc(r12, hub1, hub2)
        res = TriuToSymm(mat)
        if self._batch is not None:
            self.Matrix_batch()
        return res
        
    def Matrix_batch(self, gammaFunc=_Gamma12):
        #raise Exception('Matrix_batch not yet implemented for gamma')
        self._batch.set_oper('G')
        shellList = []
        for iatom, atom in enumerate( self.__atomList ):
            par = self.__parDict[atom.elem + '-' + atom.elem]
            orbType = atom.orbType
            sShell = [(iatom, atom, 's', par.GetAtomProp('Us'))]
            pShell = [(iatom, atom, 'p', par.GetAtomProp('Up'))] if 'p' in orbType else []
            dShell = [(iatom, atom, 'd', par.GetAtomProp('Ud'))] if 'd' in orbType else []
            shellList += sShell + pShell + dShell
        numShell = len(shellList)
        # will hold pointer values returned by batch.add_raw
        mat = np.zeros((numShell, numShell))
        for ind1, (iatom1, atom1, shell1, hub1) in enumerate(shellList):
            for ind2, (iatom2, atom2, shell2, hub2) in enumerate(shellList[ind1:]):
                if (atom1 == atom2):
                    self._batch.set_atoms( (iatom1,) )
                    self._batch.set_Zs( (atom1.elemNum,) )
                    if (shell1 > shell2):
                        orb_type = shell2 + shell1
                    else:
                        orb_type = shell1 + shell2
                    r12 = 0.0
                    flip_atoms = False
                else:
                    self._batch.set_atoms((iatom1,iatom2))
                    Zs = (atom1.elemNum, atom2.elemNum)
                    self._batch.set_Zs(Zs)
                    # want to make sure each type of interaction included only
                    # once in the model types. 
                    if (Zs[0] < Zs[1]) or    \
                       ((Zs[0] == Zs[1]) and (shell1 > shell2)):
                        flip_atoms = True
                        orb_type =  shell2+shell1
                    else:
                        flip_atoms = False
                        orb_type = shell1 + shell2
                    r12 = np.linalg.norm(atom2.coord - atom1.coord)
                dftb_val = gammaFunc(r12, hub1, hub2)
                raw = self._batch.add_raw(orb_type, dftb_val,
                          r12,flip_atoms)
                res_ind = self._batch.add_rot([raw],np.array([1]))
                mat[ind1, ind1 + ind2] = res_ind[0]
                gamma_shell = TriuToSymm(mat)
                gamma_full = self._dftb.ShellToFullBasis(gamma_shell)
                self._batch.set_mol_oper( gamma_full )

    
    def Deriv1List(self):
        distDeriv1Mat = self.Matrix(gammaFunc=_Gamma12Deriv)
        shellCoord = np.array(sum([[atom.coord] * atom.numShell
                                    for atom in self.__atomList], []))
        numShell = len(shellCoord)
        deriv1List = []
        for shell, atom1 in zip(_AtomShell(self.__atomList), self.__atomList):
            coord1 = atom1.coord
            distDeriv1Shell = distDeriv1Mat[shell, :]
            offInd = list(range(0, shell[0])) + list(range(shell[-1] + 1, numShell))
            rOff = np.array([np.linalg.norm(coord1 - coord2)
                             for coord2 in shellCoord[offInd]])
            xyz = []
            for ci in range(3):
                dirDiff = np.zeros(numShell)
                dirDiff[offInd] = (coord1[ci] - shellCoord[offInd, ci]) / rOff
                xyz += [distDeriv1Shell * dirDiff]
            deriv1List += [{'shellInd': shell, 'xyz': xyz}]
        return deriv1List

# The W matrix in spin-polarized (spin-unrestricted) scc-dftb
class Doublju(object):
    
    def __init__(self, atomList, parDict):
        self.__atomList = atomList
        self.__parDict = parDict
    
    def Matrix(self):
        nullMat = np.zeros((1, 0))
        emptyMat = np.zeros((0, 0))
        diag = []
        for atom in self.__atomList:
            par = self.__parDict[atom.elem + '-' + atom.elem]
            pOnAtom = 'p' in atom.orbType
            dOnAtom = 'd' in atom.orbType
            ss = np.array([[par.GetAtomProp('Wss')]])
            sp = np.array([[par.GetAtomProp('Wsp')]]) if pOnAtom else nullMat
            pp = np.array([[par.GetAtomProp('Wpp')]]) if pOnAtom else emptyMat
            sd = np.array([[par.GetAtomProp('Wsd')]]) if dOnAtom else nullMat
            pdDum = np.zeros((sp.shape[1], 0))
            pd = np.array([[par.GetAtomProp('Wpd')]]) if dOnAtom else pdDum
            dd = np.array([[par.GetAtomProp('Wdd')]]) if dOnAtom else emptyMat
            diag += [BMatArr([[ss, sp, sd], [sp.T, pp, pd], [sd.T, pd.T, dd]])]
        return scipy.linalg.block_diag(*diag)

