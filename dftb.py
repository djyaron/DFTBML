
import numpy as np
import scipy.linalg
from atomentry import AtomEntry
from repulsion import Repulsion
from matelem import Overlap, CoreH
from sccparam import Gamma, Doublju
from cdiis import CDIIS
from fermilevel import determine_fermi_level, get_entropy_term,\
                       get_occupancies



from Exceptions import ConvergenceFailure

"""
##################################  WARNING ##################################
The safety checks that prevent the use of RHF on systems that contain an odd
number of electrons have been temporarily disabled!
##################################  WARNING ##################################
"""

"""
TODO:
    - Fix the commuter issues, it is currently not necessarily suitable for
        finite temperature systems. Yet its used to regenerate the Fock matrix.
    - Create less hacky method to update the entropy term, currently done in
        the fock_to_dense function.
    - "entropy term" is really the "entropy", the "entropy term" is entropy * sigma 
    - Fix convergence issues associated with running non-finite temperature calculations
"""

# Really should not be used until we figure out what to do
global BYPASS_UHF_SAFETY_CHECK
BYPASS_UHF_SAFETY_CHECK = True

# if BYPASS_UHF_SAFETY_CHECK:
#     print('#'*80);print('#'*80)
#     print('!      WARNING UNRESTRICTED HARTREE-FOCK SAFETY CHECKS HAVE BEEN BYPASSED      !\n'
#           'This will cause fractional numbers to be assigned to ‘_numElecAB’. To disable\n'
#           'this, set ‘BYPASS_UHF_SAFETY_CHECK’ to false in the dftb.py file.')
#     print('#'*80);print('#'*80)


ANGSTROM2BOHR = 1.889725989

class DFTB:

    """
    TODO:
        - Need to convert orbital_ids from list of tuples to structured array
            and make corresponding changes to the PDoS function.
    """
    
    _maxSCFIter = 200
    _thrRmsDens = _thrRmsComm = 1.0e-8
    _thrMaxDens = _thrMaxComm = 1.0e-6

    # Sets the verbosity of the DFTD objects
    verbose = False

    # Raise exception upon SCF convergence failure
    raise_SCF_exception = True

    # Will attempt to recover from a failed fermi-level search
    bypass_fermi_convergence_error = False

    __slots__ = ['smearing', 'smearing_scheme', 'atomList', 'parDict', 'repulsion_class',
                 '_numElecAB', '_overlap', '_coreH', '_gamma', '_doublju',
                 '_qN', 'solFockList', 'fermi_level', 'density_list', 'occ_rho_mask', 'entropy',
                 '_shellBasis', '_onAtom', '_numVEShellNeu', 'orbital_ids', 'is_batch']
    def __init__(self, parDict, cart, charge=0, mult=1,
                 lenUnit='Angstrom', batch=None, glabel=None,
                 smearing=None, smearing_scheme='fermi', *args, **kwargs):
        '''
        Input:
            parDict : Class providing DFTB parameters
            cart    : numpy array with  [:,0]    = Z (element number)
                                        [:,1:3]  = x,y,z coordinates
            charge  : molecular charge
            mult    : multiplicity (e.g. 1 for singlet)
            lenUnit : 'Angstrom' or 'Bohr' for units for cart
                      code uses Bohr internally
            verbose : info messages if True
            batch   : see batch.py, generates info needed for tensorflow
                      implementation of DFTB
            glabel  : used by batch.py to label the geometry
            smearing: None if zero temperature elec distribution
                      otherwise dict passed to determine_fermi_level
                     eg. ['scheme': 'gaussian', 'width': 0.01]
        '''

        # Note, changed smearing to sigma

        if batch is not None:
            batch.set_glabel(glabel)
            self.is_batch = True
            #DFTB.raise_SCF_exception = False
        else:
            self.is_batch = False

        # Code below should be removed once a consensus regarding the
        # finite temperature input format has been reached.
        # Temporary catch for different methods currently in use
        if type(smearing) == dict:
            self.smearing = smearing['width']
            self.smearing_scheme = smearing['scheme']
        else:
            # Sets the smearing width for finite temperature calculations
            self.smearing = smearing

            # Sets the method to be used with smearing
            self.smearing_scheme = smearing_scheme

        # This is now a class property
        # self.verbose = kwargs.get('verbose', False

        cartCopy = np.array(cart)
        # Any internal length is in Bohr
        cartCopy[:, 1:] *= {'Bohr': 1.0, 'Angstrom': ANGSTROM2BOHR}[lenUnit]

        # List of atom type objects
        atomList = [AtomEntry(cartRow) for cartRow in cartCopy]
        self.atomList = atomList

        # List of orbital information, temporary hack
        self.orbital_ids = DFTB.get_orbital_ids(atomList)



        self.parDict = parDict
        argTup = self.atomList, parDict
        self.repulsion_class = Repulsion(*argTup)
        if batch is not None:
            self.repulsion_class.add_to_batch(glabel, batch)
        (self._shellBasis, self._onAtom) = _ShellBasis(self.atomList)
        self._numVEShellNeu = _NumVEShellNeu(*argTup)
        self._numElecAB = _NumElecAB(sum(self._numVEShellNeu) - charge, mult)
        self._overlap = Overlap(self.atomList, parDict, batch).Matrix()
        self._coreH = CoreH(self.atomList, parDict, batch).Matrix()
        self._gamma = Gamma(self.atomList, parDict, self, batch).Matrix()
        self._doublju = Doublju(*argTup).Matrix()
        self._qN = self._calc_qN()
        self.solFockList = None
        self.fermi_level = None
        self.density_list = None
        self.occ_rho_mask = None
        self.entropy = 0.


    @property
    def repulsion(self):
        return self.repulsion_class.Energy()

    @staticmethod
    def get_orbital_ids(atom_list=None):
        """
        Returns a list specifying, in order, the identities of each orbital in
        the form: (a, z, l, m) where a is the atom index z, is the atomic number
        of the atom and l & m are the angular-momentum and magnetic quantum
        number respectively.

        Arguments:
            - atom_list (list): An ordered list of AtomEntry objects.

        Returns:
            - A list of tuples of the form (a, z, l, m).
        """

        # Angular-momentum lookup dictionary
        s2i = {'s': 0, 'p': 1, 'd': 2}

        # Magnetic quantum number lookup dictionary
        m_values = {0: [0], 1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

        # Array to hold the list of orbital data
        orbitals = []

        # Loop over the atoms
        for atom_index, atom in enumerate(atom_list):
            # Loop over the available angular-momentum values
            for l in atom.orbType:
                # Convert the angular-momentum from string to integer
                l = s2i[l]

                # Loop over the associated magnetic quantum numbers
                for m in m_values[l]:
                    # Add a tuple to the "orbitals" list of the form:
                    #   (atomic_number, angular-momentum, magnetic_number)
                    orbitals.append((atom_index, atom.elemNum, l, m))

        # Return the orbitals list
        return orbitals

    def SCF(self, *args, **kwargs):
        """
        Description goes here

        Parameters:
            - H0 (array(float)): Core Hamiltonian matrix
            - n_elec (array(int)): number spin up and spin down electrons

        Optional:
            - guess (string or list): Initial guess of, or method to generate
               an initial guess of the density matrix.
            - get_occ_rho_mask (bool): Specifies whether or not the rho
              occupation mask will be returned.

        Returns:

        Notes:
            - Function should be changed to return a fixed number of
              variables. Non-fixed return counts can be problematic.
              Can be easily handled with a get function or class call.


        TODO:
            - Sort out the handling of the convergence function call, currently
                we have code duplication.
            - Strip spin or add in full handling for it.
            - Should be returning free energy when doing finite temperature
              calculations. Should also project energies to 0 K.

        # NEED TO UPDATE DOC-STRING

        """

        # Check whether the rho occupation mask should be returned
        get_occ_rho_mask = kwargs.get('get_occ_rho_mask', False)

        # Check for initial density matrix or generation method keyword
        density_guess = kwargs.get('guess', 'core')

        if type(density_guess) == list:  # If density matrix was provided, then set it
            density_list = guess
        elif density_guess == 'core':  # If core is specified
            # Then use the core Hamiltonian
            density_list = self._GuessCore()
        else:  # Otherwise throw an exception
            NotImplementedError('Unknown density initialiser provided')

        # Boolean to track convergence 
        converged = False

        # Special method to retry scf convergence with a greater number of
        # fock histories for batch operations
        if kwargs.get('retry_for_convergence', False):
            cdiis = CDIIS(self._overlap, kwargs.get('max_n_fock', 20))

        # Otherwise just act normally
        else:
            cdiis = CDIIS(self._overlap)

        # Start the self constant charge convergence loop
        for numIter in range(self._maxSCFIter + 1):

            # Store the current density matrix
            old_density_list = density_list

            # Compute new Fock matrices
            fock_list = self._DensToFock(density_list)

            # Build the commuter
            commuter = self._Comm(fock_list, density_list)

            # Regenerate the fock matrices
            fock_list = cdiis.NewFock(fock_list, commuter)

            # Calculate the new density matrices
            density_list, occ_rho_mask = self.fock_to_dense(fock_list)

            # <HACK>
            # Check for convergence (method for finite temperature)
            if self.smearing:
                converged = self.converged(density_list, old_density_list)
            else:  
                converged = self.converged(density_list, old_density_list, commuter)
            # If convergence is reached then break
            if converged:
                break
            # </HACK>

        # Print out loop info for debugging purposes
        if self.verbose:
            state = {True: 'converged', False: 'failed'}[converged]
            print(f'SCF {state} after {numIter} iterations.')

        # If the system did not converge, and this is for a batch operation
        # and this is not already a retry.
        if not converged and self.is_batch and \
            not kwargs.get('retry_for_convergence', False):
            # Then try again with a greater number of fock histories
            results = self.SCF(get_occ_rho_mask=get_occ_rho_mask,
                               retry_for_convergence=True,
                               max_n_fock=40)
            return results


        # If the SCF cycle fails to converge 
        if not converged and self.raise_SCF_exception:
            raise ConvergenceFailure('SCF cycle failed to coverage within the '
                                     f'permitted number of iterations ({self._maxSCFIter}).')
        elif not converged:
            print('Not converged')

        # Store the Fock matrices
        self.solFockList = fock_list

        # Store the density matrices
        self.density_list = density_list

        # Store the rho occupation mask
        self.occ_rho_mask = occ_rho_mask


        # This will always return the mermin free energy if finite temperature
        # is active
        if self.smearing:
            energy = self.mermin_elec_energy(density_list)
        else:
            energy = self._ElecEnergy(density_list)
        if get_occ_rho_mask:
            return energy, fock_list, density_list, occ_rho_mask
        else:
            return energy, fock_list, density_list
            

    # def ElecEnergyXYZDeriv1(self):
    #     argTup = self.atomList, self.parDict
    #     overlapXYZDeriv1 = Overlap(*argTup).Deriv1List()
    #     coreHXYZDeriv1 = CoreH(*argTup).Deriv1List()
    #     gammaXYZDeriv1 = Gamma(*argTup).Deriv1List()
    #     if self.solFockList is None:
    #         self.SCF()
    #     solList = [self.SolveFock(fock) for fock in self.solFockList]
    #     occSolList = [(ev[:ne], orb[:, :ne])
    #                   for (ev, orb), ne in zip(solList, self._numElecAB)]
    #     factor = 2.0 / len(self.solFockList)
    #     densList = [occOrb.dot(occOrb.T) for _, occOrb in occSolList]
    #     dens = sum(densList) * factor
    #     densDiff = densList[0] - densList[-1]
    #     enDens = sum([(orb * ev).dot(orb.T) for ev, orb in occSolList]) * factor
    #     deltaQShell, magQShell = self._DeltaQShellMagQShell(densList)
    #     dftb2eCou = self._Dftb2eMatrix(self._gamma, deltaQShell)
    #     dftb2eExc = self._Dftb2eMatrix(self._doublju, magQShell)
    #     numAtom = len(self.atomList)
    #     deriv1 = np.zeros((numAtom, 3))
    #     for ind in range(numAtom):
    #         basis = coreHXYZDeriv1[ind]['basisInd']
    #         shell = gammaXYZDeriv1[ind]['shellInd']
    #         densPart = dens[basis, :].ravel()
    #         enDensPart = enDens[basis, :].ravel()
    #         densDiffPart = densDiff[basis, :].ravel()
    #         dftb2eCouPart = dftb2eCou[basis, :].ravel()
    #         dftb2eExcPart = dftb2eExc[basis, :].ravel()
    #         for ci in range(3):
    #             coreHDeriv1 = coreHXYZDeriv1[ind]['xyz'][ci].ravel()
    #             partH = densPart.dot(coreHDeriv1)
    #             gammaDeriv1 = gammaXYZDeriv1[ind]['xyz'][ci]
    #             partGamma = gammaDeriv1.dot(deltaQShell).dot(deltaQShell[shell])
    #             overlapDeriv1 = overlapXYZDeriv1[ind]['xyz'][ci].ravel()
    #             couOver = dftb2eCouPart * overlapDeriv1
    #             excOver = dftb2eExcPart * overlapDeriv1
    #             part2e = densPart.dot(couOver) + densDiffPart.dot(excOver)
    #             partS = enDensPart.dot(overlapDeriv1)
    #             deriv1[ind, ci] = 2.0 * (partH + part2e - partS) + partGamma
    #     return deriv1

    
    def RepulsionXYZDeriv1(self):
        return Repulsion(self.atomList, self.parDict).XYZDeriv1()
    
    def SolveFock(self, fock):
        try:
            w,v = scipy.linalg.eigh(a=fock, b=self._overlap, turbo=True)
        except np.linalg.LinAlgError:
            print('DFTB: eigh failed, S eigenvalues are')
            S = self._overlap
            Sevals, Svecs = scipy.linalg.eigh(a=S)
            print(Sevals)
            raise
        except:
            print('DFTB: eigh unexpected error')
            raise
        return w,v
    
    def GetOverlap(self):
        return self._overlap
    
    def GetCoreH(self):
        return self._coreH
    
    def GetGamma(self):
        return self._gamma
    
    def GetDoublju(self):
        return self._doublju
    
    def GetNumElecAB(self):
        return self._numElecAB
    
    def GetShellBasis(self):
        return self._shellBasis

    def GetOnAtom(self):
        return self._onAtom

    def nBasis(self):
        return self._coreH.shape[0]
        
    def nAtom(self):
        return len(self.atomList)
    
    def GetShellQNeutral(self):
        return self._numVEShellNeu
    
    def ShellToFullBasis(self, matOrVec):
        # imap[iorb] = ishell   (iorb is full atomic basis, ishell is shell #)
        sbasis = self.GetShellBasis()
        imap = sum([[i1] * len(bas) for i1, bas in enumerate(sbasis)], [])
        shp = matOrVec.shape
        if len(shp) == 1:
            res = matOrVec[imap]
        if (len(shp) == 2) and (shp[0] == 1):
            res = matOrVec[0,imap]
        if (len(shp) == 2) and (shp[1] == 1):
            res = matOrVec[imap,0]
        else:
            res = matOrVec[np.ix_(imap,imap)]
            
        return res
        
    def FullBasisToShell(self, matOrVec):
        # assumes that matOrVec is the same for all elements in a shellxshell
        # block and this just copies that element into the result
        sbasis = self.GetShellBasis()
        imap = [x[0] for x in sbasis]        
        shp = matOrVec.shape
        if len(shp) == 1:
            res = matOrVec[imap]
        if (len(shp) == 2) and (shp[0] == 1):
            res = matOrVec[0,imap]
        if (len(shp) == 2) and (shp[1] == 1):
            res = matOrVec[0,imap]
        else:
            res = matOrVec[np.ix_(imap,imap)]
            
        return res

    def _GuessCore(self):
        # TODO: This assumes equal density matrices for alpha and beta, even if
        # numElecAB does not hold equal values, i.e. if num alpha != num beta
        # electrons. This does not seem correct for UHF
        res, _ = self.fock_to_dense([self._coreH] * len(set(self._numElecAB)))
        return res

    def fock_to_dense(self, fock_matrices, DEBUG=False):
        """
        This function takes in a fock matrix, or a list of fock matrices,
        calculated the corresponding density matrices and returns them.

        Parameters:
            - fock_matrices (array): List of Fock matrices to be used

        Uses:
            - n_elec (array(int)): Corresponding number of electrons.
                (spin up and spin down if unrestricted)
            - S (array): Overlap matrix
            - sigma (float): Smearing width (kT) in Ha (for finite temperature)
            - scheme (string): Smearing method: 'gaussian' or 'fermi' when
                doing finite temperature, otherwise use None.

        Returns:
            - array (float): List of density matrices
            - array (float): List of occupancy masks (only of use to the 
                tensorflow code).

        TODO:
            - Remove sigma_in from the arguments.
            - Double check that we are indexing 'C' correctly.

        Notes:
            - WARNING feeding sigma to the function in this manner is
                dangerous and is likely to result in unintended side-
                effects. 

        NOTES MUST BE UPDATED:
            - sigma -> smearing
            - fock list to single fock
        """

        # Check that only a single Fock matrix is being provided
        if len(fock_matrices) != 1:
            raise NotImplementedError('UHF is not currently implemented.')

        # Spawn an empty list to hold the density matrices
        density_matrices = []

        # Spawn an empty list to hold the occupancy masks (used by TF code)
        occupancy_masks = [] # This should only be full if using smearing

        # If temperature smearing is active (determined by sigma != None)
        if self.smearing:

            # Loop over pairs of Fock matrices and their electron counts.
            # While we only expect a single Fock matrix, this code is
            # retained for when UHF is implemented.
            for F, ne in zip(fock_matrices, self._numElecAB):

                # Diagonalize the Fock matrix to get the eigen values & vectors
                eigenvalues, v = scipy.linalg.eigh(F, self._overlap, turbo=True)

                # Determine the fermi level
                try:
                    fermi_level = determine_fermi_level(eigenvalues, 2 * ne,
                                                        self.smearing,
                                                        self.smearing_scheme,
                                                        powell=False)
                # If a convergence failure is encountered
                except ConvergenceFailure as error:
                    # If the user has specifiably stated that it is ok to try and
                    # bypass this error.
                    if self.bypass_fermi_convergence_error:
                        # Then try using last know good fermi level
                        fermi_level = self.fermi_level
                    else:
                        print('Failed to converge fermi level')
                        # Otherwise raise the original error
                        raise error


                # Calculate occupancies according to the smearing function
                occupancies = get_occupancies(eigenvalues, fermi_level,
                                              self.smearing_scheme, self.smearing)

                # Scale the occupancies & multiply them with the eigen-vectors
                occupancies_scaled = np.sqrt(0.5 * occupancies) * v

                # Dotting occupancies by their transpose gives density matrices
                density_matrix = np.dot(occupancies_scaled, occupancies_scaled.T)

                # Construct the occupancy mask to be used in TF calculations
                temp = np.expand_dims(np.sqrt(0.25 * occupancies), 0)
                occupancy_mask = np.repeat(temp, temp.shape[1], 0)

                # Some debugging printouts
                if self.verbose:
                    print(f'Fermi Level: {fermi_level:8.4e} Ha')
                    # Fix text print out function below
                    #print(f'Filling: ', [f'{i:5.2}' for i in occupancies_scaled])

                # Add the calculated density matrix to the list 
                density_matrices.append(density_matrix)

                # Add the occupancy mask to the appropriate list
                #occupancy_masks.append(occupancy_mask)
                occs = np.sqrt(0.5 * occupancies)

                #occs.append(np.sqrt(0.5 * occupancies))
                t1 = np.expand_dims(occs,0)
                occ_mask = np.repeat(t1,t1.shape[1],0)
                occupancy_masks.append(occ_mask)
                # Update the object's fermi level
                self.fermi_level = fermi_level

                # Hack for entropy term (AJM to fix)
                entropy = get_entropy_term(eigenvalues, self.fermi_level,
                                           self.smearing_scheme, self.smearing)

        # If we are not doing finite temperature
        else:
            # WARNING THIS WILL BREAK WHEN USED WITH SYSTEMS WITH AN ODD NUMBER OF ELECTRONS
            # Construct the orbital eigenvector list
            orbs = [self.SolveFock(fock)[1] for fock in fock_matrices]
            # Limit the list to only occupied orbitals
            filled_orbs = [o[:, :ne] for o, ne in zip(orbs, self._numElecAB)]

            # Dot the occupancies by their transpose to yield the
            # density matrices, and add them to the list.
            density_matrices.extend([np.dot(i, i.T) for i in filled_orbs])

            # Compute the occupancy mask
            occ_mask = np.zeros(orbs[0].shape)
            ne = self._numElecAB[0]
            occ_mask[:,:ne] = 1.0
            occupancy_masks = [occ_mask]

            #occupancy_masks.extend([None for i in self._numElecAB])

        return density_matrices, occupancy_masks[0]

    
    def _Comm(self, fockList, densList):
        fdsList = [fock.dot(dens).dot(self._overlap)
                   for fock, dens in zip(fockList, densList)]
        indices = np.triu_indices_from(self._overlap, 1)
        return np.concatenate([(fds - fds.T)[indices] for fds in fdsList])
    
    def _DensToFock(self, densList):
        deltaQShell, magQShell = self._DeltaQShellMagQShell(densList)
        couMat = self._overlap * self._Dftb2eMatrix(self._gamma, deltaQShell)
        excMat = self._overlap * self._Dftb2eMatrix(self._doublju, magQShell)
        fockNoSpin = self._coreH + couMat
        return [fockNoSpin + exc for exc in [excMat, -excMat][:len(densList)]]
    
    def _ElecEnergy(self, densList):
        deltaQShell, magQShell = self._DeltaQShellMagQShell(densList)
        factor = 2.0 / len(densList)
        energy = self._coreH.ravel().dot(sum(densList).ravel()) * factor
        energyCou = self._Dftb2eEnergy(self._gamma, deltaQShell)
        energyExc = self._Dftb2eEnergy(self._doublju, magQShell)
        return energy + energyCou + energyExc

    def mermin_elec_energy(self, density_list = None):
        """
        This function calculates the Mermin free electron energy. This
        should be used instead of the standard electron energy when using
        finite temperature. 

        Parameters:
            - density_list (array): List of the density matrices
            - sigma_in (float): An override or the local sigma value.
                (to be removed as it is too 'hacky')

        Returns:
            - float: The free energy of the target system.

        TODO:
            - Remove the need for the sigma_in and density_list arguments
                as it is far too messy. (parts of this and other modules
                will need to be rewritten)
            - Current implementation of this function is a temporary hack.

        NEED TO UPDATE DOC-STRING HERE
        """

        if self.smearing:
            # if density list not given to, use the class instances'
            if not density_list:
                density_list = self.density_list

            # Calculate the base electron energy
            elec_energy = self._ElecEnergy(density_list)

            # Get the eigen-energies
            w, _ = self.SolveFock(self.solFockList[0])

            # Fetch the entropy term
            S = get_entropy_term(w, self.fermi_level, self.smearing_scheme, self.smearing)

            mermin_energy = elec_energy - self.smearing * S

        else:
            mermin_energy = 0.0

        return mermin_energy
    
    def _DeltaQShellMagQShell(self, densList):
        # Mulliken population
        mlkPopList = [dens * self._overlap for dens in densList]
        qShellSpin = np.array([[np.sum(pop[bas, :]) for pop in mlkPopList]
                               for bas in self._shellBasis])
        # Total and delta charge
        qShell = np.sum(qShellSpin, axis=1) * 2.0 / len(densList)
        deltaQShell = qShell - self._numVEShellNeu
        # Magnetization charge
        magQShell = (qShellSpin[:, 0] - qShellSpin[:, -1]).ravel()
        return deltaQShell, magQShell

    def _Dftb2eMatrix(self, shellMat, shellVec):
        zipList = list(zip(0.5 * shellMat.dot(shellVec), self._shellBasis))
        epBasis = np.array([sum([[ep] * len(bas) for ep, bas in zipList], [])])
        return epBasis + epBasis.T
    
    def _Dftb2eEnergy(self, shellMat, shellVec):
        return 0.5 * shellMat.ravel().dot(np.outer(shellVec, shellVec).ravel())

    def converged(self, dm1, dm2, com=None):
        """
        Class method used to check for convergence during the self
        consistent field cycle (SCF).

        Parameters:
            - dm1 (array): The updated density matrix, or in the case
                           of an unrestricted method a list of matrices.
            - dm2 (array): The previous density matrix, or in the case
                           of an unrestricted method a list of matrices.
            - com (array): Presumably the commuter matrix? This can be
                           omitted if desired.

        Returns:
            - boolean: Stating whether or not the system has converged
                       according to the class settings.

        Notes:
            - Actual order in which the new and old density matrices
              are supplied is irrelevant. i.e dm1 can be the old matrix.
        """

        # If the matrices have not been supplied as a numpy array, convert them
        if type(dm1) != np.ndarray:
            dm1, dm2 = np.asarray(dm1), np.asarray(dm2)

        # Unravel dm1 and dm2 and take the difference (agnostic array shape)
        dm_diff = dm1.ravel() - dm2.ravel()

        # Calculate the root mean squared difference
        dm_rms = np.sqrt(np.mean(dm_diff ** 2))

        # Identify the maximum deviation
        dm_max = np.max(np.abs(dm_diff))

        # Booleans indicating status of convergence
        convergence_bools = [dm_rms < self._thrRmsDens,
                             dm_max < self._thrMaxDens]

        # If commuter is provided then calculate the commuter's convergence
        if com is not None:
            # Ensure com is unraveled, and get rims and max values
            com = np.asarray(com).ravel()
            com_rms = np.sqrt(np.mean(com ** 2))
            com_max = np.max(np.abs(com))

            # Add commuter convergence booleans
            convergence_bools.extend([com_rms < self._thrRmsComm,
                                      com_max < self._thrMaxComm])

        # Check if all convergence thresholds have been met
        has_converged = all(convergence_bools)


        # If verbosity is required; print out various details
        if self.verbose:
            i = '10.3e'
            print('SCF Convergence Information (Threshold):\n'
                  f'\tDensity  RMS: {dm_rms:{i}} ({self._thrRmsDens:{i}})\n'
                  f'\tDensity  Max: {dm_max:{i}} ({self._thrMaxDens:{i}})')

            if com:  # Commuter convergence status
                print(f'\tCommuter RMS: {com_rms:{i}} ({self._thrRmsComm:{i}})\n'
                      f'\tCommuter Max: {com_max:{i}} ({self._thrMaxComm:{i}})')

        # Return the boolean
        return has_converged

    def _calc_qN(self):
        qnShell = np.array(self.GetShellQNeutral())
        sbasis  = self.GetShellBasis()
        return sum([[qneut / len(bas)] * len(bas) for qneut, bas in zip(qnShell, sbasis)], [])


    def get_dQ_from_H(self, newH, newG = None):
        Hsave = self._coreH
        self._coreH = newH
        if newG is not None:
            Gsave = self._gamma
            self._gamma = self.FullBasisToShell(newG)
        E,Flist,rholist, occ_rho_mask = self.SCF(get_occ_rho_mask=True)                            
        S  = self.GetOverlap()
        rho = 2.0 * rholist[0]
        qBasis = (rho)*S
        GOP = np.sum(qBasis,axis=1)
        self._coreH = Hsave

        # Current hack to deal with smearing = None
        if self.smearing:
            entropy_term = self.entropy * self.smearing
        else:
            entropy_term = 0.0
        if newG is not None:
            self._gamma = Gsave
        return self._qN - GOP, occ_rho_mask, entropy_term
        
    def dQ(self, Flist, rholist):
        S  = self.GetOverlap()
        rho = 2.0 * rholist[0]
        qBasis = (rho)*S
        GOP = np.sum(qBasis,axis=1)
        return self._qN - GOP


def _ShellBasis(atomList):
    shellBasis = []
    onAtom = []
    for iatom,atom in enumerate(atomList):
        for nShell in range(atom.numShell):
            curInd = sum([len(bas) for bas in shellBasis])
            shellBasis += [list(range(curInd, curInd + 2 * nShell + 1))]
            onAtom += [iatom]* ( 2 * nShell + 1 )
    return (shellBasis, onAtom)

def _NumVEShellNeu(atomList, parDict):
    numVEShellNeu = []
    for atom in atomList:
        par = parDict[atom.elem + '-' + atom.elem]
        sShell = [par.GetAtomProp('fs')]
        pShell = [par.GetAtomProp('fp')] if 'p' in atom.orbType else []
        dShell = [par.GetAtomProp('fd')] if 'd' in atom.orbType else []
        numVEShellNeu += sShell + pShell + dShell
    return numVEShellNeu

def _NumElecAB(numElecTotal, mult):
    '''
      input:
        numElecTotal: total number of electrons in molecule
        mult:         1 for singlet, 3 for triplet etc
      returns:
         array of length 2, with number of alpha and beta electrons
      for example
        numElecTotal    mult      returns [numElecA numElecB]
        6               1          [3,3]
        7               2          [4,3]
        6               3          [4,2]
        7               4          [5,2]
    '''
    if numElecTotal % 2 == 1 and BYPASS_UHF_SAFETY_CHECK:
        return [numElecTotal/2, numElecTotal/2]
    else:
        numElecA = (numElecTotal + mult - 1) / 2.0
        if numElecA % 1 != 0.0:
            raise Exception('numElecTotal %d and mult %d ??' % (numElecTotal, mult))
        return [int(numElecA), int(numElecTotal - numElecA)]

