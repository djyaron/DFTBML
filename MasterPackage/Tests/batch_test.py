# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:36:15 2021

@author: fhu14

TODO: Fix this test function!
"""
#%% Imports, definitions
from collections import OrderedDict
import numpy as np
import scipy

from Geometry import random_triatomics, to_cart
from DFTBpy import determine_fermi_level, fermi, DFTB

import math, pickle
from DFTBLayer import DFTBList, create_batch, create_dataset, ParDict,\
    np_segment_sum, maxabs

#%% Code behind

def test_dataset(data, smearing = None, print_all = True):
    sigma = 0.003 #Define top level constants
    # **** Get DFTB values for the output of the ML input layer ***
    # Transforms mod_raw into net_vals
    # data['models'] is a list of Model's, e.g. Model(oper='G',Zs=(6,),orb='ps')
    # batch['mod_raw'] is a list of RawData specifying the meaning of each
    # value to be output by the ML layer, e.g. batch['mod_raw'][model][0] may
    # be RawData(index=30, glabel=0, Zs=(6,), atoms=(1,), oper='G', orb='ps', 
    #            dftb=0.3646664973925, rdist=0.0)
    #
    # Will save calculated values in dictionary with same keys as in data
    calc = OrderedDict()
    # We first concatenate the RawData from all models
    net_raw = []
    for m in data['models']:
        net_raw.extend(data['mod_raw'][m])
    # Test that the geometry and atom labels are consistent
    geoms = data['geoms']
    dist1 = np.array([x.rdist for x in net_raw])
    dist2 = np.array([geoms[x.glabel].dist(x.atoms[0], x.atoms[-1]) for x in net_raw])
    if print_all:
        print('check labels in mod_raw: err = ', np.max(np.abs(dist1-dist2)))
    # We then pull out the dftb values from this list. If the ML layers put
    # out these values, then the model would correspond to DFTB
    net_vals = np.array([r.dftb for r in net_raw])
    
    #    *** Slater-Koster (SK) Rotations  ***
    # Transforms net_vals into rot_out
    # rot_out starts with 0 and 1, and then holds all rotated values in 
    # an order that is consistent with the next stage, i.e. gather_for_oper
    # The SK rotations are meant to batched by the shape of the SK rotation
    # matrix, e.g. shape = (1,) for ss, (3,1) for sp, (9,2) for pp
    # data['gather_for_rotation'][shape] gathers from net_vals
    # data['rot_tensors'][shape] holds the rotation matrices
    # by convention, first and second values are 0 and 1
    rot_out = [0.0, 1.0]
    # Loop over all shapes
    for s,gather in data['gather_for_rot'].items():
        # if there are no rot_tensors for this shape, no rotation is needed
        if data['rot_tensors'][s] is None:
            # can simply add the gathered values onto rot_out
            rot_out.extend(net_vals[gather])
        else:
            # The rotation operation is a batch multiply
            #    rotated(:,s[0]) = tensor(:,s[0],s[1]) net_vals(:,s[1])
            # gather and reshape net_vals into  shape (:,s[1])
            vals = net_vals[gather].reshape([-1,s[1]])
            # retreive the tensors, which are stored with shape (:,s[0],s[1])
            tensor = data['rot_tensors'][s]
            # In tensorflow, this will be a single batch multiply, but
            # here we do it as a loop over number of bonds
            nb = vals.shape[0]
            rot_out_temp = np.zeros([nb,s[0]])
            for ib in range(nb): # looping over tensorflow batches
                rot_out_temp[ib,:] = np.dot(tensor[ib,:,:], vals[ib,:])
            # convention for rotated values is flattened form
            rot_out.extend(rot_out_temp.flatten() )
    # gather operations only work on np arrays
    rot_out = np.array(rot_out)
    
    #  *** Assemble values from SK rotations into operators ***
    # Transforms rot_out into oper_mats
    # data['onames'] is string of operator names, e.g. ['G', 'H', 'S']
    for oname in data['onames']:
        calc[oname] = {}
        if oname != 'R':
            for bsize in data['basis_sizes']:
                # gather is a 1-D np array
                gather = data['gather_for_oper'][oname][bsize]
                # reshape the gathered 1-D array into batch form (:,bsize,bsize)
                # note that the order of geometries in (:,bsize,bsize)
                # is that in the list data['glabels'][bsize]
                calc[oname][bsize] = \
                 rot_out[gather].reshape([len(data['glabels'][bsize]),bsize,bsize])

    if 'S' not in data['onames']:
        calc['S'] = data['S']
    if 'G' not in data['onames']:
        calc['G'] = data['G']
    # *** Construct the Fock operators ***
    # Transforms oper_mats, qneutral, rhos into Fock
    # see check_fock_routine for more detailed discussion. The 
    # required operators are
    # qN = data['qneutral'] # qN[bsize](ngeom,bsize)
    # rhos = data['rhos']     # rhos[bsize](ngeom,bsize,bsize))
    # qBasis = rho*S
    # qHalf = np.sum(qBasis,axis=1)
    # dQ = qHalf - qN
    # ep = np.array([ Gfull.dot(dQ) ])
    # couMat = S * (ep + ep.T)
    # fock = H1 + couMat
    #
    # F1[bsize] = np.array with shape (:, bsize, bsize) ordered as in oper_mats
    calc['F'] = {}
    calc['dQ'] = {}
    calc['Erep'] = {}
    for bsize in data['basis_sizes']:
        qbasis = (data['rho'][bsize]) * calc['S'][bsize]
        GOP  = np.sum(qbasis,2,keepdims=True)
        calc['dQ'][bsize] = data['qneutral'][bsize] - GOP
        ngeom = calc['dQ'][bsize].shape[0]
        couMat = np.zeros([ngeom,bsize,bsize])
        for igeom in range(ngeom):
            ep = np.dot( calc['G'][bsize][igeom,:,:],
                         calc['dQ'][bsize][igeom,:] ).reshape([1,bsize])
            couMat[igeom,:,:] = -0.5*calc['S'][bsize][igeom,:,:] * \
                  (ep + ep.T)
        calc['F'][bsize] = calc['H'][bsize] + couMat 
        vals = net_vals[data['gather_for_rep'][bsize]]
        calc['Erep'][bsize] = np_segment_sum(vals,
                            data['segsum_for_rep'][bsize])

    # *** Solve generalized eigenvalue problem for Fock operator ***
    # Transforms Fock and S into orbitals
    # see check_fock_routine() and the following for more details
    # http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html
    #
    # In TF, this will be batch operations, but here we do a loop
    calc['Eelec']= {}
    calc['eorb'] = {}
    calc['rho'] = {}
    for bsize in data['basis_sizes']:
        ngeom = len(data['glabels'][bsize])
        calc['Eelec'][bsize] = np.zeros([ngeom])
        calc['eorb'][bsize] = np.zeros([ngeom,bsize])
        calc['rho'][bsize]  = np.zeros([ngeom,bsize,bsize])
        for igeom in range(ngeom):
            S1 = calc['S'][bsize][igeom,:,:]
            fock = calc['F'][bsize][igeom,:,:]
            if 'phiS' not in list(data.keys()):
                Svals, Svecs = scipy.linalg.eigh( S1 )
                phiS = np.dot(Svecs,np.diag( np.sqrt(1.0/Svals) ) )
            else:
                phiS = data['phiS'][bsize][igeom,:,:]
            fockp = np.dot(phiS.T, np.dot(fock,phiS))
            Eorb, temp2  = scipy.linalg.eigh(a = fockp)

            calc['eorb'][bsize][igeom,:] = Eorb
            orb = np.dot(phiS, temp2)
            if smearing is None:
                orb_filled = np.multiply(data['occ_rho_mask'][bsize][igeom,:,:], orb)
            else:
                occ_mask_orig = data['occ_rho_mask'][bsize][igeom,:,:]
                num_filled = sum(occ_mask_orig[0,:])
                
                fermi_level = determine_fermi_level(Eorb, 2*num_filled,
                    sigma = smearing['width'],scheme=smearing['scheme'],threshold=1.E-10)
                fermi_val = 2./(1.+np.exp((Eorb-fermi_level)/sigma))
                pop_sqrt = np.sqrt(0.5*fermi_val)
                # we want to evaluate:
                #   rho(i,j) = sum_a pop(a) orb(i,a) orb(j,a)
                # so we do instead
                #   rho(i,j) = sum_a pop_sqrt(a) orb(i,a) pop_sqrt orb(j,a)
                # so we define orb_filled(i,a) = pop_sqrt(a) * orb(i,a)
                t1 = np.expand_dims(pop_sqrt,0)
                occ_mask = np.repeat(t1,bsize,0)
                orb_filled = np.multiply(occ_mask, orb)
            rho = 2.0 * np.dot(orb_filled, orb_filled.T)
            calc['rho'][bsize][igeom,:,:] = rho
            ener1 = np.sum(np.multiply(rho.ravel(),calc['H'][bsize][igeom,:,:].ravel()))
            dQ = calc['dQ'][bsize][igeom,:]
            Gamma = calc['G'][bsize][igeom,:,:]
            ener2 = 0.5 * np.dot(dQ.T, np.dot(Gamma, dQ))
            calc['Eelec'][bsize][igeom] = ener1 + ener2[0,0]
            
            
    print('Results from test_dataset()')
    max_error = 0.0
    for data_type, calc_by_bsize in calc.items():
        for bsize, calc_test in calc_by_bsize.items():
            err = np.max(np.abs(
                    (calc_test - data[data_type][bsize]).flatten() ))
            if print_all:
                print(data_type,' max error ', err)
            if err > max_error:
                max_error = err
    if print_all:
        print(' Overall maximum error ', max_error)
    return max_error

def check_fock_routine(dftb):
    # Saved primarily to document the way the Hartree Fock calcs are done
    # Took the code from dftb.py and morphed it to make it easier to 
    # implement in tensorflow 
    E,Flist,rholist = dftb.SCF()
    F = Flist[0]
    rho = 2.0*rholist[0]
    
    H1 = dftb.GetCoreH()
    S  = dftb.GetOverlap()
    G  = dftb.GetGamma()
    sbasis = dftb.GetShellBasis()
    qNeutral = np.array(dftb.GetShellQNeutral())
    
    # re-writing the following: 
    #qShell = np.array( [2.0 * np.sum(qBasis[bas, :]) for bas in sbasis])
    #deltaQShell = qShell - qNeutral
    #zipList = zip(0.5 * G.dot(deltaQShell), sbasis)
    #epBasis = np.array([sum([[ep] * len(bas) for ep, bas in zipList], [])])
    #couMat = S* (epBasis + epBasis.T)
    
    
    # Old version
    #zipList = zip(0.5*qNeutral, sbasis)
    #qN = sum([[qneut/len(bas)] * len(bas) for qneut, bas in zipList], [])
    #print 'qN',qN
    #qHalf = np.sum(qBasis,axis=1)
    #print 'qBasis',qBasis
    #dQ = qHalf - qN
    #print 'dQ',dQ
    #Gfull = dftb.ShellToFullBasis(G)   
    #ep = np.array([ Gfull.dot(dQ) ])
    #print 'ep shape',ep.shape
    #print 'ep',ep
    #print 'S', S
    #couMat = S * (ep + ep.T)
    #print 'couMat',couMat
    #print 'H1', H1
    #fock = H1 + couMat
    #print 'fock',fock
    #print 'max diff in F ', maxabs(fock-F)
   
    # New version, with charges being standard Mulliken charges
    # evenly distribute qNeutral across each basis function in the shell
    zipList = list(zip(qNeutral, sbasis))
    qN = sum([[qneut/len(bas)] * len(bas) for qneut, bas in zipList], [])
    # GOPi, as in https://en.wikipedia.org/wiki/Mulliken_population_analysis
    qBasis = rho*S
    GOP = np.sum(qBasis,axis=1)
    # GOP is a population of electrons, so if it is larger than qN, the charge
    # should be negative (this is a sign error in original Elstner paper?)
    dQ = qN - GOP
    Gfull = dftb.ShellToFullBasis(G)
    ep = np.array([ Gfull.dot(dQ) ])
    # interaction of electron with negative charges is positive energy, so
    # also need this change in charge
    couMat = -0.5*S * (ep + ep.T)
    fock = H1 + couMat
    print('F error ',np.max(np.abs(F-fock)))
    
    # Solving the generlized eigenvalue problem
    #http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html
    
    Eorb1, orb1 = scipy.linalg.eigh(a=fock, b=S)
    Svals, Svecs = np.linalg.eigh(S)
    print('test of S diagonalization ', \
       maxabs(np.dot(Svecs.T, np.dot(S,Svecs)) - np.diag(Svals) ))
    phiS = np.dot(Svecs,np.diag( np.sqrt(1.0/Svals) ) )
    print('test of phiS.T S phiS = unit matrix ', \
       maxabs( np.dot(phiS.T, np.dot(S,phiS)) - np.eye(S.shape[0]) ))
    fockp = np.dot(phiS.T, np.dot(fock,phiS))
    Eorb2, temp2 = scipy.linalg.eigh(a = fockp)
    orb2 = np.dot(phiS, temp2)
    print('Test of orbital energies ', maxabs(Eorb2 - Eorb1)) 
    # abs is used to ignore overall phase of orbitals, but that makes this
    # not a perfect test
    print('Partial test of orbitals ', maxabs(np.abs(orb2) - np.abs(orb1))) 
    
    if dftb.smearing is None:
        # Generate density matrix from orbs
        #orbList = [self.SolveFock(fock)[1] for fock in fockList]
        #occOrbList = [orb[:, :ne] for orb, ne in zip(orbList, self.__numElecAB)]
        #return [occOrb.dot(occOrb.T) for occOrb in occOrbList]
        ne = dftb.GetNumElecAB()[1]
        occ_mask = np.zeros(rho.shape)
        occ_mask[:,:ne] = 1.0
        orb_filled = np.multiply(occ_mask, orb2)
        rhoTest = 2.0*np.dot(orb_filled, orb_filled.T)
    else:
        nfilled = dftb.GetNumElecAB()[0]
        w,v = dftb.SolveFock(fock)
        fermi_level = determine_fermi_level(w, 2*nfilled, scheme=dftb.smearing_scheme,
                        sigma = dftb.smearing , threshold=1.E-10)
        occs = []
        for ei in w:
            occs.append(np.sqrt(.5*fermi(ei, fermi_level, dftb.smearing)))
        t1 = np.expand_dims(occs,0)
        occ_mask = np.repeat(t1,t1.shape[1],0)
        vfilled = np.multiply(occ_mask, v)
        rhoTest = 2.0 * vfilled.dot(vfilled.T)
        
    print('Test of rho ',maxabs(rho-rhoTest))
    
    E1ener = np.sum(np.multiply(rho,H1), axis=(0,1))
    E2ener = 0.5* np.dot(dQ.T, np.dot(Gfull,dQ))
    Etest = E1ener + E2ener
    print('ener from dftb', E,' Etest ', Etest,' diff ',abs(E-Etest))      
    print(' E1 ener ', E1ener, ' E2ener ', E2ener)
    
def run_batch_tests():
    print('batch.py generating data for random triatomics and testing')
    graph_data_fields = ['models','onames','basis_sizes','glabels','qneutral', 
                     'qneutral','occ_rho_mask','occ_eorb_mask','mod_raw',
                     'dQ','gather_for_rot', 'rot_tensors', 'gather_for_oper',
                     'gather_for_rep', 'segsum_for_rep',
                     'dftb_elements', 'dftb_r', 'Eelec', 'Erep','Etot',
                     'unique_gtypes', 'gtype_indices']
    mol_type = 'custom' #'HAu2' # 'organic'
    sigma = 0.003 #10.0
    if mol_type == 'organic':
        ngeom = 10
        charges = [0] * 3* ngeom
        gtypes = []
        geoms1 = random_triatomics(ngeom, [1,6,7],[0.7,1.1],[0.7,1.1],
                           [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0])
        gtypes.extend(['HCN'] * ngeom)
        geoms1.extend(random_triatomics(ngeom, [7,6,1],[0.7,1.1],[0.7,1.1],
                           [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]))
        gtypes.extend(['NCH'] * ngeom)
        geoms1.extend(random_triatomics(ngeom, [1,8,1],[0.7,1.1],[0.7,1.1],
                           [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]))
        gtypes.extend(['H2O'] * ngeom)
    elif mol_type == 'HAu2':
        gtypes = []
        ngeom = 1
        Zs = [1,79,79]
        charges = [1] * ngeom # To keep it a singlet
        length_range1 = [1.0,1.3]
        length_range2 = [2.0,2.5]
        angle_range = [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]
        geoms1 = random_triatomics(ngeom, Zs, length_range1, length_range2,
                           angle_range)
        gtypes.extend(['HAu2'] * ngeom)
    elif mol_type == 'HCN':
        gtypes = []
        ngeom = 1
        Zs = [1,6,7]
        charge = [0] * ngeom # To keep it a singlet
        length_range1 = [0.7,1.1]
        length_range2 = [1.2,1.6]
        angle_range = [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]
        geoms1 = random_triatomics(ngeom, Zs, length_range1, length_range2,
                           angle_range)
        gtypes.extend(['HCN'] * ngeom)
    elif mol_type == 'custom':
        target = 'data/small_test.pkl'
        db = np.array(pickle.load(open(target, 'rb'))).T

        ngeom = 8
        geoms1 = [i['geom'] for i in db[0]][:ngeom]
        Zs = geoms1[0].z
        charges = [0] * ngeom
        gtypes = ['x'] * ngeom
    else:
        print('unknown molecule type + ' + mol_type)
        exit()

    print('mol type: ' + mol_type + ' sigma ' + str(sigma))
    if sigma is not None:
        smearing = {'scheme': 'fermi',
                'width' :  sigma}
        dftb = DFTB(ParDict(),to_cart(geoms1[0]), charge = charges[0], smearing=smearing)
    
        print('check_fock_routine()')
        check_fock_routine(dftb)
    
        data_fields = graph_data_fields + DFTBList.data_fields + ['geoms']
        batch = create_batch(geoms1, gtypes = gtypes, charges = charges, smearings=[smearing]*ngeom)
        dftblist = DFTBList(batch)
        dftb_data = create_dataset(batch,dftblist,data_fields)
        print('test_dataset()')
        test_dataset(dftb_data, smearing)
    else:
        dftb = DFTB(ParDict(),to_cart(geoms1[0]), charge = charges[0])
    
        print('check_fock_routine()')
        check_fock_routine(dftb)
    
        data_fields = graph_data_fields + DFTBList.data_fields + ['geoms']
        batch = create_batch(geoms1, gtypes = gtypes, charges = charges)
        dftblist = DFTBList(batch)
        dftb_data = create_dataset(batch,dftblist,data_fields)
        print('test_dataset()')
        test_dataset(dftb_data)

if __name__ == "__main__":
    run_batch_tests()