# -*- coding: utf-8 -*-

import math
import numbers
import numpy as np
import collections
from operator import itemgetter
try:
    import matplotlib.pyplot as plt
except:
    pass


from Rotations import rotation_matrix_atom, rotation_matrix_bond, \
    get_axis_rotation_matrix
from Elements import ELEMENTS

def to_cart(geom):
    return np.hstack([geom.z.reshape([-1,1]), geom.rcart.T])

def write_gjf(filename, checkpointname, keywords, geom, title = 'Title'):
    Zs = geom.z
    cart = geom.rcart.T
    f1 = open(filename,'w')
    f1.write('%chk='+checkpointname + '\n')
    f1.write('#' + keywords+ '\n')
    f1.write('\n')
    f1.write(title + '\n')
    f1.write('\n')
    f1.write('0 1'+ '\n')
    for i,Z in enumerate(Zs):
        line = ELEMENTS[Z].symbol 
        #line += ' ' + str(frozen[i])
        line += ' %.8f' % cart[i,0] + ' %.8f' % cart[i,1] + ' %.8f' % cart[i,2]
        f1.write(line+ '\n')
    f1.write('\n')
    f1.close()

class Geometry(object):
    """
    Manages properties associated with atomic structure, such as bonding
    patterns, inertial tensor etc. 
    """

    def __init__(self, z, rcart, addon_for_isbonded=0.35):
        """
        Required input:
        z -- [natom] atomic numbers of the atoms
        rcart -- float [3,natom]
        addon_for_isbonded -- two atoms are considered bonded if the bond 
           is less than addon_for_isbonded + the sum or their covalent radii

        """
        self.z                      = np.asarray(z,np.int_).copy()
        self.rcart                  = np.asarray(rcart).copy()
        self.natom                  = len(self.z)
        self.addon_for_isbonded     = addon_for_isbonded
        self.connections            = self.__initialize_connections()

    def sort_atoms(self):
        i = np.argsort(self.z)
        self.z = self.z[i]
        self.rcart = self.rcart[:,i]
        self.connections = self.__initialize_connections()
        return i

    def __calc_isbonded(self):
        ## from: https://en.wikipedia.org/wiki/Covalent_radius
        covalent_radii = [ELEMENTS[i].covrad for i in self.z]
        natom = self.natom
        res = np.zeros([natom, natom])
        for iatom in range(natom):
            for jatom in range(iatom+1, natom):
                rdist = self.dist(iatom,jatom)
                bond_cutoff = ( covalent_radii[iatom ]
                              + covalent_radii[jatom ]) + self.addon_for_isbonded
                if (rdist < bond_cutoff):
                    res[iatom,jatom] = 1
                    res[jatom,iatom] = 1
        return res        
    def __initialize_connections(self):
        natom = self.natom
        res = np.zeros([natom,natom])
        idiag = np.diag_indices_from(res)
        
        bonded = self.__calc_isbonded()
        conn = np.identity(natom)
        for ilevel in range(1,10):
            #extend connections by 1 level
            upConn = np.dot(conn, bonded)
            # set diagonal to zero
            upConn[idiag] = 0
            # set all elements that were already specified to zero
            upConn[res>0] = 0
            res[upConn > 0] = ilevel
            conn = upConn
        return res        
        
    def rbond(self,atom0, atom1):
        '''
        cartesian vector pointing from atom0 to atom1 
        '''
        return (self.rcart[:,atom1] - self.rcart[:,atom0]).reshape(-1)
        
    def dist(self, atom0, atom1):
        '''
        distance between atom0 and atom1
        '''
        return np.linalg.norm(self.rbond(atom0, atom1))

    def angle(self,atom0,atom1,atom2):
        '''
           bond angle, with atom1 as central atom
        '''
        r0 = np.array(self.rbond(atom1,atom0))
        r1 = np.array(self.rbond(atom1,atom2))
        e0 = r0/np.linalg.norm(r0)
        e1 = r1/np.linalg.norm(r1)
        xcos = np.sum(e0*e1)
        xsin = np.linalg.norm( np.cross(e0,e1) )
        return math.atan2(xsin,xcos)
    def empirical_formula(self):
        '''
           returns dict mapping Z to # atoms of that element
           {Z: number of atoms}
        '''
        return collections.Counter(self.z)
    def empirical_formula_str(self, padding = 4):
        ef = self.empirical_formula()
        syms = [ELEMENTS[z].symbol for z in list(ef.keys())]
        nums = [ef[z] for z in list(ef.keys())]
        # from https://stackoverflow.com/questions
        #25539593/sort-a-list-then-give-the-indexes-of-
        # the-elements-in-their-original-order
        indices = list(zip(*sorted(enumerate(syms), key=itemgetter(1))))[0]
        if padding == 0:
            res = ''.join([str(syms[i])+str(nums[i]) for i in indices])
        else:
            res = ''.join([(str(syms[i])+str(nums[i])).ljust(padding) for i in indices])
        return res
        
    def nheavy_atoms(self):
         return np.count_nonzero(self.z-1)
    def connection_order(self, atom0, atom1):
        '''
        Get minimum number of bonds connecting atom0 to atom1. For example,
        if atom0==atom1, return 0
        if atoms are bonded, return 1
        if atoms are bonded to a common atom, return 2
        The data is precomputed to make evaluation lowcost
        
        In molecular mechanics, nonbonded atoms have a connectivity >=3
        
        Matrix is directly accessible as Geometry.connections        
        
        Require input:
        atom1 -- number of first atom 
        atom2 -- number of second atom
                 (with convention that first atom is 0)
        '''
        return self.connections[atom0,atom1]
        
    def get_symbols(self):
        '''
        Returns array of strings with the element names
        '''
        return [ELEMENTS[i].symbol for i in self.z]
        
    def get_atoms(self, z):
        '''
        Returns list of atom numbers for element type Z
        '''
        return [i for i, x in enumerate(self.z) if x == z]
    
    def get_bonds(self, z0, z1, conn_order_target=None):
        '''
        Returns list of dubles (atom_number0, atom_number1) where
        atom0 is of type z0, atom1 is of type z1, and the connection order
        agrees with the provided target values. 
        
        If z0 == z1, returned bonds have atom_number0 < atom_number1
        
        Input:
        z0, z1 : element numbers of first and second atom
        conn_order_target: order is the number of bonds along shortest path 
                           between two atoms. Return only those with this
                           value.
        '''
        res = []
        for a0 in self.get_atoms(z0):
            for a1 in self.get_atoms(z1):
                if (not ((z0 == z1) & (a0 >= a1))):
                    if conn_order_target == None:
                        res.append( (a0, a1) )
                    elif self.connection_order(a0,a1) == conn_order_target:
                        res.append( (a0, a1) )
        return res

    def get_bonded(self, iatom, conn_order_target):
        '''
        Returns list of all atoms connected to iatom with the target
        connection order
        '''
        res = []
        for a2 in range(len( self.z )):
            if self.connection_order(iatom,a2) == conn_order_target:
                res.append(a2)
        return res
    
    def get_dists(self):
        """
        returns dict mapping (z1,z2) to lists of distances between these
        elements. For (z1 != z2), res[(z1,z2)] and res[z2,z1] point to same
        list
        """
        res = {}
        z = self.z
        zs = np.unique(z)
        for z1 in zs:
            for z2 in zs:
                if z1 == z2:
                    res[(z1, z1)] = []
                elif z1 < z2:
                    res[(z1, z2)] = []
                    res[(z2, z1)] = res[(z1, z2)]
        for a0 in range(self.natom):
            for a1 in range(a0+1, self.natom):
                res[(z[a0], z[a1])].append(self.dist(a0, a1))
        return res

    def rcm(self, atoms = None):
        '''
          center of mass of requested atoms. if atoms is None, all atoms
          are included.
        '''
        if atoms is None:
            atoms = np.arange(len(self.z))
        natoms = len(atoms)
        ratoms = self.rcart[:,atoms]
        m = [ELEMENTS[i].mass for i in self.z[atoms]]
        # determine center of mass
        rcm = np.asarray([0.0,0.0,0.0])
        mtotal = 0
        for iatom in range(natoms):
            mtotal += m[iatom]
            ratom = ratoms[:,iatom].reshape((3,))
            rcm += np.multiply( m[iatom] , ratom )
        rcm = rcm/mtotal  
        return rcm

    def inertial_tensor(self,atoms):
        '''
        Moment of inertia tensor for requested atoms
        Note: Currently not efficiently coded
        
        Required input:
        atoms -- array of atom numbers to include (first atom is 0)
        
        returns:
        inertia -- 3x3 matrix of inertial tensor
        '''
        natoms = len(atoms)
        ratoms = self.rcart[:,atoms]
        m = [ELEMENTS[i].mass for i in self.z[atoms]]
        # determine center of mass
        rcm = np.asarray([0.0,0.0,0.0])
        mtotal = 0
        for iatom in range(natoms):
            mtotal += m[iatom]
            ratom = ratoms[:,iatom].reshape((3,))
            rcm += np.multiply( m[iatom] , ratom )
        rcm = rcm/mtotal
        # reference all atoms to center of mass
        for iatom in range(natoms):
            ratoms[:,iatom] = ratoms[:,iatom] - rcm;
        res = np.zeros([3,3])
        for ia in range(natoms):
            for i in range(3):
                res[i,i] += m[ia] * sum(ratoms[:,ia]**2)
        for ia in range(natoms):
            for i in range(3):
                for j in range(3):
                    res[i,j] -= m[ia]*ratoms[i,ia]*ratoms[j,ia]
        return res           

    def local_rotation(self, atom_or_bond):
        '''
        matrix for rotation to a unique orientation for atoms and bonds
        
        atom_or_bond --
                 if length 1: rotation to local inertial axes about that atom
                 if length 2: rotate x along bond axis and y,z to inertial
                              axes for 1-d rotation about that bond
        '''
        if isinstance(atom_or_bond, numbers.Number):
            atom_or_bond = [atom_or_bond]          
        if len(atom_or_bond) == 1:
             for conn_order in range(1,self.natom ):
                temp = [atom_or_bond[0]]
                temp.extend( self.get_bonded(atom_or_bond[0], conn_order) )
                atoms = np.unique(temp)
                iner = self.inertial_tensor(atoms)
                rot_matrix, evs = rotation_matrix_atom(iner)
                mindiff = np.min(np.diff(np.sort(evs)))
                if (mindiff > 1.0e-8) or (len(atoms) == self.natom):
                    break           
        elif len(atom_or_bond) == 2:
            for conn_order in range(1,self.natom ):
                temp = [atom_or_bond[0], atom_or_bond[1]]
                temp.extend( self.get_bonded(atom_or_bond[0], conn_order) )
                temp.extend( self.get_bonded(atom_or_bond[1], conn_order) )
                atoms = np.unique(temp)
                iner = self.inertial_tensor(atoms)
                rbond = self.rbond(atom_or_bond[0], atom_or_bond[1])
                rot_matrix, evs = rotation_matrix_bond(rbond,iner)
                if (abs(evs[1]-evs[0]) > 1.0e-8) or (len(atoms) == self.natom):
                    break
        else:
            raise Exception('Geometry:local_rotation atom_or_bond has incorrect length')
        return rot_matrix

def analyze_geometries(geoms, draw_histogram=False, fig_num=1):
    '''
       Get (and plot histogram of) distances between element types
    Input:
       geoms -- list of geometries
       draw_histogram -- single figure with panes for different (z1,z2)
                         nbins = len(data)/10
       fig_num -- number for the histogram figure
    Returns:
       dists -- dictionary with key of (z1,z2)
                                value of sorted list of distances (Angstroms)
                (z1,z2) and (z2,z1) hold identical data
    '''
    zs = []
    for geom in geoms:
      zs.extend(geom.z)
    zall = np.unique(np.array(zs))
    draw = {}
    for z1 in zall:
        for z2 in zall:
            draw[(z1,z2)] = []
            
    for geom in geoms:
        for k,v in geom.get_dists().items():
            draw[k].extend(v)
    dists = {}
    for k,v in draw.items():
        if len(v) > 0:
            dists[k] = np.sort(np.array(v))
    
    if draw_histogram:
        nplot = 0
        for k in list(dists.keys()):
            if (k[0] <= k[1]):
                nplot += 1
        nplots = len(dists)
        ncols = 2
        nrows = int(math.ceil(nplots*1.0/ncols))
        plt.figure(fig_num)
        iplot = 0
        for k,v in dists.items():
            if (k[0] <= k[1]):
                iplot += 1
                plt.subplot(ncols,nrows,iplot)
                plt.title('Z1, Z2= ' + str(k[0]) + ' ' + str(k[1]))
                plt.hist(v, bins=len(v) // 10)
        plt.savefig('out.png')
    return dists


def geom_diatomic(Zs, bond_vector):
    '''
     create diatomic molecule with:
       first atom of element Zs[0] at -bond_vector/2
       second atom of element Zs[1] at +bond_vector/2
    '''
    rvec = np.asarray(bond_vector)
    rgeom = np.vstack([-0.5*rvec,0.5*rvec]).T
    return Geometry(Zs, rgeom)
    
def geom_triatomic(Zs, rs, theta):
    '''
     create triatomic with structrure Zs[0] -- Zs[1] -- Zs[2] 
       (i.e. Zs[1] is the central atom)
     rs[0] is bond length from 0 to 1
     rs[1] is bond length form 1 to 2
     theta is the bond angle
     
     bond 1 lies along x axis,  bond 2 is in xy plane
    '''
    rvecs = []
    rvecs.append(np.array([rs[0], 0.0, 0.0]))
    rvecs.append(np.array([0.0, 0.0, 0.0]))
    rvecs.append(np.array([rs[1] * math.cos(theta),rs[1]*math.sin(theta), 0.0]))
    rcart = np.array(rvecs).T
    return Geometry(Zs,rcart)
    
def geoms_diatomic_xyz(Zs, bond_length):
    '''
     create diatomic molecule along x,y and z axis
     
     Zs -- atomic number of the two elements
     bond_length - in angstroms
     
     Returns list of 3 geometries, with molecule orientied along x, then y
     then z axis
     
    '''
    rvecs = []
    rvecs.append(np.array([bond_length,0.0,0.0]))
    rvecs.append(np.array([0.0,bond_length,0.0]))
    rvecs.append(np.array([0.0,0.0,bond_length]))
    return [geom_diatomic(Zs,rvec) for rvec in rvecs]

def random_rotation(geom):
    rot_axis = np.random.uniform(0,1,3)
    rot_angle = np.random.uniform(0,3*np.pi,1)[0]
    rot = get_axis_rotation_matrix(rot_axis,rot_angle)
    geom.rcart = np.asarray(np.dot(rot,geom.rcart))
    return geom

def random_diatomics(ngeom, Zs, bond_range, rotate = True):
    bond_length = np.random.uniform(bond_range[0],bond_range[1], ngeom)
    geoms = [geom_diatomic(Zs,[bl,0.0,0.0]) for bl in bond_length]
    if rotate:
        geoms = [random_rotation(geom) for geom in geoms]
    return geoms
    
def random_triatomics(ngeom, Zs, bond_range0, bond_range1, angle_range, 
                      rotate = True):
    bl0 = np.random.uniform(bond_range0[0],bond_range0[1], ngeom)
    bl1 = np.random.uniform(bond_range1[0],bond_range1[1], ngeom)
    theta = np.random.uniform(angle_range[0],angle_range[1], ngeom)
    geoms = [geom_triatomic(Zs,[bl0[i],bl1[i]],theta[i]) for i in range(ngeom)]
    if rotate:
        geoms = [random_rotation(geom) for geom in geoms]
    return geoms

def get_geoms(ngeom = 50):
    geom_list = []
    # HCN: C-H 1.068 C-N 1.156 ± 0.001
    #geom_list.extend(random_triatomics(ngeom, [1,6,7],[1.0,1.1],[1.2,1.3],
    #                   [(10.0)*math.pi/180.0,(20.0)*math.pi/180.0]))
    geom_list.extend(random_triatomics(ngeom, [1,8,1],[0.7,1.1],[0.7,1.1],
                       [(104.7-20.0)*math.pi/180.0,(104.7+20.0)*math.pi/180.0]))
    geom_dict = []    
    for geom in geom_list:
        geom_dict.append({'geom': geom})
    return geom_dict

def get_triatomic_gdict(batch_size,ntrain,ntest):
    gdict = {}
    gdict['train'] = [get_geoms(batch_size) for i in range(ntrain)]
    gdict['test']  = [get_geoms(batch_size) for i in range(ntest) ]
    return gdict


# https://openbabel.org/docs/dev/FileFormats/XYZ_cartesian_coordinates_format.html
def get_benzene():
    Zbenzene = [6,1,6,1,6,1,6,1,6,1,6,1]
    rbenzene = np.array([
            [  0.00000,        1.40272,        0.00000 ],
            [  0.00000,        2.49029,        0.00000 ],
            [ -1.21479,        0.70136,        0.00000 ],
            [ -2.15666,        1.24515,        0.00000 ],
            [ -1.21479,       -0.70136,        0.00000 ],
            [ -2.15666,       -1.24515,        0.00000 ],
            [  0.00000,       -1.40272,        0.00000 ],
            [  0.00000,       -2.49029,        0.00000 ],
            [  1.21479,       -0.70136,        0.00000 ],
            [  2.15666,       -1.24515,        0.00000 ],
            [  1.21479,        0.70136,        0.00000 ],
            [  2.15666,        1.24515,        0.00000 ]]).T
    
    return Geometry(Zbenzene,rbenzene)
         
if __name__ == "__main__":
    z = [6, 6, 1, 1, 1, 1, 1, 1]
    rcart = [[  0.00000000e+00,   0.00000000e+00,   1.00880600e+00,
         -1.00880600e+00,   5.04572000e-01,   5.04234000e-01,
         -5.04403000e-01,  -5.04403000e-01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.95000000e-04,   8.73554000e-01,  -8.73749000e-01,
         -8.73651000e-01,   8.73651000e-01],
       [  0.00000000e+00,   1.54000000e+00,  -3.56667000e-01,
          1.89666700e+00,   1.89666700e+00,   1.89666700e+00,
         -3.56667000e-01,  -3.56667000e-01]]
    geom = Geometry(z,rcart)
    ratom = geom.local_rotation(1)
    rbond = geom.local_rotation([1,2])
    
    
