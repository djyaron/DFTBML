from .tfspline import Bcond

ANGSTROM2BOHR = 1.889725989
HARTREE = 627.50947406

# Lower cut-off of each pairwise interaction is set to 0.05 Angstroms below
# the minimum of that pairwise distance in the cec or aec dataset
# Upper cut-offs are set to include certain number of peaks
# in the distribution of each pairwise distance in the cec dataset

# Short cut-offs: include the first peak
# Medium cut-offs: include the third peak
# Long cut-offs: include the fourth peak
# Extend cut-offs: uniformly set to 5.00 Angstroms
# Full cut-offs: Lower set to 0.00, upper set to 10.00, both uniformly

# Cut-off radii for ANI-1ccx_energy_clean ("cec") dataset

CUTOFFS_SRT = {(1, 1): (0.63, 2.10),
               (1, 6): (0.60, 1.60),
               (1, 7): (0.59, 1.60),
               (1, 8): (0.66, 1.50),
               (6, 6): (1.04, 1.80),
               (6, 7): (0.95, 1.80),
               (6, 8): (1.00, 1.80),
               (7, 7): (0.99, 1.80),
               (7, 8): (0.93, 1.80),
               (8, 8): (1.06, 1.80)}
#FH: Set the first value to 0 as a test to see if this fixes SKF translation issues
CUTOFFS_SRT={k : (0, v[1]) for k, v in CUTOFFS_SRT.items()}

CUTOFFS_MED = {(1, 1): (0.63, 3.30),
               (1, 6): (0.60, 3.00),
               (1, 7): (0.59, 3.00),
               (1, 8): (0.66, 3.00),
               (6, 6): (1.04, 3.30),
               (6, 7): (0.95, 3.30),
               (6, 8): (1.00, 3.30),
               (7, 7): (0.99, 3.30),
               (7, 8): (0.93, 3.20),
               (8, 8): (1.06, 3.20)}
CUTOFFS_LNG = {(1, 1): (0.63, 4.80),
               (1, 6): (0.60, 3.70),
               (1, 7): (0.59, 3.70),
               (1, 8): (0.66, 3.70),
               (6, 6): (1.04, 4.00),
               (6, 7): (0.95, 4.00),
               (6, 8): (1.00, 4.00),
               (7, 7): (0.99, 3.80),
               (7, 8): (0.93, 3.80),
               (8, 8): (1.06, 3.80)}
#FH: Set the first value to 0 as a test to see if extending the range slightly helps with SKFs
CUTOFFS_LNG={k : (0, v[1]) for k, v in CUTOFFS_LNG.items()}


CUTOFFS_EXT = {(1, 1): (0.63, 5.00),
               (1, 6): (0.60, 5.00),
               (1, 7): (0.59, 5.00),
               (1, 8): (0.66, 5.00),
               (6, 6): (1.04, 5.00),
               (6, 7): (0.95, 5.00),
               (6, 8): (1.00, 5.00),
               (7, 7): (0.99, 5.00),
               (7, 8): (0.93, 5.00),
               (8, 8): (1.06, 5.00)}
CUTOFFS_FUL = {(1, 1): (0.00, 10.00),
               (1, 6): (0.00, 10.00),
               (1, 7): (0.00, 10.00),
               (1, 8): (0.00, 10.00),
               (6, 6): (0.00, 10.00),
               (6, 7): (0.00, 10.00),
               (6, 8): (0.00, 10.00),
               (7, 7): (0.00, 10.00),
               (7, 8): (0.00, 10.00),
               (8, 8): (0.00, 10.00)}

# Cut-off radii for Au_energy_clean dataset
CUTOFFS_AST = {(1, 1): (0.63, 2.10),
               (1, 6): (0.60, 1.60),
               (1, 7): (0.59, 1.70),
               (1, 8): (0.66, 1.50),
               (1, 79): (1.32, 3.50),
               (6, 6): (1.04, 1.80),
               (6, 7): (0.95, 1.80),
               (6, 8): (1.00, 1.80),
               (6, 79): (0.83, 2.70),
               (7, 7): (0.99, 1.80),
               (7, 8): (0.93, 1.80),
               (7, 79): (0.97, 2.70),
               (8, 8): (1.06, 1.80),
               (8, 79): (0.81, 2.70),
               (79, 79): (2.37, 3.40)}
CUTOFFS_AMD = {(1, 1): (0.63, 3.30),
               (1, 6): (0.60, 3.20),
               (1, 7): (0.59, 3.00),
               (1, 8): (0.66, 3.00),
               (1, 79): (1.32, 6.20),
               (6, 6): (1.04, 2.80),
               (6, 7): (0.95, 2.80),
               (6, 8): (1.00, 2.80),
               (6, 79): (0.83, 6.50),
               (7, 7): (0.99, 3.00),
               (7, 8): (0.93, 3.00),
               (7, 79): (0.97, 6.30),
               (8, 8): (1.06, 3.20),
               (8, 79): (0.81, 6.20),
               (79, 79): (2.37, 6.00)}
CUTOFFS_ALG = {(1, 1): (0.63, 4.60),
               (1, 6): (0.60, 3.90),
               (1, 7): (0.59, 3.70),
               (1, 8): (0.66, 3.60),
               (1, 79): (1.32, 10.00),
               (6, 6): (1.04, 4.20),
               (6, 7): (0.95, 4.00),
               (6, 8): (1.00, 4.00),
               (6, 79): (0.83, 10.00),
               (7, 7): (0.99, 4.00),
               (7, 8): (0.93, 3.80),
               (7, 79): (0.97, 10.00),
               (8, 8): (1.06, 3.70),
               (8, 79): (0.81, 10.00),
               (79, 79): (2.37, 7.50)}
CUTOFFS_AEX = {(1, 1): (0.63, 5.00),
               (1, 6): (0.60, 5.00),
               (1, 7): (0.59, 5.00),
               (1, 8): (0.66, 5.00),
               (1, 79): (1.32, 10.00),
               (6, 6): (1.04, 5.00),
               (6, 7): (0.95, 5.00),
               (6, 8): (1.00, 5.00),
               (6, 79): (0.83, 10.00),
               (7, 7): (0.99, 5.00),
               (7, 8): (0.93, 5.00),
               (7, 79): (0.97, 10.00),
               (8, 8): (1.06, 5.00),
               (8, 79): (0.81, 10.00),
               (79, 79): (2.37, 7.50)}
CUTOFFS_AFL = {(1, 1): (0.00, 10.00),
               (1, 6): (0.00, 10.00),
               (1, 7): (0.00, 10.00),
               (1, 8): (0.00, 10.00),
               (1, 79): (0.00, 10.00),
               (6, 6): (0.00, 10.00),
               (6, 7): (0.00, 10.00),
               (6, 8): (0.00, 10.00),
               (6, 79): (0.00, 10.00),
               (7, 7): (0.00, 10.00),
               (7, 8): (0.00, 10.00),
               (7, 79): (0.00, 10.00),
               (8, 8): (0.00, 10.00),
               (8, 79): (0.00, 10.00),
               (79, 79): (0.00, 10.00)}

# Map natural language to cut-off radii dictionaries
CUTOFFS = {"short": CUTOFFS_SRT.copy(),
           "medium": CUTOFFS_MED.copy(),
           "long": CUTOFFS_LNG.copy(),
           "extend": CUTOFFS_EXT.copy(),
           "full": CUTOFFS_FUL.copy(),
           "au_short": CUTOFFS_AST.copy(),
           "au_medium": CUTOFFS_AMD.copy(),
           "au_long": CUTOFFS_ALG.copy(),
           "au_extend": CUTOFFS_AEX.copy(),
           "au_full": CUTOFFS_AFL.copy()}

# Map natural language to boundary conditions
BCONDS = {'natural': (Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)),
          'vanishing': (Bcond(0, 2, 0.0), Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0))}

ALIAS2TARGET = {'dt': 'dftb.total_energy',  # Dftb Total
                'de': 'dftb.elec_energy',  # Dftb Electronic
                'dr': 'dftb.rep_energy',  # Dftb Repulsive
                'pt': 'dftb_plus.total_energy',  # dftb Plus Total
                'pe': 'dftb_plus.elec_energy',  # dftb Plus Electronic
                # Non-SCC energy plus other contributions to
                # electronic energy (SCC, spin, ...)
                'pr': 'dftb_plus.rep_energy',  # dftb Plus Repulsive
                # Pairwise contribution to total energy
                'hd': 'hf_dz.energy',  # Hf Dz
                'ht': 'hf_tz.energy',
                'hq': 'hf_qz.energy',
                'wd': 'wb97x_dz.energy',  # Wb97x Dz
                'wt': 'wb97x_tz.energy',
                'md': 'mp2_dz.energy',  # Mp2 Dz
                'mt': 'mp2_tz.energy',
                'mq': 'mp2_qz.energy',
                'td': 'tpno_ccsd(t)_dz.energy',  # Tpno Dz
                'nd': 'npno_ccsd(t)_dz.energy',  # Npno Dz
                'nt': 'npno_ccsd(t)_tz.energy',
                'cc': 'ccsd(t)_cbs.energy',
                # Entries in Au dataset
                'dm': 'dftb.mermin_free_energy',
                'dem': 'dftb.elec_mermin_energy',
                'ft': 'fhi_aims_md.total_energy',
                'fm': 'fhi_aims_md.mermin_energy',  # total system energy with mermin
                'fh': 'fhi_aims_md.homo_energy',
                'fl': 'fhi_aims_md.lumo_energy',
                # Entries in Au dataset with dispersion
                'p0': 'dftb_plus.0K_energy',
                'pd': 'dftb_plus.disp_energy',
                'pf': 'dftb_plus.force_related_energy',
                'pm': 'dftb_plus.mermin_energy',  # total system energy with mermin
                'pc': 'dftb_plus.rep_corrected_energy'  # force_related - rep
                }

TARGET2ALIAS = {target: alias for alias, target in ALIAS2TARGET.items()}

ATOM2SYM = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
            6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
            21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
            26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
            31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
            36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
            41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
            46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs',
            56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
            61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
            66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
            71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re',
            76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
            81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
            86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
            91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
            96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
            101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
            106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt'}

SYM2ATOM = {sym: atom for atom, sym in ATOM2SYM.items()}

MAX_ANGULAR_MOMENTUM = {1: 's', 6: 'p', 7: 'p', 8: 'p', 79: 'd'}
