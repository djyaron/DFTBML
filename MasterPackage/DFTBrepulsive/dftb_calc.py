from multiprocessing import Pool
from os.path import join
from subprocess import run

from h5py import File

from consts import ATOM2SYM, MAX_ANGULAR_MOMENTUM
from util import path_check


def dftb_calc(opts):
    r"""Run DFTB+ calculations in parallel

    Args:
        opts (dict): options

    Returns:
        None
    """
    path_check(opts['save_dir'])
    n_worker = opts.get('n_worker', 1)
    pool = Pool(processes=n_worker)
    coords, atomic_numbers, i_mol_confs = flat_dataset(opts)
    args = [(c, a, i, opts) for c, a, i in zip(coords, atomic_numbers, i_mol_confs)]
    out = pool.starmap_async(dftb_single_conf, args)
    out.wait()


def flat_dataset(opts):
    r"""Extract coordinates, atomic numbers, molecule name and conformation indices

    Args:
        opts (dict):

    Returns:
        coords (list): coordinates of each conformation
        atomic_numbers (list): atomic numbers of each conformation
        i_mol_confs (list): (mol_name, i_conf) tuples of each conformation

    """
    dataset_path = opts['dataset_path']
    coords = []
    atomic_numbers = []
    i_mol_confs = []
    with File(dataset_path, 'r') as dataset:
        for mol, moldata in dataset.items():
            c = moldata['coordinates'][()]
            coords.extend(c)
            nconfs = len(c)
            a = moldata['atomic_numbers'][()]
            atomic_numbers.extend([a] * nconfs)
            i_mol_confs.extend([(mol, i_conf) for i_conf in range(nconfs)])
    return coords, atomic_numbers, i_mol_confs


def dftb_single_conf(coords, atomic_numbers, i_mol_conf, opts):
    r"""Run DFTB+ for a single conformation

    Args:
        coords:
        atomic_numbers:
        i_mol_conf:
        opts:

    Returns:

    """
    dftb_dir = opts['dftb_dir']
    save_dir = opts['save_dir']

    mol, i_conf = i_mol_conf
    save_path = join(save_dir, f"{mol}__{i_conf}/")
    path_check(save_path)  # create working dir

    # with Timer(f'DFTB+ on {mol}, conf {i_conf}'):
    # noinspection PyBroadException
    try:
        write_dftb_input(coords, atomic_numbers, i_mol_conf, opts, save_path)
        with open(join(save_path, 'dftb.stdout'), 'w') as f:
            run(join(dftb_dir, 'bin/dftb+'), cwd=save_path, stdout=f)
    except:
        # Create an empty file named 'failed' for failed calculations
        with open(join(save_path, 'failed'), 'w'):
            pass


def write_dftb_input(coords, atomic_numbers, i_mol_conf, opts, save_path):
    hsd_path = join(save_path, "dftb_in.hsd")

    # Geometry block
    n_atoms = len(atomic_numbers)
    mol, i_conf = i_mol_conf
    conf_info = f"{mol} conformation {i_conf}"
    xyz = coords_to_xyz(coords, atomic_numbers, conf_info)
    geo_block = ["Geometry = xyzFormat {",
                 str(n_atoms),
                 *xyz,
                 "}"]

    # Driver block
    # drv_block = ["Driver = LBFGS {",
    #             "  LatticeOpt = NO",
    #             "  MaxSteps = -1",
    #             "}"]
    drv_block = ["Driver = {}"]

    # Hamiltonian block
    # Filling
    fill = []
    if opts.get('FermiTemp'):
        fill = ["  Filling = Fermi {",
                f"    Temperature [eV] = {opts.get('FermiTemp')}",
                "  }"]
    # MaxAngularMomentum
    mmt = ["  MaxAngularMomentum = {",
           *get_max_angular_momentum(atomic_numbers),
           "  }"]
    # Mixer
    mixer = ["  Mixer = Broyden {",
             "    MixingParameter = 0.01",
             "    InverseJacobiWeight = 0.01",
             "    MinimalWeight = 1.0",
             "    MaximalWeight = 1e5",
             "    WeightFactor = 1e-2",
             " }"]
    # SCC
    scc = [f"  ShellResolvedSCC = {'YES' if opts.get('ShellResolvedSCC') else 'NO'}",
           "  SCC = YES",
           "  MaxSCCIterations = 200",
           "  SCCTolerance = 1e-05"]
    # SlaterKosterFiles
    skf = ["  SlaterKosterFiles = Type2FileNames{",
           f"    Prefix = {opts['skf_dir']}",
           '    Separator = "-"',
           '    Suffix = ".skf"',
           "    LowerCaseTypeName = No",
           "  }"]
    # Dispersion
    disp = ["  Dispersion = {",
            "    MBD {",
            "      NOmegaGrid = 15",
            "      Beta = 0.95",
            "      KGrid = {",
            "1 1 1",
            "      }",
            "    }",
            "  }"]
    # Solver
    # solver = ["  Solver = ELPA{",
    #           "  }"]
    solver = []
    # Combine sub-blocks into a Hamiltonian block
    h_block = ["Hamiltonian = DFTB{",
               *fill, *mmt, *mixer, *scc, *skf, *disp, *solver,
               "}"]

    # Options block
    opt_block = ["Options {",
                 "  WriteResultsTag = Yes",
                 "  WriteDetailedOut = Yes",
                 "  TimingVerbosity = -1",
                 "}"]

    # Write to file
    hsd_lines = [*geo_block, *drv_block, *h_block, *opt_block]
    with open(hsd_path, 'w') as hsd:
        hsd.writelines("%s\n" % l for l in hsd_lines)


def coords_to_xyz(coords, atomic_numbers, comment=None):
    res = [comment] if comment else []
    xyz = [f"{ATOM2SYM[a]} {x} {y} {z}"
           for a, (x, y, z) in zip(atomic_numbers, coords)]
    res.extend(xyz)
    return res


def get_max_angular_momentum(atomic_numbers):
    atom_types = sorted(set(atomic_numbers))
    res = [f'    {ATOM2SYM[a]} = "{MAX_ANGULAR_MOMENTUM[a]}"' for a in atom_types]
    return res
