from math import ceil
from scipy.interpolate import PPoly
from typing import List, Tuple
from util import path_check
from util import mpl_default_setting

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

HARTREE = 627.50947406

# TODO: SKF.__init__() should evaluate the splines on the last grid point given by spl_grid

class SKF:
    def __init__(self, skf_path: str, spl_ngrid: int = 500, spl_grid: np.ndarray = None) -> None:
        r"""SKF object with data parsed and converted into DataFrames

        Args:
            skf_path (str): Path to skf file
            spl_ngrid (int): Number of grid points to evaluate splines on

        Returns:
            None

        Examples:
            >>> from skfvis import SKF
            >>> skf_path = "PATH_TO_SLATER_KOSTER_FILE"
            >>> skf = SKF(skf_path, spl_ngrid=500)
        """
        with open(skf_path, 'r') as f:
            skf_raw = f.readlines()
        self.spl_ngrid = spl_ngrid

        # Determine SKF file type and data blocks included
        self.atom_pair = os.path.split(skf_path)[-1].rsplit('.')[0].split('-')
        # Check if the SKF describes homonuclear interaction
        self.homo = (self.atom_pair[0] == self.atom_pair[1])
        # Check if the SKF is in extended format
        self.extend = (skf_raw[0][0] == '@')
        # Check if the SKF includes spline and document block
        self.spl_enabled = False
        self.doc_enabled = False
        # Determine the starting line number of the spline and the documentation block
        for i_line, line in enumerate(skf_raw):
            if 'spline' in line or 'Spline' in line:
                self.spl_enabled = True
                self.spl_start = i_line
            if '<Documentation>' in line:
                self.doc_enabled = True
                self.doc_start = i_line
                break

        # Determine available entries in the SKF
        if self.extend:
            # extended format
            self.HS_ENTRIES = ('Hff0', 'Hff1', 'Hff2', 'Hff3', 'Hdf0', 'Hdf1', 'Hdf2', 'Hdd0',
                               'Hdd1', 'Hdd2', 'Hpf0', 'Hpf1', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1',
                               'Hsf0', 'Hsd0', 'Hsp0', 'Hss0', 'Sff0', 'Sff1', 'Sff2', 'Sff3',
                               'Sdf0', 'Sdf1', 'Sdf2', 'Sdd0', 'Sdd1', 'Sdd2', 'Spf0', 'Spf1',
                               'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssf0', 'Ssd0', 'Ssp0', 'Sss0')
        else:
            # simple format
            self.HS_ENTRIES = ('Hdd0', 'Hdd1', 'Hdd2', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsd0',
                               'Hsp0', 'Hss0', 'Sdd0', 'Sdd1', 'Sdd2', 'Spd0', 'Spd1', 'Spp0',
                               'Spp1', 'Ssd0', 'Ssp0', 'Sss0')

        """## **Split SKF into blocks and convert to DataFrames**"""

        # Split SKF into blocks
        # Split HS block and convert it to dataframe
        HS_start = int(self.extend)
        if self.spl_enabled and self.doc_enabled:
            header = skf_raw[HS_start:min(self.spl_start, self.doc_start)]
        elif self.spl_enabled and not self.doc_enabled:
            header = skf_raw[HS_start:self.spl_start]
        elif not self.spl_enabled and self.doc_enabled:
            header = skf_raw[HS_start:self.doc_start]
        else:
            header = skf_raw[HS_start:]
        # Read HS block in SKF
        # Generate integration grid
        if self.extend:
            # skip the first line
            gridDist, nGridPoints = header[1].split()
            if self.homo:
                intTable = header[4:]
            else:
                intTable = header[3:]
        else:
            gridDist, nGridPoints = header[0].split()
            if self.homo:
                intTable = header[3:]
            else:
                intTable = header[2:]
        gridDist = float(gridDist)
        nGridPoints = int(nGridPoints)
        grid = np.linspace(gridDist, gridDist * (nGridPoints + 1), nGridPoints)
        grid = pd.Series(grid, name='distance')
        # Convert the integration table to DataFrame
        intTable = list(line.split() for line in intTable)
        intTable = pd.DataFrame(intTable, columns=self.HS_ENTRIES)
        # noinspection PyTypeChecker
        intTable = intTable.apply(pd.to_numeric, errors='raise')
        # Concatenate integration grid and integration table
        self.HS_val = pd.concat([grid, intTable], axis=1)

        # Split spline block and convert it to dataframe
        if self.spl_enabled:
            if self.doc_enabled:
                spline = skf_raw[self.spl_start + 1:self.doc_start]
            else:
                spline = skf_raw[self.spl_start + 1:]

            nInt, cutoff = spline[0].split()
            nInt = int(nInt)
            cutoff = float(cutoff)

            _spl_grid = np.linspace(0.0, cutoff, self.spl_ngrid) if spl_grid is None else spl_grid

            # Coefficients of the exponential part
            expCoeff = spline[1].split()
            expCoeff = list(float(a) for a in expCoeff)
            cub_start = float(spline[2].split()[0])
            exp_range = [0.0, cub_start]
            expCoeff = np.concatenate([exp_range, expCoeff]).reshape(1, -1)
            expCoeff = pd.DataFrame(expCoeff, columns=('start', 'end', 'a1', 'a2', 'a3'))

            # Coefficients of the cubic spline part
            cubCoeff = spline[2:-1]
            cubCoeff = list(line.split() for line in cubCoeff)
            cubCoeff = pd.DataFrame(cubCoeff, columns=('start', 'end', 'c0', 'c1', 'c2', 'c3'))
            # noinspection PyTypeChecker
            cubCoeff = cubCoeff.apply(pd.to_numeric, errors='raise')

            # Coefficients of the 5th-order spline part
            fifCoeff = spline[-1].split()
            fifCoeff = list(float(a) for a in fifCoeff)
            fifCoeff = np.array(fifCoeff).reshape(1, -1)
            fifCoeff = pd.DataFrame(fifCoeff, columns=('start', 'end', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'))

            # Evaluate exponential part
            exp_start = expCoeff['start'][0]
            exp_end = expCoeff['end'][0]
            expGrid = _spl_grid[(exp_start <= _spl_grid) & (_spl_grid < exp_end)]
            a1 = expCoeff['a1'][0]
            a2 = expCoeff['a2'][0]
            a3 = expCoeff['a3'][0]
            expVal = np.exp(-a1 * expGrid + a2) + a3

            # Evaluate cubic spline part
            #: x is the vector of breaking points (knots)
            x = cubCoeff.loc[:, 'start':'end']  # slice all the start and end points
            x = np.unique(x)  # deduplication creates a vector of all the start points and the last end point
            #: c is the polynomial coefficients. For its structure, refer to
            #: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html#scipy.interpolate.PPoly
            c = cubCoeff.loc[:, 'c0':]  # slice the coefficient block
            c = c.iloc[:, ::-1]  # reverse the order of the columns
            c = np.array(c).T  # pd.DataFrame.T may crash iPython session
            # Slice the grid points in the range of cubic splines
            cubGrid = _spl_grid[(x[0] <= _spl_grid) & (_spl_grid < x[-1])]
            # Create and evaluate cubic splines
            cubSpl = PPoly(c, x)
            cubVal = cubSpl(cubGrid)

            # Evaluate 5-th order spline part
            x = fifCoeff.loc[:, 'start':'end']
            x = np.unique(x)
            c = fifCoeff.loc[:, 'c0':]
            c = c.iloc[:, ::-1]
            c = np.array(c).T
            fifGrid = _spl_grid[(x[0] <= _spl_grid) & (_spl_grid < x[-1])]
            fifSpl = PPoly(c, x)
            fifVal = fifSpl(fifGrid)

            # DataFrame of the values
            grid = _spl_grid[:-1]
            grid = pd.Series(grid, name='distance')
            val = np.concatenate([expVal, cubVal, fifVal])
            val = pd.Series(val, name='Spline')
            self.Spl_val = pd.concat([grid, val], axis=1)

    def get_PLOT_ENTRIES(self, entries: List[str]) -> Tuple[str]:
        r"""Convert vague entries to specified ones

        Args:
            entries (List[str]): SKF entries to be plotted
                Use 'H' to include all the Hamiltonian matrices.
                Use 'S' to include all the overlap matrices.
                Use 'Spline' or 'spline' to include the splines (if applicable).
                Use 'All' or 'all' to include all available entries

        Returns:
            Tuple[str]: specified plotting entries
        """
        # Check entries
        if 'All' in entries or 'all' in entries:
            if self.spl_enabled:
                _entries = [*self.HS_ENTRIES, 'Spline']
            else:
                _entries = [*self.HS_ENTRIES]
        else:
            _entries = []
            if 'H' in entries:
                _all_H = list(e for e in self.HS_ENTRIES if 'H' in e)
                _entries.extend(_all_H)
            else:
                for e in entries:
                    if 'H' in e:
                        _entries.append(e)
            if 'S' in entries:
                _all_S = list(e for e in self.HS_ENTRIES if 'S' in e and len(e) == 4)
                _entries.extend(_all_S)
            else:
                for e in entries:
                    if 'S' in e and len(e) == 4:
                        _entries.append(e)
            if 'Spline' in entries or 'spline' in entries:
                if not self.spl_enabled:
                    print("Splines are not included in the specified SKF")
                else:
                    _entries.append('Spline')
        return tuple(_entries)

    def plot(self, entries: List[str], figsize: tuple = (16, 10),
             HS_ylim: tuple = (-0.5, 0.5),
             Spl_ylim: tuple = (-10, 200),
             save: bool = False, save_path: str = None):
        r"""Plot data in SKF

        Args:
            entries (List[str]): SKF entries to be plotted
                Use 'H' to include all the Hamiltonian matrices.
                Use 'S' to include all the overlap matrices.
                Use 'Spline' or 'spline' to include the splines (if applicable).
                Use 'All' or 'all' to include all available entries
            figsize (tuple): Size of output figure
            HS_ylim (tuple): y range of plots of HS data
            Spl_ylim (tuple): y range of plots of spline data
            save (bool): Save the plot when set to True
            save_path (str): Path to save the plot. Will be set to atom pair of
                the SKF when set to None

        Returns:
            None

        Examples:
            >>> from skfvis import SKF
            >>> skf_path = "SKF_PATH"
            >>> skf = SKF(skf_path)
            >>> skf.plot(entries=['all'])
        """

        # Plot data
        mpl_default_setting()  # Use default matplotlib preferences
        PLOT_ENTRIES = self.get_PLOT_ENTRIES(entries)
        res = {}
        nrows = ncols = ceil(len(PLOT_ENTRIES) ** 0.5)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for row in range(nrows):
            for col in range(ncols):
                plot_id = col + row * ncols
                ax = axes[row, col]
                try:
                    entry = PLOT_ENTRIES[plot_id]
                except IndexError:
                    fig.delaxes(ax)
                    continue
                if entry == "Spline":
                    data = self.Spl_val
                    ylim = Spl_ylim
                else:
                    data = self.HS_val
                    ylim = HS_ylim
                res[entry] = data
                ax = sns.lineplot(x='distance', y=entry, data=data, ax=ax)
                ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
                ax.set_ylabel(f"{entry} (hartree)")
                ax.set_ylim(ylim)
        plt.show()
        if save:
            if save_path is None:
                save_path = f"{self.atom_pair[0]}-{self.atom_pair[1]}.png"
            else:
                path_check(save_path)
            plt.savefig(save_path)

        return res


def skfs_plot(skfs: List[SKF], output: str,
              entries: List[str], figsize: tuple = (16, 10),
              HS_ylim: tuple = (-0.5, 0.5), Spl_ylim: tuple = (-10, 200),
              save: bool = True, save_path: str = None) -> None:
    r"""Combined or animated plot of multiple skfs

    Args:
        skfs (List[SKF]):
        output (str): specify the type of plot. Available options:
            "combined": overlay the plots of skfs
            "animated": combine the plots sequentially into an animated GIF
        entries (List[str]): SKF entries to be plotted
            Use 'H' to include all the Hamiltonian matrices.
            Use 'S' to include all the overlap matrices.
            Use 'Spline' or 'spline' to include the splines (if applicable).
            Use 'All' or 'all' to include all available entries
        figsize (tuple): Size of output figure
        HS_ylim (tuple): y range of plots of HS data
        Spl_ylim (tuple): y range of plots of spline data
        save (bool): Save the plot when set to True
        save_path (str): Path to save the plot. Will be set to atom pair of
            the SKF when set to None, must be specified when generating animated
            GIF in order to save intermediate plots in the same directory.

    Returns:
        None
    """
    PLOT_ENTRIES = skfs[0].get_PLOT_ENTRIES(entries)
    HS_enabled = (len(PLOT_ENTRIES) > 1)
    spl_enabled = ('Spline' in PLOT_ENTRIES)

    if output == 'combined':
        # Merge HS and spline data across SKFs
        if HS_enabled:
            HS_ENTRIES = tuple([entry for entry in PLOT_ENTRIES if entry != 'Spline'])
            HS_val_merged = []
            for i_skf, skf in enumerate(skfs):
                HS_val = skf.HS_val.loc[:, ['distance', *HS_ENTRIES]]
                id_col = np.ones(HS_val.shape[0], dtype=int)
                id_col = pd.Series(id_col, name='SKF_ID')
                HS_val_with_id = pd.concat([id_col, HS_val], axis=1)
                HS_val_merged.append(HS_val_with_id)
            HS_val_merged = pd.concat(HS_val_merged)
            """
            --------------------------
            Structure of HS_val_merged
            --------------------------
            SKF_ID    distance    Hdd0    Hdd1    ...
            0         ????        ????    ????    ...
            0         ????        ????    ????    ...
            0         ....        ....    ....    ...
            1         ????        ????    ????    ...
            1         ????        ????    ????    ...
            1         ....        ....    ....    ...
            ...
            """
        if spl_enabled:
            Spl_val_merged = []
            for i_skf, skf in enumerate(skfs):
                Spl_val = skf.Spl_val
                id_col = np.ones(Spl_val.shape[0], dtype=int)
                id_col = pd.Series(id_col, name='SKF_ID')
                Spl_val_with_id = pd.concat([id_col, Spl_val], axis=1)
                Spl_val_merged.append(Spl_val_with_id)
            Spl_val_merged = pd.concat(Spl_val_merged)
            """
            ---------------------------
            Structure of Spl_val_merged
            ---------------------------
            SKF_ID    distance    Spline  ...
            0         ????        ????    ...
            0         ????        ????    ...
            0         ....        ....    ...
            1         ????        ????    ...
            1         ????        ????    ...
            1         ....        ....    ...
            ...
            """

        # Combined plot
        mpl_default_setting()  # Use default matplotlib preferences
        nrows = ncols = ceil(len(PLOT_ENTRIES) ** 0.5)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for row in range(nrows):
            for col in range(ncols):
                plot_id = col + row * ncols
                ax = axes[row, col]
                try:
                    entry = PLOT_ENTRIES[plot_id]
                except IndexError:
                    fig.delaxes(ax)
                    continue
                if entry == "Spline":
                    # noinspection PyUnboundLocalVariable
                    data = Spl_val_merged
                    ylim = Spl_ylim
                else:
                    # noinspection PyUnboundLocalVariable
                    data = HS_val_merged
                    ylim = HS_ylim
                ax = sns.lineplot(x='distance', y=entry, hue='SKF_ID', data=data, ax=ax)
                ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
                ax.set_ylim(ylim)
        if save:
            if save_path is None:
                atom_pair = skfs[0].atom_pair
                save_path = f"{atom_pair[0]}-{atom_pair[1]}_combined.png"
            else:
                path_check(save_path)
            plt.savefig(save_path)

    elif output == 'animated':
        assert save_path is not None, "save_path must be specified when generating animation"
        save_dir = os.path.split(save_path)[0]
        atom_pair = skfs[0].atom_pair
        # Plot and save frames
        for i_skf, skf in enumerate(skfs):
            frame_name = f"{atom_pair[0]}-{atom_pair[1]}_{i_skf}.png"
            frame_path = os.path.join(save_dir, frame_name)
            skf.plot(entries, figsize, HS_ylim, Spl_ylim, save=True, save_path=frame_path)
        with imageio.get_writer(save_path, mode='I') as writer:
            for i_skf in range(len(skfs)):
                frame_name = f"{atom_pair[0]}-{atom_pair[1]}_{i_skf}.png"
                frame_path = os.path.join(save_dir, frame_name)
                image = imageio.imread(frame_path)
                writer.append_data(image)
        # Remove frames
        for i_skf in range(len(skfs)):
            frame_name = f"{atom_pair[0]}-{atom_pair[1]}_{i_skf}.png"
            frame_path = os.path.join(save_dir, frame_name)
            os.remove(frame_path)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    # skf_path = "/home/francishe/opt/dftb+/slakos/auorg-1-1/C-N.skf"
    # skf_path = "/home/francishe/Documents/DFTBrepulsive/SKF/auorg-1-1-rep-50-3-short-None/Au-C.skf"
    # skf_path = "/home/francishe/Documents/DFTBrepulsive/SKF/aed_convex/H-H.skf"
    skf_path = "/home/francishe/Documents/DFTBrepulsive/SKF/a1k/H-H.skf"

    #: SKF entries to be plotted
    # entries = ['spline']
    # entries = ['Hss', 'spline']
    # entries = ['H']
    # entries = ['H', 'S']
    entries = ['spline']

    import pickle as pkl
    from consts import SYM2ATOM, ANGSTROM2BOHR
    from sklearn.metrics import mean_absolute_error
    # cv_path = '/home/francishe/Documents/DFTBrepulsive/Au_cv/Au_cv (cvxopt, nknots=50, deg=3, rmax=au_short~au_short, ptype=convex)/Au_cv_rmax.pkl'
    cv_path = '/home/francishe/Documents/DFTBrepulsive/a1K_full_xydata.pkl'
    with open(cv_path, 'rb') as f:
        atoms = skf_path.rsplit('/', 1)[-1].split('.')[0].split('-')
        _Zs = tuple(sorted([SYM2ATOM[atoms[0]], SYM2ATOM[atoms[-1]]]))
        cv_res = pkl.load(f)['all_xydata']['sparse_xydata'][0][0][0][_Zs]
        spl_grid = cv_res[0] * ANGSTROM2BOHR
        y_cv = cv_res[1]

    #: Size of the combined plot
    figsize = (32, 20)

    #: Plot ranges
    HS_ylim = (-0.5, 0.5)
    # Spl_ylim = (-10, 200)
    Spl_ylim = (-1.5, 1.5)

    skf = SKF(skf_path, spl_grid=spl_grid)
    skf_res = skf.plot(entries, figsize, HS_ylim, Spl_ylim)
    y_skf = np.array(skf_res['Spline']['Spline'])

    print(mean_absolute_error(y_cv, y_skf) * HARTREE)
