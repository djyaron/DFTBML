from typing import Iterable, List

import numpy as np
from matplotlib import pyplot as plt

from .tfspline import spline_linear_model, spline_vals


# TODO: integrate tfspline.py
class BSpline:
    def __init__(self, xknots: np.ndarray, bconds: Iterable,
                 deg: int, maxder: int, coef: np.ndarray = None) -> None:
        self.xknots = xknots
        self.bconds = bconds
        self.deg = deg
        self.maxder = maxder
        self.coef = coef

    # TODO: allow users to specify the desired order of derivatives
    def __call__(self, grid: np.ndarray, bases_only: bool = False) -> List[np.ndarray]:
        # return the derivatives of the splines (up to order of self.max_der)
        _spl_dict = spline_linear_model(xknots=self.xknots, xeval=grid, xyfit=None,
                                                 bconds=self.bconds, max_der=self.maxder, deg=self.deg)
        # bases_only: returns the derivatives of the bases
        if bases_only:
            return _spl_dict['X']
        else:
            return [spline_vals(_spl_dict, ider=i, coefs=self.coef) for i in range(self.maxder)]

    def fit(self, x: np.ndarray, y: np.ndarray, trim: bool = True) -> np.ndarray:
        r"""Fit splines to given spline grid and values

        Args:
            x: np.ndarray
            y: np.ndarray
            trim: bool

        Returns:

        """
        if trim:
            mask = ~np.isnan(y)  # y becomes nan when x is out of range of SKF
            xyfit = (x[mask], y[mask])
        else:
            xyfit = (x, y)
        _spl_dict = spline_linear_model(xknots=self.xknots, xeval=None, xyfit=xyfit,
                                                 bconds=self.bconds, max_der=self.maxder, deg=self.deg)
        self.coef = _spl_dict['coefs']
        return self.coef

    def plot(self, grid: np.ndarray, der: int = 0) -> None:
        _spl_dict = spline_linear_model(xknots=self.xknots, xeval=grid, xyfit=None,
                                                 bconds=self.bconds, max_der=self.maxder, deg=self.deg)
        x = grid
        y = spline_vals(_spl_dict, ider=der, coefs=self.coef)
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    from consts import BCONDS

    xknots = np.linspace(1.0, 10.0, 25)
    bconds = BCONDS['vanishing']
    deg = 3
    max_der = 3
    coef = np.arange(24)
    coef[12] = 1
    grid = np.linspace(2.0, 8.0, 500)

    spl = BSpline(xknots, bconds, deg, max_der, coef)
    spl.plot(grid, der=0)
    spl.plot(grid, der=1)
    spl.plot(grid, der=2)