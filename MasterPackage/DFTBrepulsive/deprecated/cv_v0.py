from fold import *
from model_v0 import *
from util import get_sha256
from deprecated.solver_v1 import Solver
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression


class GridSearchCV:

    def __init__(self, dataset_path, gammas_path, target_type,
                 param_grid, map_grid=500,
                 pen_param_grid=None, pen_grid=100,
                 folds_cv=None, cv=5, shuffle=(1, 1), reverse=False,
                 shift=True, load=False,
                 solver='cvxopt', show=False):
        self.dataset_path = dataset_path
        self.gammas_path = gammas_path
        # check if gammas file is generated from the dataset specified
        with File(gammas_path, 'r') as gammas:
            if get_sha256(dataset_path) != gammas['_INFO']['SHA256'][()]:
                print("WARNING: gammas and dataset do not match")
        self.target_type = target_type
        self.param_grid = ParameterGrid(param_grid)
        self.map_grid = map_grid
        self.pen_param_grid = pen_param_grid if pen_param_grid is None else ParameterGrid(pen_param_grid)
        self.pen_grid = pen_grid
        self.folds_cv = get_folds_cv(dataset_path, cv, shuffle, reverse) if folds_cv is None else folds_cv
        self.shift = shift
        self.load = load
        self.solver = solver
        self.show = show
        self.dense_models_cv = None
        self.dense_loss_funcs_cv = None
        self.dense_coefs_cv = None
        self.map_models_cv = None
        self.sparse_models_cv = None
        self.sparse_loss_funcs_cv = None
        self.sparse_coefs_cv = None
        self.dense_train_losses_cv = None
        self.dense_test_losses_cv = None
        self.sparse_train_losses_cv = None
        self.sparse_test_losses_cv = None
        self.nvals = None
        self.dense_xydata_cv = None
        self.sparse_xydata_cv = None
        self.dense_xydata_std = None
        self.sparse_xydata_std = None
        self.xydata_ider = None

    def get_shifted_dataset(self):
        workdir, dataset_name = os.path.split(self.dataset_path)
        # Flatten training set
        with File(self.dataset_path, 'r') as src:
            for ifold, (train_fold, _) in enumerate(self.folds_cv):
                if self.show:
                    print(f"Shifting fold {ifold}", end='\r')
                # Get flattened training set
                train_flat = flatten_by_fold(src, train_fold, self.target_type)
                # Record atom types
                Xis = [c for c in train_flat.columns if c.isnumeric()]
                # Separate atom counts from target values and fit to linear model
                atom_count_cols = [c.isnumeric() for c in train_flat.columns]
                X = train_flat.loc[:, atom_count_cols]
                regs = {}
                for tt in self.target_type:
                    y = train_flat.loc[:, ALIAS2TARGET[tt]]
                    regs.update({tt: LinearRegression().fit(X, y)})
                # Shift dataset using the fitted linear model
                shifted_name = f"{dataset_name.rsplit('.')[0]}_{ifold}.h5"
                shifted_path = os.path.join(workdir, shifted_name)
                with File(shifted_path, 'w') as des:
                    for mol, moldata in src.items():
                        des_mol = des.create_group(mol)
                        des_mol.create_dataset('atomic_numbers', data=moldata['atomic_numbers'])
                        des_mol.create_dataset('coordinates', data=moldata['coordinates'])
                        atomic_numbers = list(des_mol['atomic_numbers'][()])
                        atom_count_mol = np.array([atomic_numbers.count(int(atom)) for atom in Xis])
                        # shift target values using pre-trained linear model
                        for tt, reg in regs.items():
                            pred = reg.predict(atom_count_mol.reshape(1, -1))
                            data = moldata[ALIAS2TARGET[tt]][()] - pred
                            des_mol.create_dataset(ALIAS2TARGET[tt], data=data)

    def get_all_models(self):
        """
          dense_models_cv: [mod1, mod2, ...], length = cv
          dense_loss_funcs_cv = [fun1, fun2, ...], length = cv
          map_models_cv = [mod1, mod2, ...], length = cv
          sparse_models_cv: [[mod11, mod12, ...] (length = len(param_grid)), ...], length = cv
          sparse_loss_funcs_cv: [[fun11, fun12, ...] (length = len(param_grid)), ...], length = cv
        """
        self.dense_models_cv = list()
        self.dense_loss_funcs_cv = list()
        self.dense_coefs_cv = list()
        self.map_models_cv = list()
        self.sparse_models_cv = list()
        self.sparse_loss_funcs_cv = list()
        self.sparse_coefs_cv = list()

        if self.shift:
            if self.load:
                try:
                    for ifold, _ in enumerate(self.folds_cv):
                        dataset_path = f"{self.dataset_path.rsplit('.')[0]}_{ifold}.h5"
                        assert os.path.exists(dataset_path)
                except AssertionError:
                    self.get_shifted_dataset()
            else:
                self.get_shifted_dataset()

        for ifold, (train_fold, test_fold) in enumerate(self.folds_cv):
            if self.show:
                print(f"Current fold: {ifold}", end='\r')

            if self.shift:
                dataset_path = f"{self.dataset_path.rsplit('.')[0]}_{ifold}.h5"
            else:
                dataset_path = self.dataset_path
            dense_model_Z, dense_loss_func = create_dense(self.target_type, train_fold,
                                                          dataset_path, self.gammas_path)
            dense_coefs = dense_loss_func.solve()

            self.dense_models_cv.append(dense_model_Z)
            self.dense_loss_funcs_cv.append(dense_loss_func)
            self.dense_coefs_cv.append(dense_coefs)

            Zs = dense_model_Z.Zs()
            sparse_models_fold = list()
            sparse_loss_funcs_fold = list()
            sparse_coefs_fold = list()
            map_models_fold = list()

            for param_dict in self.param_grid:
                sparse_models_only = {Z: SplineModel({k: v[Z] for k, v in param_dict.items()}) for Z in Zs}
                map_models, sparse_model_Z = map_linear_models_Z(sparse_models_only, dense_model_Z, self.map_grid)
                sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)
                if self.pen_param_grid is None:
                    sparse_coefs = sparse_loss_func.solve()
                else:
                    sparse_coefs = list()
                    for pen_param in self.pen_param_grid:
                        solver = Solver(sparse_model_Z, sparse_loss_func, pen_param, self.pen_grid, self.show)
                        sparse_coefs.append(solver.solve(self.solver))

                map_models_fold.append(map_models)
                sparse_models_fold.append(sparse_model_Z)
                sparse_loss_funcs_fold.append(sparse_loss_func)
                sparse_coefs_fold.append(sparse_coefs)

            self.sparse_models_cv.append(sparse_models_fold)
            self.sparse_loss_funcs_cv.append(sparse_models_fold)
            self.sparse_coefs_cv.append(sparse_coefs_fold)
            self.map_models_cv.append(map_models_fold)

        if self.show:
            print("Models generated.", end='\r')

    def get_all_losses(self):
        # results are in hartrees
        self.dense_train_losses_cv = list()
        self.dense_test_losses_cv = list()
        self.sparse_train_losses_cv = list()
        self.sparse_test_losses_cv = list()

        for ifold, (train_fold, test_fold) in enumerate(self.folds_cv):
            if self.show:
                print(f"Current fold: {ifold}", end='\r')

            if self.shift:
                dataset_path = f"{self.dataset_path.rsplit('.')[0]}_{ifold}.h5"
            else:
                dataset_path = self.dataset_path

            train_targets = get_targets_from_dataset(self.target_type, train_fold, dataset_path)
            test_targets = get_targets_from_dataset(self.target_type, test_fold, dataset_path)
            # dense
            dense_train_preds = get_predictions_from_dense(self.dense_coefs_cv[ifold], train_fold, self.gammas_path)
            dense_test_preds = get_predictions_from_dense(self.dense_coefs_cv[ifold], test_fold, self.gammas_path)
            self.dense_train_losses_cv.append(compare_target_pred(train_targets, dense_train_preds))
            self.dense_test_losses_cv.append(compare_target_pred(test_targets, dense_test_preds))
            # sparse
            sparse_train_losses_fold = list()
            sparse_test_losses_fold = list()
            for j in range(len(self.param_grid)):
                if self.pen_param_grid is None:
                    sparse_train_preds = get_predictions_from_sparse(self.sparse_coefs_cv[ifold][j],
                                                                     self.map_models_cv[ifold][j],
                                                                     train_fold, self.gammas_path)
                    sparse_test_preds = get_predictions_from_sparse(self.sparse_coefs_cv[ifold][j],
                                                                    self.map_models_cv[ifold][j],
                                                                    test_fold, self.gammas_path)
                    sparse_train_losses_fold.append(compare_target_pred(train_targets, sparse_train_preds))
                    sparse_test_losses_fold.append(compare_target_pred(test_targets, sparse_test_preds))
                else:
                    sparse_train_preds = list()
                    sparse_test_preds = list()
                    for k in range(len(self.pen_param_grid)):
                        sparse_train_preds.append(get_predictions_from_sparse(self.sparse_coefs_cv[ifold][j][k],
                                                                              self.map_models_cv[ifold][j],
                                                                              train_fold, self.gammas_path))
                        sparse_test_preds.append(get_predictions_from_sparse(self.sparse_coefs_cv[ifold][j][k],
                                                                             self.map_models_cv[ifold][j],
                                                                             test_fold, self.gammas_path))
                    sparse_train_losses_fold.append(
                        [compare_target_pred(train_targets, preds) for preds in sparse_train_preds])
                    sparse_test_losses_fold.append(
                        [compare_target_pred(test_targets, preds) for preds in sparse_test_preds])

            self.sparse_train_losses_cv.append(sparse_train_losses_fold.copy())
            self.sparse_test_losses_cv.append(sparse_test_losses_fold.copy())

        if self.show:
            print("Losses generated.", end='\r')

        res = {"dense_train": self.dense_train_losses_cv.copy(),
               "dense_test": self.dense_test_losses_cv.copy(),
               "sparse_train": self.sparse_train_losses_cv.copy(),
               "sparse_test": self.sparse_test_losses_cv.copy()}
        return res

    def get_all_xydata(self, nvals=500, ider=0):
        # results are in (Angstrom, hartree)
        self.nvals = nvals
        self.xydata_ider = ider
        self.dense_xydata_cv = list()
        self.sparse_xydata_cv = list()

        for ifold, _ in enumerate(self.folds_cv):
            if self.show:
                print(f"Current fold: {ifold}", end='\r')
            # dense
            dense_xydata = get_xydata_Z(self.dense_models_cv[ifold], self.dense_coefs_cv[ifold],
                                        self.nvals, self.xydata_ider)
            self.dense_xydata_cv.append(dense_xydata)
            # sparse
            sparse_xydata_fold = list()
            for j in range(len(self.param_grid)):
                if self.pen_param_grid is None:
                    sparse_xydata = get_xydata_Z(self.sparse_models_cv[ifold][j], self.sparse_coefs_cv[ifold][j],
                                                 self.nvals, self.xydata_ider)
                else:
                    sparse_xydata = list()
                    for k in range(len(self.pen_param_grid)):
                        sparse_xydata.append(
                            get_xydata_Z(self.sparse_models_cv[ifold][j], self.sparse_coefs_cv[ifold][j][k],
                                         self.nvals, self.xydata_ider))
                sparse_xydata_fold.append(sparse_xydata)
            self.sparse_xydata_cv.append(sparse_xydata_fold.copy())

        if self.show:
            print("Xydata generated.", end='\r')

        res = {"dense_xydata": self.dense_xydata_cv,
               "sparse_xydata": self.sparse_xydata_cv,
               "ider": self.xydata_ider}
        return res

    def get_xydata_std(self, ider=0):

        if ider != self.xydata_ider:
            self.get_all_xydata(self.nvals, ider)

        def std_metric(ys):
            y_std = np.mean(np.std(ys, axis=0))
            return y_std

        # results are in hartrees
        Zs = list(self.dense_xydata_cv[0].keys())
        # dense
        dense_y_Z = {Z: [] for Z in Zs}
        for xydata in self.dense_xydata_cv:
            for Z, (x, y) in xydata.items():
                dense_y_Z[Z].append(y)

        self.dense_xydata_std = dict().fromkeys(Zs)
        for Z, ys in dense_y_Z.items():
            y_std = std_metric(ys)
            self.dense_xydata_std[Z] = y_std

        # sparse
        sparse_y_Z = list()  # len(param_grid)
        for j in range(len(self.param_grid)):
            if self.pen_param_grid is None:
                sparse_y_Z_param = {Z: [] for Z in Zs}
                for i in range(len(self.folds_cv)):
                    xydata = self.sparse_xydata_cv[i][j]
                    for Z, (x, y) in xydata.items():
                        sparse_y_Z_param[Z].append(y)
                sparse_y_Z.append(sparse_y_Z_param.copy())
            else:  # first layer: len(param_grid), second layer: len(pen_param_grid)
                sparse_y_Z_param = list()
                for k in range(len(self.pen_param_grid)):
                    sparse_y_Z_param_pen = {Z: [] for Z in Zs}
                    for i in range(len(self.folds_cv)):
                        xydata = self.sparse_xydata_cv[i][j][k]
                        for Z, (x, y) in xydata.items():
                            sparse_y_Z_param_pen[Z].append(y)
                    sparse_y_Z_param.append(sparse_y_Z_param_pen)
                sparse_y_Z.append(sparse_y_Z_param.copy())

        self.sparse_xydata_std = list()
        for sparse_y_Z_param in sparse_y_Z:
            if self.pen_param_grid is None:
                sparse_xydata_std_param = dict().fromkeys(Zs)
                for Z, ys in sparse_y_Z_param.items():
                    y_std = std_metric(ys)
                    sparse_xydata_std_param[Z] = y_std
                self.sparse_xydata_std.append(sparse_xydata_std_param.copy())
            else:
                sparse_xydata_std_param = list()
                for k in range(len(self.pen_param_grid)):
                    sparse_xydata_std_param_pen = dict().fromkeys(Zs)
                    for Z, ys in sparse_y_Z_param[k].items():
                        y_std = std_metric(ys)
                        sparse_xydata_std_param_pen[Z] = y_std
                    sparse_xydata_std_param.append(sparse_xydata_std_param_pen.copy())
                self.sparse_xydata_std.append(sparse_xydata_std_param.copy())

        if self.show:
            print("Xydata_std generated.", end='\r')

        res = {"dense_xydata_std": self.dense_xydata_std,
               "sparse_xydata_std": self.sparse_xydata_std,
               "ider": self.xydata_ider}
        return res


def grid_convert(search_grid, gammas_path):
    with File(gammas_path, 'r') as gammas:
        Zs = [tuple(Z) for Z in gammas['_INFO']['Zs']]
        cutoffs_original = {Z: tuple(c) for Z, c in zip(Zs, gammas['_INFO']['R_cutoffs'])}

    deg_grid = [{Z: d for Z in Zs for d in search_grid['deg']}]
    bconds_grid = [{Z: b for Z in Zs for b in search_grid['bconds']}]

    cutoffs_grid = list()
    for rmax in search_grid['rmax']:
        if isinstance(rmax, (int, float)):
            cutoffs = {tuple(Z): (cutoff_Z[0], rmax) for Z, cutoff_Z in cutoffs_original.items()}
            cutoffs_grid.append(cutoffs)
        elif isinstance(rmax, str):
            try:
                cutoffs_grid.append(CUTOFFS[rmax])
            except KeyError:
                raise NotImplementedError
        else:
            raise NotImplementedError

    xknots_grid = list()
    # The order of the two loops doesn't matter, since there's at most only one var_param
    # i.e. cutoffs_grid and search_grid['nknots'] will not have length > 1 at the same time
    for cutoffs in cutoffs_grid:
        for n in search_grid['nknots']:
            xknots = {Z: np.linspace(rmin, rmax, n) for Z, (rmin, rmax) in cutoffs.items()}
            xknots_grid.append(xknots)

    param_grid = {'xknots': xknots_grid,
                  'deg': deg_grid,
                  'bconds': bconds_grid}

    return param_grid


"""
=========================================== APPENDIX ===========================================

------------------------------------------------------------------------------------------------
Structure of dense_coefs_cv

[np.array, np.array, ...]
 fold_1,   fold_2,   ...
------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------
Structure of sparse_coefs_cv

[[[np.array,  np.array, ...], [np.array,   np.array, ...], ...], (fold_1)
   param_1,                    param_2,                    ...
   penalty_1, penalty_2, ...   penalty_1, penalty_2, ...
[[np.array,  np.array,...], [np.array,   np.array,    ...], ...], (fold_2)
...]
------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------
Structure of all_losses

{"dense_train": [err_dict, err_dict, ...]
                 fold_1,   fold_2,   ...
"dense_test": same as dense_train
"sparse_train: [[err_dict,  err_dict, ...], [err_dict, ...], ...], (fold_1)
                 param_1,                    param_2, ...
                 penalty_1, penalty_2,...    penalty_1, ...
                [err_dict, err_dict,  ...], [err_dict, ...], ...], (fold_2)
               ...]
"sparse_test: same as sparse_train}
------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------
Structure of all_xydata

{"dense_xydata": [xydata, xydata, ...]
                  fold_1, fold_2, ...
"sparse_xydata": [[[xydata,    xydata,    ...], [xydata,    ...], ...], (fold_1)
                    param_1,                     param_2,          ...
                    penalty_1, penalty_2, ...    penalty_1, ...
                  [[xydata, xydata, ...],       [xydata, ...],    ...], (fold_2)
                 ...]}
------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------
Structure of xydata_std

{"dense_xydata_std": std_dict, i.e. {Z_1: std_1, Z_2: std_2, ...}
"sparse_xydata_std": [[std_dict,  std_dict,  ...], [std_dict,  ...], ...]}
                       param_1,                     param_2,          ...
                       penalty_1, penalty_2, ...    penalty_1, ...
------------------------------------------------------------------------------------------------

"""
