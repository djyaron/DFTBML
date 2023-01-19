import os
import pickle as pkl
from deprecated.cv_v1 import GridSearchCV, grid_convert
from util import path_check, Timer

dataset_path = "/home/francishe/Downloads/Datasets/Au_energy_clean.h5"
gammas_path = "/home/francishe/Downloads/Datasets/gammas_50_5_full_aec.h5"
cv_root = "/home/francishe/Downloads/Au_cv"

assert os.path.exists(dataset_path), "dataset_path not exists"
assert os.path.exists(gammas_path), "gammas_path not exists"

# target_type = ['pr']
target_type = ['wd', 'pe']

# select one hyperparameter as the variable when scanning the parameter grid
var_param = 'nknots'
# Use argument parser to take arguments from command line
# parser = ArgumentParser()
# parser.add_argument("var_param")
# args_input = parser.parse_args()
# var_param = str(args_input.var_param)

# Vary one parameter each time and fix the others according to default_params
full_grid = {'nknots': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
             'deg': [3],
             'bconds': ["vanishing"],
             'rmax': [2.50, 3.00, 3.50, 4.00, 4.50, 5.00],
             'ptype': ["monotonic"],
             'p': [None]}

default_params = {'nknots': [25],
                  'deg': [3],
                  'bconds': ["vanishing"],
                  'rmax': ["au_medium"],
                  'ptype': ['monoconv'],
                  'p': [None]}

solver = 'cvxopt'

cv = 5
reverse = False  # reverse cv: smaller training set than test test

shift = True
load = True

show = False

with Timer("GridSearchCV"):
    # Create workdir
    workdir = f"Au_cv ({solver}, "
    for param, param_val in default_params.items():
        if param == var_param:
            workdir += f"{var_param}={full_grid[var_param][0]}~{full_grid[var_param][-1]}, "
        else:
            if param in ('p', 'bconds'):
                continue
            workdir += f"{param}={param_val[0]}, "
    workdir = workdir.rsplit(",", 1)[0] + ')/'
    cv_root = os.path.join(cv_root, workdir)
    path_check(cv_root)

    # Create parameter grid
    search_grid = default_params.copy()
    search_grid[var_param] = full_grid[var_param].copy( )
    param_grid = grid_convert(search_grid, gammas_path)

    # Create penalty parameter grid
    pen_param_grid = [{ptype: search_grid['p'].copy()} for ptype in search_grid['ptype']]

    # Create GridSearchCV model
    gs = GridSearchCV(dataset_path, gammas_path, target_type,
                      param_grid=param_grid, pen_param_grid=pen_param_grid,
                      cv=cv, reverse=reverse, shift=shift, load=load,
                      solver=solver, show=show)

    # Calculation
    gs.get_all_models()
    res = {"search_grid": search_grid,  # parameter grid specified by user, including nknots and rmax
           "param_grid": gs.param_grid,  # parameter grid with nknots and rmax converted to xknots
           "pen_param_grid": gs.pen_param_grid,  # penalty parameter grid used by the searcher
           "dense_coefs_cv": gs.dense_coefs_cv,
           "sparse_coefs_cv": gs.sparse_coefs_cv,
           "all_losses": gs.get_all_losses(),
           "all_xycurv": gs.get_all_xydata(nvals=500, ider=2),
           "all_xyslope": gs.get_all_xydata(nvals=500, ider=1),
           "all_xydata": gs.get_all_xydata(nvals=500, ider=0),
           "xydata_std": gs.get_xydata_std()}

    res_name = f"Au_cv_{var_param}.pkl"
    res_path = os.path.join(cv_root, res_name)
    with open(res_path, "wb") as f:
        pkl.dump(res, f)
