import os
import pickle as pkl
from cv_v1 import GridSearchCV, grid_convert
from util import path_check, Timer

# dataset_path = "/home/francishe/Downloads/Datasets/ANI-1ccx_energy_clean.h5"
# gammas_path = "/home/francishe/Downloads/Datasets/gammas_50_5_full_cec.h5"
# cv_root = "/home/francishe/Downloads/ANI_cv"
dataset_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/ANI-1ccx_clean_fullentry.h5'
gammas_path = "/home/francishe/Documents/DFTBrepulsive/Datasets/gammas_50_5_full_cf.h5"
cv_root = "/home/francishe/Documents/DFTBrepulsive/ANI_cv/"

# target_type = ['pr']
target_type = ['cc', 'pe']

# select one hyperparameter as the variable when scanning the parameter grid
var_param = 'rmax'

# Vary one parameter each time and fix the others according to default_params
full_grid = {'nknots': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
             'deg': [3],
             'bconds': ["vanishing"],
             'rmax': ["short"],
             'ptype': ['convex', 'monoconv', 'monotonic'],
             'p': [None]}

default_params = {'nknots': [50],
                  'deg': [3],
                  'bconds': ["vanishing"],
                  'rmax': ["short"],
                  'ptype': ['convex'],
                  'p': [None]}

solver = 'cvxopt'

# Train on the entire dataset
from fold import Fold
from h5py import File
with File(dataset_path, 'r') as dataset:
    all_data = Fold.from_dataset(dataset)
    folds_cv = [(all_data, all_data)]

cv = 5
reverse = True  # reverse cv: smaller training set than test test
shift = True
load = False
show = False

with Timer("GridSearchCV"):
    # Create workdir
    workdir = f"cv ({solver}, "
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
    search_grid[var_param] = full_grid[var_param].copy()
    param_grid = grid_convert(search_grid, gammas_path)

    # Create penalty parameter grid
    pen_param_grid = [{ptype: search_grid['p'].copy()} for ptype in search_grid['ptype']]

    # Create GridSearchCV model
    gs = GridSearchCV(dataset_path, gammas_path, target_type,
                      param_grid=param_grid, pen_param_grid=pen_param_grid,
                      map_grid=500, pen_grid=500,
                      folds_cv=folds_cv,
                      cv=cv, reverse=reverse,
                      shift=shift, load=load,
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
           "all_xydata": gs.get_all_xydata(nvals=500, ider=0)}

    res_name = f"cv_{var_param}.pkl"
    res_path = os.path.join(cv_root, res_name)
    with open(res_path, "wb") as f:
        pkl.dump(res, f)
