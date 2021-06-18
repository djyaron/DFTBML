from model_v0 import *
from deprecated.solver_v1 import Solver
from fold import *
from util import Timer

dataset_path = "/home/francishe/Downloads/ANI-1ccx_clean_shifted.h5"
gammas_path = "/home/francishe/Downloads/gammas_50_5_extend.h5"
cv_root = "/home/francishe/Downloads/cv_inflect/"

target_type = ['cc', 'pe']
folds_cv = get_folds_cv(dataset_path, 1, (1, 1), True)

Zs = list(CUTOFFS_EXT.keys())
param_dict = {"xknots": {Z: np.linspace(r[0], r[-1], 25) for Z, r in CUTOFFS_EXT.items()},
              "deg": {Z: 3 for Z in Zs},
              "bconds": {Z: "vanishing" for Z in Zs}}
map_grid = 500
pen_grid = 500
max_iter = 3
show = False

solver = 'cvxopt'

train_fold, test_fold = folds_cv[0]
train_targets = get_targets_from_dataset(target_type, train_fold, dataset_path)

dense_model_Z, dense_loss_func = create_dense(target_type, train_fold, dataset_path, gammas_path)
dense_coefs = dense_loss_func.solve()

sparse_models_only = {Z: SplineModel({k: v[Z] for k, v in param_dict.items()}) for Z in Zs}
map_models, sparse_model_Z = map_linear_models_Z(sparse_models_only, dense_model_Z, map_grid)
sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)

pp_start = {"inflect": {Z: 0 for Z in Zs}}
best_pp = pp_start.copy()
best_c = None
best_loss = 1e99

# ============ Iteration start =============
for i_loop in range(max_iter):
    with Timer(f"Loop {i_loop}"):
        for Z in Zs:
            for i in range(0, pen_grid, 25):
                pp = best_pp.copy()
                pp['inflect'][Z] = i
                s = Solver(sparse_model_Z, sparse_loss_func, pp, pen_grid, show)
                sparse_c = s.solve(solver)

                sparse_train_preds = get_predictions_from_sparse(sparse_c, map_models, train_fold, gammas_path)
                sparse_train_loss = compare_target_pred(train_targets, sparse_train_preds)[0]['mae']

                if sparse_train_loss < best_loss:
                    best_pp = pp.copy()
                    best_c = sparse_c.copy()
                    best_loss = sparse_train_loss
        print(f"Loop {i_loop}: best loss {best_loss * HARTREE:.3f} kcal/mol")
