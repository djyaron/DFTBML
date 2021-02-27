# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:04:22 2021

@author: Frank
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:03:27 2021

@author: Frank

Does a cross-validation_approach on pre-defined folds

Alternative implementation that uses the fold classes defined in dftbrep_fold.py
"""
from dftb_layer_splines_4 import *
import auorg_1_1
import trainedskf
from skfwriter import main
from dftb import ANGSTROM2BOHR
from model_ranges import plot_skf_values
from predictiongen import PredictionGen
from functools import reduce
from dftbrep_fold import get_folds_cv_limited, extract_data_for_molecs
import os, os.path

#%% Top level variable declaration
'''
If loading data from h5 files, make sure to note the allowed_Zs and heavy_atoms of the dataset and
set them accordingly!
'''
allowed_Zs = [1,6,7,8]
heavy_atoms = [1,2,3,4,5]
dataset_path_str = os.path.join("data", "ANI-1ccx_clean_fullentry.h5")
#Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
max_config = 10
# target = 'dt'
target = {'Etot' : 'cc',
           'dipole' : 'wb97x_dz.dipole',
           'charges' : 'wb97x_dz.cm5_charges'}
exclude = ['O3', 'N2O1', 'H1N1O3', 'H2']
# Parameters for configuring the spline
num_knots = 50
max_val = None
num_per_batch = 10

#Parameters for cross-validation
num_folds = 5 #number of folds for cv generation
cv_mode = 'normal' #one of 'normal' cv, 'reversed' cv

#Method for eigenvalue decomposition
eig_method = 'new'

#Parameters for spline mode and degree
spline_mode = 'joined'
spline_deg = 3

#Proportion for training and validation
prop_train = 0.8
prop_valid = 0.2


#Dictionary containing the inflection point variables, update and re-use the variables
# between folds
inflection_point_var_start = dict()

training_losses = list()
validation_losses = list()
times = collections.OrderedDict()
times['init'] = time.process_time()
dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
times['dataset'] = time.process_time()
print('number of molecules retrieved', len(dataset))

config = dict()
config['opers_to_model'] = ['H', 'R', 'G', 'S'] #This actually matters now

#loss weights
losses = dict()
target_accuracy_energy = 6270 #Ha^-1
target_accuracy_dipole = 100 # debye
target_accuracy_charges = 100
target_accuracy_convex = 1000
target_accuracy_monotonic = 1000

losses['Etot'] = target_accuracy_energy
losses['dipole'] = target_accuracy_dipole 
losses['charges'] = target_accuracy_charges #Not working on charge loss just yet
losses['convex'] = target_accuracy_convex
# losses['monotonic'] = target_accuracy_monotonic

all_loss_trackers = list() #List used to store all loss trackers

#Compute or load?
loaded_data = False

#Parameters and names for loading/saving data
#Each 2 element lists holds two names. The first is for the training set, the
# second is for the validation set
molec_file_names = ["final_molec_test.h5", "final_valid_molec_test.h5"]
batch_file_names = ["final_batch_test.h5", "final_valid_batch_test.h5"]

dftblst_names = ["training_dftblsts.p", "validation_dftblsts.p"]
reference_data_names = ["reference_data1.p", "reference_data2.p"]

ragged_dipole = True #Whether or not dipole matrices are ragged
run_check = True #Whether or not to run check between loaded and reference data

#Training scheme
# If this flag is set to true, the dataset will be changed such that you 
# train on up to lower_limit heavy atoms and test on the rest

# If test_set is set to 'pure', then the test set will only have molecules with
# more than lower_limit heavy atoms; otherwise, test set will have a blend of 
# molecules between those with up to lower_limit heavy atoms and those with more

# impure_ratio indicates what fraction of the molecules found with up to lower_limit
# heavy atoms should be added to the test set if the test_set is not 'pure'
transfer_training = False
transfer_train_params = {
    'test_set' : 'pure',
    'impure_ratio' : 0.2,
    'lower_limit' : 4
    }

# Flag indicates whether or not to fit to the total energy per molecule or the 
# total energy as a function of the number of heavy atoms. 
train_ener_per_heavy = True

# Debug flag. If set to true, get_feeds() for the loss models adds data based on
# dftb results rather than from ANI-1
# Note that for total energy, debug mode gives total energy per molecule,
# NOT total energy per heavy atom!
debug = False

# debug and train_ener_per_heavy should be opposite
assert(not(debug and train_ener_per_heavy))

# Flag indicating whether or not to include the dipole in backprop
include_dipole_backprop = True

#Constants for writing out skf files
atom_nums = {
    6 : 'C',
    1 : 'H',
    8 : 'O',
    7 : 'N',
    79 : 'Au'
    }

atom_masses = {
    6 : 12.01,
    1 : 1.008,
    8 : 15.999,
    7 : 14.007,
    79 : 196.967
    }
ref_direct = 'auorg-1-1'

#%% Grab the dataset and folds
def energy_correction(molec: Dict) -> None:
    r"""Performs in-place total energy correction for the given molecule by dividing Etot/nheavy
    
    Arguments:
        molec (Dict): The dictionary in need of correction
    
    Returns:
        None
    """
    zcount = collections.Counter(molec['atomic_numbers'])
    ztypes = list(zcount.keys())
    heavy_counts = [zcount[x] for x in ztypes if x > 1]
    num_heavy = sum(heavy_counts)
    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy

# Variables to hold on to all_models, loss_tracker, etc. between folds; only write out
# skf files at the very end
established_models = None #Holder for all_models
established_variables = None #Holder for model_variables
established_range_dict = None #Holder for model_range_dict


#'cc' reference energies sorted in terms of atomic numbers H, C, N, O
# Will likely change with the addition of a constant
reference_energy_start = [-2.30475824e-01, -3.63327215e+01, -5.23253002e+01, -7.18450781e+01,
  1.27026973e-03] #Computed through least squares method for 'cc' target against 'dt'

folds_cv = get_folds_cv_limited(allowed_Zs, heavy_atoms, dataset_path_str, num_folds, max_config, 
                                exclude = exclude, shuffle = (1, 1), reverse = False if cv_mode == 'normal' else True)
        
for ind, fold in enumerate(folds_cv):
    print(f"Performing train and validation on fold {ind}")
    print("Reinitialize the parameter dictionary")
    #Initialize the parameter dictionary
    par_dict = auorg_1_1.ParDict() 
    print(par_dict.keys())
    print("Getting validation, training molecules")
    training_molecs, validation_molecs = extract_data_for_molecs(fold, target, dataset_path_str)
    assert(len(training_molecs) + len(validation_molecs) == len(dataset))
    print(f"{len(training_molecs)} training molecules, {len(validation_molecs)} validation molecules")
    if train_ener_per_heavy:
        for molec in training_molecs:
            energy_correction(molec)
        for molec in validation_molecs:
            energy_correction(molec)
    random.shuffle(training_molecs)
    random.shuffle(validation_molecs)
    
    print("Getting training graphs")
    training_feeds, training_dftblsts, training_batches = graph_generation(training_molecs, config, allowed_Zs, par_dict, num_per_batch)
    print("Getting validation graphs")
    validation_feeds, validation_dftblsts, validation_batches = graph_generation(validation_molecs, config, allowed_Zs, par_dict, num_per_batch)
    
    print("Initializing models")
    all_models, model_variables, loss_tracker, all_losses, model_range_dict = model_loss_initialization(training_feeds, validation_feeds,
                                                                               allowed_Zs, losses, ref_ener_start = reference_energy_start)
    if ind > 0:
        # If not the first fold, carry over the models and variables that are already initialized
        # Can do this since the complete set of molecules in the dataset should remain constant
        # The loss_tracker and all_losses can continue to be initialized from the feeds
        print("Loading established models, variables, and ranges")
        all_models = established_models
        model_variables = established_variables
        model_range_dict = established_range_dict
        print(model_variables[Model(oper='S', Zs=(1, 6), orb='sp')])
        print(model_variables[Model(oper='S', Zs=(1, 6), orb='sp_inflect')])
        print(model_variables['Eref'])
        assert(model_variables[Model(oper='S', Zs=(1, 6), orb='sp')] is established_variables[Model(oper='S', Zs=(1, 6), orb='sp')])
        assert(model_variables['Eref'] is established_variables['Eref'])
        assert(established_models is all_models)
        assert(established_variables is model_variables)
        assert(established_range_dict is model_range_dict)
    
    print("Model range correction")
    cutoff_dict = {
    (6, 6) : 1.04,
    (1, 6) : 0.602,
    (7, 7) : 0.986,
    (6, 7) : 0.948,
    (1, 7) : 0.573,
    (1, 8) : 0.599,
    (6, 8) : 1.005,
    (7, 8) : 0.933,
    (8, 8) : 1.062
    }

    new_dict = dict()
    
    for mod, dist_range in model_range_dict.items():
        xlow, xhigh = dist_range
        Zs, Zs_rev = mod.Zs, (mod.Zs[1], mod.Zs[0])
        if Zs in cutoff_dict:
            xlow = cutoff_dict[Zs]
        elif Zs_rev in cutoff_dict:
            xlow = cutoff_dict[Zs_rev]
        new_dict[mod] = (xlow, xhigh)
    
    model_range_dict = new_dict
    
    # print("Inflection variable correction")
    
    print("Generating training feeds")
    feed_generation(training_feeds, training_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, spline_mode, spline_deg, debug, loaded_data)
    print("Generating validation feeds")
    feed_generation(validation_feeds, validation_batches, all_losses, all_models, model_variables, model_range_dict, par_dict, spline_mode, spline_deg, debug, loaded_data)

    print("Performing type conversion")
    total_type_conversion(training_feeds, validation_feeds, ignore_keys = ['glabels', 'basis_sizes', 'charges', 'dipole_mat'])
    
    print(f"inflect mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S' and 'inflect' in mod.orb]}")
    print(f"s_mods: {[mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S']}")
    print(f"len of s_mods: {len([mod for mod in model_variables if mod != 'Eref' and mod.oper == 'S'])}")
    print(f"len of s_mods in all_models: {len([mod for mod in all_models if mod != 'Eref' and mod.oper == 'S'])}")
    print("losses")
    print(losses)
    
    # if inflection_point_var_start: #non-empty dictionaries evaluate to True
    #     print("Correcting inflection point variables")
    #     for model_spec in all_models:
    #         if model_spec in inflection_point_var_start:
    #             print(f"Correcting inflection point for {model_spec}")
    #             all_models[model_spec].set_inflection_pt(inflection_point_var_start[model_spec])
    #             oper, Zs, orb = model_spec
    #             orb += '_inflect'
    #             inflect_mod = Model(oper, Zs, orb)
    #             model_variables[inflect_mod] = all_models[model_spec].get_inflection_pt()

#%% Training loop
    dftblayer = DFTB_Layer(device = None, dtype = torch.double, eig_method = eig_method)
    learning_rate = 1.0e-5
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad=True)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01) 
    
    #Instantiate the loss layer here
    
    times_per_epoch = list()
    
    #First charge update
    print("running training set charge update")
    for j in range(len(training_feeds)):
        # Charge update for training_feeds
        feed = training_feeds[j]
        dftb_list = training_dftblsts[j]
        op_dict = assemble_ops_for_charges(feed, all_models)
        try:
            update_charges(feed, op_dict, dftb_list)
        except Exception as e:
            print(e)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for")
            print(result_lst)
    print("training charge update done, doing validation set")
    for k in range(len(validation_feeds)):
        # Charge update for validation_feeds
        feed = validation_feeds[k]
        dftb_list = validation_dftblsts[k]
        op_dict = assemble_ops_for_charges(feed, all_models)
        try:
            update_charges(feed, op_dict, dftb_list)
        except Exception as e:
            print(e)
            glabels = feed['glabels']
            basis_sizes = feed['basis_sizes']
            result_lst = []
            for bsize in basis_sizes:
                result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
            print("Charge update failed for")
            print(result_lst)
    print(f"charge updates done for start")
    
    nepochs = 150
    for i in range(nepochs):
        #Initialize epoch timer
        start = time.time()
        
        #Validation routine
        #Comment out, testing new loss
        validation_loss = 0
        for elem in validation_feeds:
            with torch.no_grad():
                output = dftblayer(elem, all_models)
                # loss = loss_layer.get_loss(output, elem)
                tot_loss = 0
                for loss in all_losses:
                    if loss == 'Etot':
                        if train_ener_per_heavy:
                            val = losses[loss] * all_losses[loss].get_value(output, elem, True)
                        else:
                            val = losses[loss] * all_losses[loss].get_value(output, elem, False)
                        tot_loss += val
                        loss_tracker[loss][2] += val.item()
                    elif loss == 'dipole':
                        val = losses[loss] * all_losses[loss].get_value(output, elem)
                        loss_tracker[loss][2] += val.item()
                        if include_dipole_backprop:
                            tot_loss += val
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, elem)
                        tot_loss += val 
                        loss_tracker[loss][2] += val.item()
                validation_loss += tot_loss.item()
        print("Validation loss:",i, (validation_loss/len(validation_feeds)))
        validation_losses.append((validation_loss/len(validation_feeds)))
        
        for loss in all_losses:
            loss_tracker[loss][0].append(loss_tracker[loss][2] / len(validation_feeds))
            #Reset the loss tracker after being done with all feeds
            loss_tracker[loss][2] = 0
    
        #Shuffle the validation data
        # random.shuffle(validation_feeds)
        temp = list(zip(validation_feeds, validation_dftblsts))
        random.shuffle(temp)
        validation_feeds, validation_dftblsts = zip(*temp)
        validation_feeds, validation_dftblsts = list(validation_feeds), list(validation_dftblsts)
        
        #Training routine
        epoch_loss = 0.0
        for feed in training_feeds:
            optimizer.zero_grad()
            output = dftblayer(feed, all_models)
            #Comment out, testing new loss
            # loss = loss_layer.get_loss(output, feed) #Loss still in units of Ha^2 ?
            tot_loss = 0
            for loss in all_losses:
                if loss == 'Etot':
                    if train_ener_per_heavy:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, True)
                    else:
                        val = losses[loss] * all_losses[loss].get_value(output, feed, False)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
                elif loss == 'dipole':
                    val = losses[loss] * all_losses[loss].get_value(output, feed)
                    loss_tracker[loss][2] += val.item()
                    if include_dipole_backprop:
                        tot_loss += val
                else:
                    val = losses[loss] * all_losses[loss].get_value(output, feed)
                    tot_loss += val
                    loss_tracker[loss][2] += val.item()
    
            epoch_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
        scheduler.step(epoch_loss) #Step on the epoch loss
        
        #Perform shuffling while keeping order b/w dftblsts and feeds consistent
        temp = list(zip(training_feeds, training_dftblsts))
        random.shuffle(temp)
        training_feeds, training_dftblsts = zip(*temp)
        training_feeds, training_dftblsts = list(training_feeds), list(training_dftblsts)
        
        print(i, (epoch_loss/len(training_feeds)))
        training_losses.append((epoch_loss/len(training_feeds)))
        
        for loss in all_losses:
            loss_tracker[loss][1].append(loss_tracker[loss][2] / len(training_feeds))
            loss_tracker[loss][2] = 0
        
        # Update charges every 10 epochs
        # Do the charge update for the validation and the training sets
        if (i % 10 == 0):
            print("running training set charge update")
            for j in range(len(training_feeds)):
                # Charge update for training_feeds
                feed = training_feeds[j]
                dftb_list = training_dftblsts[j]
                op_dict = assemble_ops_for_charges(feed, all_models)
                try:
                    update_charges(feed, op_dict, dftb_list)
                except Exception as e:
                    print(e)
                    glabels = feed['glabels']
                    basis_sizes = feed['basis_sizes']
                    result_lst = []
                    for bsize in basis_sizes:
                        result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
                    print("Charge update failed for")
                    print(result_lst)
            print("training charge update done, doing validation set")
            for k in range(len(validation_feeds)):
                # Charge update for validation_feeds
                feed = validation_feeds[k]
                dftb_list = validation_dftblsts[k]
                op_dict = assemble_ops_for_charges(feed, all_models)
                try:
                    update_charges(feed, op_dict, dftb_list)
                except Exception as e:
                    print(e)
                    glabels = feed['glabels']
                    basis_sizes = feed['basis_sizes']
                    result_lst = []
                    for bsize in basis_sizes:
                        result_lst += list(zip(feed['names'][bsize], feed['iconfigs'][bsize]))
                    print("Charge update failed for")
                    print(result_lst)
            print(f"charge updates done for epoch {i}")
        #Save timing information for diagnostics
        times_per_epoch.append(time.time() - start)
    
    print(f"Finished with {nepochs} epochs")
    
    print("Saving reference energy parameters")
    reference_energy_start = list(model_variables['Eref'].detach().numpy())
    
    print("Saving loss_tracker")
    all_loss_trackers.append(loss_tracker)
    
    # print("Saving the new inflection point variable values")
    # #TODO: Save inflection point variable values and reuse between folds
    # for model_spec in all_models:
    #     if model_spec != 'Eref' and len(model_spec.Zs) == 2 and model_spec.oper != 'G':
    #         inflect_point_var = all_models[model_spec].get_inflection_pt()
    #         if inflect_point_var is not None:
    #             print(f"Saving inflection point for {model_spec}")
    #             true_val = inflect_point_var.detach().numpy() #Should be a tensor of one element
    #             assert(len(true_val) == 1)
    #             inflection_point_var_start[model_spec] = [true_val[0]]
    

    print("Saving models, variables, and ranges")
    established_models = all_models
    established_variables = model_variables
    established_range_dict = model_range_dict
    print(model_variables[Model(oper='S', Zs=(1, 6), orb='sp')])
    print(model_variables[Model(oper='S', Zs=(1, 6), orb='sp_inflect')])
    print(model_variables['Eref'])
    assert(model_variables[Model(oper='S', Zs=(1, 6), orb='sp')] is established_variables[Model(oper='S', Zs=(1, 6), orb='sp')])
    assert(model_variables['Eref'] is established_variables['Eref'])
    assert(established_models is all_models)
    assert(established_variables is model_variables)
    assert(established_range_dict is model_range_dict)
    
    if ind == num_folds - 1:
        print("Writing skf files from trained models")
        main(all_models, atom_nums, atom_masses, True, ref_direct, ext = 'newskf')
    
    print(f"Done with fold {ind}")

with open("all_loss_tracker.p", "wb") as handle:
    pickle.dump(all_loss_trackers, handle)