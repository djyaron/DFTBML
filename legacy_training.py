# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:46:40 2021

@author: fhu14

This is just to serve as a reference of how training was done in the past
    (before refactored cldriver)
"""

if __name__ == "__main__":
    '''
    If loading data from h5 files, make sure to note the allowed_Zs and heavy_atoms of the dataset and
    set them accordingly!
    '''
    allowed_Zs = [1,6,7,8]
    heavy_atoms = [1,2,3,4,5]
    #Still some problems with oxygen, molecules like HNO3 are problematic due to degeneracies
    max_config = 10
    # target = 'dt'
    target = {'Etot' : 'dt',
              'dipole' : 'wb97x_dz.dipole',
              'charges' : 'wb97x_dz.cm5_charges'}
    exclude = ['O3', 'N2O1', 'H1N1O3']
    # Parameters for configuring the spline
    num_knots = 50
    max_val = None
    num_per_batch = 10
    
    #Method for eigenvalue decomposition
    eig_method = 'new'
    
    #Proportion for training and validation
    prop_train = 0.8
    prop_valid = 0.2
    
    reference_energies = list() # Save the reference energies to see how the losses are really changing
    training_losses = list()
    validation_losses = list()
    times = collections.OrderedDict()
    times['init'] = time.process_time()
    dataset = get_ani1data(allowed_Zs, heavy_atoms, max_config, target, exclude=exclude)
    times['dataset'] = time.process_time()
    print('number of molecules retrieved', len(dataset))
    
    config = dict()
    config['opers_to_model'] = ['H', 'R', 'G'] #This actually matters now
    
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
    losses['monotonic'] = target_accuracy_monotonic
    
    #Initialize the parameter dictionary
    par_dict = ParDict()
    
    #Compute or load?
    loaded_data = False
    
    #Training scheme
    # If this flag is set to true, the dataset will be changed such that you 
    # train on up to lower_limit heavy atoms and test on the rest
    
    # If test_set is set to 'pure', then the test set will only have molecules with
    # more than lower_limit heavy atoms; otherwise, test set will have a blend of 
    # molecules between those with up to lower_limit heavy atoms and those with more
    
    # impure_ratio indicates what fraction of the molecules found with up to lower_limit
    # heavy atoms should be added to the test set if the test_set is not 'pure'
    transfer_training = False
    test_set = 'pure' #either 'pure' or 'impure'
    impure_ratio = 0.2
    lower_limit = 4
    
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
    
    #%% Degbugging h5 (Extraction and combination)
    x = time.time()
    training_feeds = total_feed_combinator.create_all_feeds("final_batch_test.h5", "final_molec_test.h5", True)
    validation_feeds = total_feed_combinator.create_all_feeds("final_valid_batch_test.h5", "final_valid_molec_test.h5", True)
    print(f"{time.time() - x}")
    compare_feeds("reference_data1.p", training_feeds)
    compare_feeds("reference_data2.p", validation_feeds)
    
    training_molec_batches = []
    validation_molec_batches = []
    
    #Need to regenerate the molecule batches for both train and validation
    # master_train_molec_dict = per_molec_h5handler.extract_molec_feeds_h5("final_molec_test.h5")
    # master_valid_molec_dict = per_molec_h5handler.extract_molec_feeds_h5("final_valid_molec_test.h5")
    
    # #Reconstitute the lists 
    # training_molec_batches = per_molec_h5handler.create_molec_batches_from_feeds_h5(master_train_molec_dict,
    #                                                                         training_feeds, ["Etot", "dipoles", "charges"])
    # validation_molec_batches = per_molec_h5handler.create_molec_batches_from_feeds_h5(master_valid_molec_dict,
    #                                                                         validation_feeds, ["Etot", "dipoles", "charges"])
    
    #Load dftb_lsts
    training_dftblsts = pickle.load(open("training_dftblsts.p", "rb"))
    validation_dftblsts = pickle.load(open("validation_dftblsts.p", "rb"))
    
    print("Check me!")
    
    #%% Dataset Sorting
    print("Running degeneracy rejection")
    degeneracy_tolerance = 1.0e-3
    bad_indices = set()
    # NOTE: uncomment this section if using torch.symeig; if using new symeig, 
    #       can leave this step out
    # for index, batch in enumerate(dataset, 0):
    #     try:
    #         feed, _ = create_graph_feed(config, batch, allowed_Zs)
    #         eorb = list(feed['eorb'].values())[0]
    #         degeneracy = np.min(np.diff(np.sort(eorb)))
    #         if degeneracy < degeneracy_tolerance:
    #             bad_indices.add(index)
    #     except:
    #         print(batch[0]['name'])
    
    cleaned_dataset = list()
    for index, item in enumerate(dataset, 0):
        if index not in bad_indices:
            cleaned_dataset.append(item[0])
    
    print('Number of total molecules after degeneracy rejection', len(cleaned_dataset))
    
    if transfer_training:
        print("Transfer training dataset")
        # Separate into molecules with up to lower_limit heavy atoms and those with
        # more
        up_to_ll, more = list(), list()
        for molec in cleaned_dataset:
            zcount = collections.Counter(molec['atomic_numbers'])
            ztypes = list(zcount.keys())
            heavy_counts = [zcount[x] for x in ztypes if x > 1]
            num_heavy = sum(heavy_counts)
            if num_heavy > lower_limit:
                if train_ener_per_heavy: 
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                more.append(molec)
            else:
                if train_ener_per_heavy:
                    molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
                up_to_ll.append(molec)
        
        # Check whether test_set should be pure
        training_molecs, validation_molecs = None, None
        if test_set == 'pure':
            random.shuffle(up_to_ll)
            training_molecs = up_to_ll
            num_valid = (int(len(up_to_ll) / prop_train)) - len(up_to_ll)
            validation_molecs = random.sample(more, num_valid)
        elif test_set == 'impure':
            indices = [i for i in range(len(up_to_ll))]
            chosen_for_blend = set(random.sample(indices, int(len(up_to_ll) * impure_ratio)))
            training_molecs, blend_temp = list(), list()
            for ind, elem in enumerate(up_to_ll, 0):
                if ind not in chosen_for_blend:
                    training_molecs.append(elem)
                else:
                    blend_temp.append(elem)
            num_valid = (int(len(training_molecs) / prop_train)) - (len(training_molecs) + len(blend_temp))
            rest_temp = random.sample(more, num_valid)
            validation_molecs = blend_temp + rest_temp
            random.shuffle(validation_molecs)
    else:
        #Shuffle the dataset before feeding into data_loader
        print("Non-transfer training dataset")
        random.shuffle(cleaned_dataset)
        
        #Sample the indices that will be used for the training dataset randomly from the shuffled data
        indices = [i for i in range(len(cleaned_dataset))]
        sampled_indices = set(random.sample(indices, int(len(cleaned_dataset) * prop_train)))
        
        #Separate into the training and validation sets
        training_molecs, validation_molecs = list(), list()
        for i in range(len(cleaned_dataset)):
            molec = cleaned_dataset[i]
            if train_ener_per_heavy:
                zcount = collections.Counter(molec['atomic_numbers'])
                ztypes = list(zcount.keys())
                heavy_counts = [zcount[x] for x in ztypes if x > 1]
                num_heavy = sum(heavy_counts)
                molec['targets']['Etot'] = molec['targets']['Etot'] / num_heavy
            if i in sampled_indices:
                training_molecs.append(molec)
            else:
                validation_molecs.append(molec)
    
    #Logging data
    total_num_molecs = len(cleaned_dataset)
    total_num_train_molecs = len(training_molecs)
    total_num_valid_molecs = len(validation_molecs)
    
    #Now run through the graph and feed generation procedures for both the training
    #   and validation molecules
    
    #NOTE: The order of the geometries in feed corresponds to the order of the 
    # geometries in batch, i.e. the glabels match the indices of batch (everything
    # is added sequentially)
    
    # Can go based on the order of the 'glabels' key in feeds, which dictates the 
    # ordering for everything as a kvp with bsize -> values for each molecule, glabels are sorted
    print(f'Number of molecules used for training: {len(training_molecs)}')
    print(f"Number of molecules used for testing: {len(validation_molecs)}")
    #%% Graph generation
    x = time.time()
    print("Making Training Graphs")
    train_dat_set = data_loader(training_molecs, batch_size = num_per_batch)
    training_feeds, training_dftblsts = list(), list()
    training_molec_batches = list()
    for index, batch in enumerate(train_dat_set):
        feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        all_bsizes = list(feed['Eelec'].keys())
        
        # Better organization for saved names and config numbers
        feed['names'] = dict()
        feed['iconfigs'] = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            all_names = [batch[x]['name'] for x in glabels]
            all_configs = [batch[x]['iconfig'] for x in glabels]
            feed['names'][bsize] = all_names
            feed['iconfigs'][bsize] = all_configs
                        
        training_feeds.append(feed)
        training_dftblsts.append(batch_dftb_lst)
        training_molec_batches.append(batch) #Save the molecules to be used later for generating feeds
    
    print("Making Validation Graphs")
    validation_dat_set = data_loader(validation_molecs, batch_size = num_per_batch)
    validation_feeds, validation_dftblsts = list(), list()
    validation_molec_batches = list()
    for index, batch in enumerate(validation_dat_set):
        feed, batch_dftb_lst = create_graph_feed(config, batch, allowed_Zs, par_dict)
        all_bsizes = list(feed['Eelec'].keys())
        
        feed['names'] = dict()
        feed['iconfigs'] = dict()
        for bsize in all_bsizes:
            glabels = feed['glabels'][bsize]
            all_names = [batch[x]['name'] for x in glabels]
            all_configs = [batch[x]['iconfig'] for x in glabels]
            feed['names'][bsize] = all_names
            feed['iconfigs'][bsize] = all_configs
    
        validation_feeds.append(feed)
        validation_dftblsts.append(batch_dftb_lst) #Save the validation dftblsts for charge updates on the validation set
        validation_molec_batches.append(batch)
    print(f"{time.time() - x}")
    #%% Model and loss initialization
    all_models = dict()
    model_variables = dict() #This is used for the optimizer later on
    
    all_models['Eref'] = Reference_energy(allowed_Zs)
    model_variables['Eref'] = all_models['Eref'].get_variables()
    
    #More nuanced construction of config dictionary
    model_range_dict = create_spline_config_dict(training_feeds + validation_feeds)
    
    #Constructing the losses using the models implemented in loss_models
    all_losses = dict()
    
    #loss_tracker to keep track of values for each 
    #Each loss maps to tuple of two lists, the first is the validation loss,the second
    # is the training loss, and the third is a temp so that average losses for validation/train 
    # can be computed
    loss_tracker = dict() 
    
    for loss in losses:
        if loss == "Etot":
            all_losses['Etot'] = TotalEnergyLoss()
            loss_tracker['Etot'] = [list(), list(), 0]
        elif loss in ["convex", "monotonic", "smooth"]:
            all_losses[loss] = FormPenaltyLoss(loss)
            loss_tracker[loss] = [list(), list(), 0]
        elif loss == "dipole":
            all_losses['dipole'] = DipoleLoss2() #Use DipoleLoss2 for dipoles computed from ESP charges!
            loss_tracker['dipole'] = [list(), list(), 0]
        elif loss == "charges":
            all_losses['charges'] = ChargeLoss()
            loss_tracker['charges'] = [list(), list(), 0]
    
    #%% Feed generation
    x = time.time()
    print('Making training feeds')
    for ibatch,feed in enumerate(training_feeds):
       for model_spec in feed['models']:
           if (model_spec not in all_models):
               mod_res, tag = get_model_value_spline_2(model_spec, model_variables, model_range_dict, par_dict)
               all_models[model_spec] = mod_res
               #all_models[model_spec] = get_model_dftb(model_spec)
               if tag != 'noopt' and not isinstance(mod_res, OffDiagModel):
                   model_variables[model_spec] = all_models[model_spec].get_variables()
               # Detach it from the computational graph (unnecessary)
               elif tag == 'noopt':
                   all_models[model_spec].variables.requires_grad = False
           model = all_models[model_spec]
           feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
       
       for loss in all_losses:
           try:
               all_losses[loss].get_feed(feed, [] if loaded_data else training_molec_batches[ibatch], all_models, par_dict, debug)
           except Exception as e:
               print(e)
    
    
    print('Making validation feeds')
    for ibatch, feed in enumerate(validation_feeds):
        for model_spec in feed['models']:
            if (model_spec not in all_models):
                mod_res, tag = get_model_value_spline_2(model_spec, model_variables, model_range_dict, par_dict)
                all_models[model_spec] = mod_res
                #all_models[model_spec] = get_model_dftb(model_spec)
                if tag != 'noopt' and not isinstance(mod_res, OffDiagModel):
                    model_variables[model_spec] = all_models[model_spec].get_variables()
                elif tag == 'noopt':
                    all_models[model_spec].variables.requires_grad = False
            model = all_models[model_spec]
            feed[model_spec] = model.get_feed(feed['mod_raw'][model_spec])
        
        for loss in all_losses:
            try:
                all_losses[loss].get_feed(feed, [] if loaded_data else validation_molec_batches[ibatch], all_models, par_dict, debug)
            except Exception as e:
                print(e)
    print(f"{time.time() - x}")
    #%% Debugging h5 (Saving)
    
    #Save all the molecular information
    per_molec_h5handler.save_all_molec_feeds_h5(training_feeds, 'final_molec_test.h5')
    per_batch_h5handler.save_multiple_batches_h5(training_feeds, 'final_batch_test.h5')
    
    per_molec_h5handler.save_all_molec_feeds_h5(validation_feeds, 'final_valid_molec_test.h5')
    per_batch_h5handler.save_multiple_batches_h5(validation_feeds, 'final_valid_batch_test.h5')
    
    with open("reference_data1.p", "wb") as handle:
        pickle.dump(training_feeds, handle)
    with open("reference_data2.p", "wb") as handle:
        pickle.dump(validation_feeds, handle)
        
    # Also save the dftb_lsts for the training_feeds and validation feeds. Can do this using pickle for now
    with open("training_dftblsts.p", "wb") as handle:
        pickle.dump(training_dftblsts, handle)
        
    with open("validation_dftblsts.p", "wb") as handle:
        pickle.dump(validation_dftblsts, handle)
        
    print("molecular and batch information successfully saved, along with reference data")
    
    #%% Debugging inflection point analysis
    g_mods = [mod for mod in all_models.keys() if mod != 'Eref' and mod.oper == 'G' and len(mod.Zs) == 2]
    num_per_plot = 4
    num_row = num_col = 2
    sections = [g_mods[i : i + num_per_plot] for i in range(0, len(g_mods), num_per_plot)]
    new_dict = dict()
    rgrid = np.linspace(0, 10, 1000) #dense grid
    for sect in sections:
        fig, axs = plt.subplots(num_row, num_col) #sqrt of num_per_plot
        pos = 0
        for row in range(num_row):
            for col in range(num_col):
                axs[row, col].plot(rgrid, get_dftb_vals(sect[pos], par_dict, rgrid))
                axs[row, col].set_title(f"{sect[pos].oper}, {sect[pos].Zs}, {sect[pos].orb}")
                pos += 1
        fig.tight_layout()
        #save the figure...
        plt.show()
    
    #%% Recursive type conversion
    # Not an elegant solution but these two keys need to be ignored since they
    # should not be tensors!
    # Charges are ignored because of raggedness coming from bsize organization
    
    #If you are using the second version of dipole loss, ignore the dipole_mats too
    # because they are going to be a list of arrays
    ignore_keys = ['glabels', 'basis_sizes', 'charges', 'dipole_mat']
    
    for feed in training_feeds:
        recursive_type_conversion(feed, ignore_keys)
    for feed in validation_feeds:
        recursive_type_conversion(feed, ignore_keys)
    times['feeds'] = time.process_time()
    
    #%% Training loop
    '''
    Two different eig methods are available for the dftblayer now, and they are 
    denoted by flags 'new' and 'old'.
        'new': Implemented eigenvalue broadening method to work around vanishing 
        eigengaps, refer to eig.py for more details. Only using conditional broadening
        to cut down on gradient errors. Broadening factor is 1E-12.
        
        'old': Implementation using torch.symeig, standard approach from before
    
    Note: If you are using the old method for symmetric eigenvalue decomp, make sure
    to uncomment the section that runs the degeneracy rejection! Diagonalization will fail for 
    degenerate molecules in the old method.
    '''
    dftblayer = DFTB_Layer(device = None, dtype = torch.double, eig_method = eig_method)
    learning_rate = 1.0e-5
    optimizer = optim.Adam(list(model_variables.values()), lr = learning_rate, amsgrad=True)
    #TODO: Experiment with alternative learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01) 
    
    #Instantiate the loss layer here
    
    times_per_epoch = list()
    
    nepochs = 300
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
    times['train'] = time.process_time()
    #%% Logging
    print('dataset with', len(training_feeds), 'batches')
    time_names  = list(times.keys())
    time_vals  = list(times.values())
    for itime in range(1,len(time_names)):
        if time_names[itime] == 'train':
            print(time_names[itime], (time_vals[itime] - time_vals[itime-1])/nepochs)
        else:
            print(time_names[itime], time_vals[itime] - time_vals[itime-1])
    
    #Save the training and validation losses for visualization later
    with open("losses.p", "wb") as handle:
        pickle.dump(training_losses, handle)
        pickle.dump(validation_losses, handle)
    
    print(f"total time taken (sum epoch times): {sum(times_per_epoch)}")
    print(f"average epoch time: {sum(times_per_epoch) / len(times_per_epoch)}")
    # print(f"total number of molecules per epoch: {total_num_molecs}")
    # print(f"total number of training molecules: {total_num_train_molecs}")
    # print(f"total number of validation molecules: {total_num_valid_molecs}")
    
    #Plotting the change in each kind of loss per epoch
    for loss in all_losses:
        validation_loss = loss_tracker[loss][0]
        training_loss = loss_tracker[loss][1]
        # assert(len(validation_loss) == nepochs)
        # assert(len(training_loss) == nepochs)
        fig, axs = plt.subplots()
        axs.plot(training_loss, label = 'Training loss')
        axs.plot(validation_loss, label = 'Validation loss')
        axs.set_title(f"{loss} loss")
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Average Epoch Loss (unitless)")
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.xaxis.set_minor_locator(AutoMinorLocator())
        axs.legend()
        plt.show()
        
    from loss_methods import plot_multi_splines
    double_mods = [mod for mod in all_models.keys() if mod != 'Eref' and len(mod.Zs) == 2]
    plot_multi_splines(double_mods, all_models)
        
    # #Writing diagnostic information for later user
    # with open("timing.txt", "a+") as handle:
    #     handle.write(f"Current time: {datetime.now()}\n")
    #     handle.write(f"Allowed Zs: {allowed_Zs}\n")
    #     handle.write(f"Heavy Atoms: {heavy_atoms}\n")
    #     handle.write(f"Molecules per batch: {num_per_batch}\n")
    #     handle.write(f"Total molecules per epoch: {total_num_molecs}\n")
    #     handle.write(f"Total number of training molecules: {total_num_train_molecs}\n")
    #     handle.write(f"Total number of validation molecules: {total_num_valid_molecs}\n")
    #     handle.write(f"Number of epochs: {nepochs}\n")
    #     handle.write(f"Eigen decomp method: {eig_method}\n")
    #     handle.write(f"Total training time, sum of epoch times (seconds): {sum(times_per_epoch)}\n")
    #     handle.write(f"Average time per epoch (seconds): {sum(times_per_epoch) / len(times_per_epoch)}\n")
    #     handle.write("Infrequent charge updating for dipole loss\n")
    #     handle.write("Switched over to using non-shifted dataset\n")
    #     handle.write("Testing with new loss framework\n")
    #     handle.write("Testing dipole loss against actual dipoles\n")
    #     handle.write("\n")
