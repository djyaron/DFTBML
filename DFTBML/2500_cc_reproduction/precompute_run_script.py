if __name__ == "__main__":
    from precompute_driver import precompute_folds
    from DatasetGeneration import process_settings_files
    
    current_default_file = "example_configs/refactor_default_tst.json"
    current_settings_file = "2500_cc_reproduction/dset_2500_cc/dset_settings.json"
    s_obj, opts = process_settings_files(current_settings_file, current_default_file)
    
    precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)
