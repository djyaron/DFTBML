"""
Setup file for first-time users

This file sets up the following directories and subdirectories:
    benchtop_wdir:
        dsets:
        tmp:
        settings_files:
        results:
    analysis_dir:
        analysis_files
        results 
"""
import os, shutil

if __name__ == "__main__":
    #Create the directories if they are not already created
    #   and populate the necessary subdirectories and files
    if not os.path.isdir("benchtop_wdir"):
        os.mkdir("benchtop_wdir")
        os.mkdir("benchtop_wdir/dsets")
        os.mkdir("benchtop_wdir/results")
        os.mkdir("benchtop_wdir/settings_files")
        os.mkdir("benchtop_wdir/tmp")
    if not os.path.isdir("analysis_dir"):
        os.mkdir("analysis_dir")
        os.mkdir("analysis_dir/analysis_files")
        os.mkdir("analysis_dir/results")
    
    try:
        assert(os.path.isdir("example_configs"))
    except:
        raise ValueError("Missing example_configs directory!")
    
    src = "example_configs/refactor_default_tst.json"
    dst = "benchtop_wdir/settings_files/"
    shutil.copy(src, dst)

    print("Directory setup complete!")

    

