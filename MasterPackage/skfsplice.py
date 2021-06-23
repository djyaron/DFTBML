# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:22:32 2021

@author: fhu14

Taking trained electronic portions of skf files and putting non-trained 
repulsive on to see the extent of the electronic training.
"""

#%% Imports, definitions
import os


#%% Code behind
trained_skf_path = os.path.join(os.getcwd(), "skfarchive", "skf_8020_100knot")
#Directory for the skf files with the original repulsive
ref_direc = os.path.join(os.getcwd(), "Auorg_1_1", "auorg-1-1")
dest_dir = "spliced_skf"

if (not os.path.isdir(dest_dir)):
    os.mkdir(dest_dir)
    
all_files = os.listdir(trained_skf_path)

good_files = list(filter(lambda x : len(x.split(".")) == 2 and x.split(".")[1] == "skf", all_files))

for file_name in good_files:
    print(f"Writing out spliced {file_name}")
    trained_path = os.path.join(trained_skf_path, file_name)
    reference_path = os.path.join(ref_direc, file_name)
    trained_content = open(trained_path, 'r').read().splitlines()
    reference_content = open(reference_path, 'r').read().splitlines()
    ref_spline_ind = reference_content.index("Spline")
    train_spline_ind = trained_content.index("Spline")
    resulting_content = trained_content[:train_spline_ind] + reference_content[ref_spline_ind:]
    
    dest_path = os.path.join(dest_dir, file_name)
    with open(dest_path, "w+") as handle:
        for line in resulting_content:
            handle.write(line + "\n")

print("Finished writing spliced skf files")


            
    
    


