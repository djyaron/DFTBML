# DFTBML: A Machine-Learned Density Functional Based Tight Binding Model for the Deep Learning of Chemical Hamiltonians 
---
DFTBML provides a systematic way to parameterize the Density Functional-based Tight Binding (DFTB) semiempirical quantum chemical method for different chemical systems by learning the underlying Hamiltonian parameters rather than fitting the potential energy surface directly. By training to *ab initio* data in a manner analogous to that used to train other deep learning models, DFTBML adheres to traditional machine learning protocols while also incorporating significantly more physics by computing chemical properties via quantum chemical formalisms. This approach to semiempirical machine learning brings many advantages, namely high data efficiency, low computational cost, high accuracy, and deep interpretability.

# Relevant Publications
---
- Main DFTBML paper: https://arxiv.org/abs/2210.11682
- DFTB Layer background: https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00873

# Installation
---
1. Clone the repository:
```
git clone https://github.com/djyaron/DFTBML.git
cd DFTBML
```
2. Create and activate the virtual environment
```
conda env create -f environment.yml
conda activate DFTBML
```
3. Verify that everything works by running [tox](https://tox.wiki/en/latest/index.html):
```
tox
```
4. Run the directory_setup.py file 
```
cd DFTBML
python directory_setup.py
```
If everything runs without error, then you are good to go. Note that while DFTBML is known to work on Windows systems, running it on a Linux operating system is preferable. 

# Pre-trained models
---
A key advantage of DFTBML is that trained models can be saved as Slater-Koster files, otherwise known as SKF files. This file format is compatible with mainstream electronic structure calculation and molecular dynamics packages such as [DFTB+](https://dftbplus.org/), [AMBER](https://ambermd.org/), and [CP2K](https://www.cp2k.org/). The SKF files for pre-trained models can be found under the Example_SKFs directory along with the experimental conditions used to generate them. 

# Performing an analysis with pre-trained models
---
We can use the pretrained models in the `Example_SKFs` directory to get results and avoid the training step. For this process, we will choose the pretrained model, `DFTBML_CC_20000`, trained on 20000 molecules for CC-level energies. Any other pretrained model can be used by substituting its name in the following steps.

In the `analyze.py` file, change the `exec_path` variable to point to the `dftb+` binary of your [DFTB+ installation](https://dftbplus.org/). We recommend using version 21.1.

Run these scripts one at a time:
```bash
>> cp -r Example_SKFs/DFTBML_CC_20000 DFTBML/analysis_dir/results
# copy your test set into the directory. Ours is test_set.p and can be found in the 20000_cc_reproduction directory
>> cp DFTBML/20000_cc_reproduction/dset_20000_cc/test_set.p DFTBML/analysis_dir/results/DFTBML_CC_20000
>> cd DFTBML
>> nohup python analyze.py internal Y analysis_dir/results N &
```

Your results will populate the `dftbscratch` and `analysis_dir` directories.

# Reproducing a result from the paper
---
We provide the necessary data and scripts to run our entire workflow with a dataset containing 20000 molecules and 2500 molecules, both at a CC-level energy target. These directories, called `20000_cc_reproduction` and `2500_cc_reproduction`, respectively, are contained within the DFTBML directory. Each directory contains three bash scripts corresponding to the three steps of the workflow: `precompute_step.sh`, `train_step.sh`, and `analysis_step.sh`. For a more in-depth tutorial and explanation of these three steps, see the next section on "Training the model". Note that the `2500_cc_reproduction` is similar to the process done in "Training the model", but using the same split that was used in the paper.  

The scripts are intended to be run one at a time, as follows:
```bash
>> cd DFTBML
>> cp 2500_cc_reproduction/precompute_step.sh .
>> bash precompute_step.sh
# Wait for the precompute to finish
>> cp 2500_cc_reproduction/train_step.sh .
>> bash train_step.sh
# Wait for the model to finish training
>> cp 2500_cc_reproduction/analysis_step.sh .
>> bash analysis_step.sh
# Wait for analysis to finish  
```
Before executing the `analysis_step.sh` script, make sure you have correctly set the `exec_path` variable in `analyze.py` to point to the `dftb+` binary in your installation of [DFTB+](https://dftbplus.org/). We recommend using version 21.1. The results of the analysis will be contained in the `analysis_dir/analysis_files` directory, where there is a text file and pickle file version of the results for each experiment analyzed.

The `2500_cc_reproduction` workflow will take around two days to run whereas the `20000_cc_reproduction` will take much longer. It is best to do these calculations on a computing cluster with adequate memory and RAM. To see the process for setting up a smaller example, see the next section on "Training the model".  

# Training the model
---
## Preparing the raw data
The first step to training the DFTBML model is preparing the data. Unlike traditional machine learning, data preparation for DFTBML requires a precomputation process which generates batches in the format required for feeding through the network. The precomputation process is discussed in the following section, but first we will discuss how to best store and work with raw molecular data. 

We adopt a straightforward representation where each molecule is described by a single python dictionary and a dataset is a list of such dictionaries. To further streamline the file I/O around the raw data, we choose to save datasets and intermediate data structures using the pickle utility. To save something to pickle, one does the following:
```python
import pickle
with open("{filename}.p", "wb") as handle:
    pickle.dump(object, handle)
```
where the ```object``` is a generic python object. The advantage of this approach is that thanks to object serialization, anything in python can be directly saved to a pickle file and recovered later without additional processing. To load something from a pickle file, do the following:
```python
import pickle
with open("{filename}.p", "rb") as handle:
    object = pickle.load(handle)
```

Each molecule dictionary has the following entries:
|Field|Data Type|Dimension|Description|
|---|---|---|---|
|`name`|`str`|N/A|The empirical formula of the molecule, e.g. C1H4 for methane|
|`iconfig`|`int`|N/A|Arbitrary index used to distinguish between different molecules with the same empirical formula|
|`atomic_numbers`|`np.ndarray[uint8]`|(Natom,)|Array of atomic numbers specifying all the element types in the molecule|
|`coordinates`|`np.ndarray[float32]`|(Natom, 3)|Array of atomic cartesian coordinates given in Angstroms|
|`targets`|`dict`|N/A|Dictionary of different targets to train to, such as the total molecular energy or the molecular dipole|

DFTBML currently supports training to the following targets within the `targets` dictionary:
|Field|Data Type|Dimension|Unit|Description|
|---|---|---|---|---|
|`dipole`|`np.ndarray[float32]`|(3,)|eA|The net dipole vector of each molecule|
|`charges`|`np.ndarray[float32]`|(Natom,)|e|The atomic charge for each atom in the molecule|
|`Etot`|`float`|N/A|Ha|The total molecular energy|

Adding additional targets is an ongoing project. 

Because DFTBML was developed, trained, and benchmarked using the ANI-1ccx dataset from Olexandr Isayev and colleagues, utilities already exist to convert the ANI-1ccx hdf5 format to the molecule dictionary representation described above. Otherwise, you will have to massage your data into the correct format. 

## Setting up a precomputation
Once you have a set of molecular data in the molecule dictionary representation, the next step is to set up a precomputation. This is a fairly involved process because of the inherent complexity of the internal batch representation used in DFTBML, but here we provide a simple working example that should be sufficient for most applications. First, copy the `dataset.p` file from the `example_configs` directory to the DFTBML level and also make a directory called `precompute_test`:
```bash
>> cp example_configs/dataset.p .
>> mkdir precompute_test
```
Our directory setup is as follows, where we are working with DFTBML as our current working directory:
```
.
└── DFTBML/
    ├── dataset.p
    └── precompute_test/
```

The first step is to partition your dataset into a training and validation set. It is also good practice to set aside a disjoint test set of molecules for benchmarking the model performance later, but the test set is not involved in the precomputation process. Assuming that you have your dataset saved in the molecule dictionary representation in a file called `dataset.p`, a simple 80-20 train-valid split can be achieved as follows:
```python
import pickle, random
with open("dataset.p", "rb") as handle:
    mols = pickle.load(handle)
    random.shuffle(mols)
    index = int(len(mols) * 0.8)
    train_mols, valid_mols = mols[:index], mols[index:]
```
Once you have the two sets of molecules separated out, save them to the directory where you will perform the precomputation. In the example below, we are saving to an existing directory called `precompute_test` which sits inside the `DFTBML` directory:
```python
with open("precompute_test/Fold0_molecs.p", "wb") as handle:
    pickle.dump(train_mols, handle)
with open("precompute_test/Fold1_molecs.p", "wb") as handle:
    pickle.dump(valid_mols, handle)
```
Note that the specific names given, `Fold0_molecs.p` and `Fold1_molecs.p`, do matter since the code searches for all pickle files conforming to the generic pattern of `Fold[0-9]+_molecs.p`. Once you have saved the pickle files to the directory, the next step is to get a configuration file for the precompute. An example of one can be found in the `example_configs` directory inside `DFTBML`, called `dset_settings.json`. The json format is used for all DFTBML configuration files. Copy the `dset_settings.json` file into the `precompute_test` directory. 

For a basic precomputation, the only thing that needs to be changed in the `dset_settings.json` file is the field `top_level_fold_path`. Most paths are set from the perspective of `DFTBML` being the current working directory, and it is expected that most of the work occurs within the `DFTBML` directory. In our example, we set the `top_level_fold_path` to `precompute_test`:
```
#Other contents of dset_settings.json

    "loaded_data_fields": {
        "loaded_data": true,
        "top_level_fold_path": "precompute_test",
        "run_check": false,

#Other contents of dset_settings.json
```
Our overall directory structure now looks something like this:
```
.
└── DFTBML/
    ├── dataset.p
    └── precompute_test/
        ├── Fold0_molecs.p
        ├── Fold1_molecs.p
        └── dset_settings.json
```
Now we execute the precompute. With DFTBML still as our working directory, copy and paste the following code into a script, which we will call `precompute_run_script.py`:
```python
if __name__ == "__main__":
    from precompute_driver import precompute_folds
    from DatasetGeneration import process_settings_files
    
    current_default_file = "example_configs/refactor_default_tst.json"
    current_settings_file = "precompute_test/dset_settings.json"
    s_obj, opts = process_settings_files(current_settings_file, current_default_file)
    
    precompute_folds(s_obj, opts, s_obj.top_level_fold_path, True)
```
Activate your virtual environment and then run the script in the terminal:
```bash
>> conda activate DFTBML
>> python precompute_run_script.py
```
Depending on the size of your dataset, the precompute process may take a period of time. For this reason, we recommend running it headlessly using nohup in the background:
```bash
>> nohup python precompute_run_script.py &
```
Note that this only applies to linux systems. Once the precompute has completed, you will find that the `precompute_test` directory will become populated as follows:
```
.
└── DFTBML/
    ├── dataset.p
    └── precompute_test/
        ├── Fold0_molecs.p
        ├── Fold1_molecs.p
        ├── dset_settings.json
        ├── config_tracker.p
        ├── Fold0/
        │   └── ...
        ├── Fold0_config_tracker.p
        ├── Fold0_gammas.p
        ├── Fold1/
        │   └── ...
        ├── Fold1_config_tracker.p
        ├── Fold1_gammas.p
        └── gammas.p
```
The contents of the `Fold0` and `Fold1` directories have been ommitted for clarity. The entire directory `precompute_test` is now considered a precompute dataset and can be fed into DFTBML for model training. For clarity, let's rename this dataset as follows:
```bash
>> mv precompute_test example_dataset
```

## Training DFTBML
With our precomputed dataset `example_dataset`, we can now begin training the model. To begin, we need to create a configuration json file that specifies the parameters of our training session. This includes the generic machine learning parameters such as the number of epochs and the learning rate as well as more nuanced DFTBML-specific parameters. An example configuration file, `exp6.json`, is contained in `example_configs`.

Part of installing DFTBML was running the `directory_setup.py` script. This sets up the following directories inside of DFTBML:
```
.
└── DFTBML/
    ├── dataset.p
    ├── example_dataset/
    │   └── ...
    ├── benchtop_wdir/
    │   ├── dsets/
    │   ├── results/
    │   ├── settings_files/
    │   │   └── refactor_default_tst.json
    │   └── tmp/
    └── analysis_dir/
        ├── analysis_files/
        └── results/
```
Training requires `benchtop_wdir`. We will return to `analysis_dir` when we discuss benchmarking our trained models.

First, we need to copy our dataset into `benchtop_wdir/dsets` and our experiment file, `exp6.json`, into `benchtop_wdir/settings_files`. Then our directory structure is as follows:
```
.
└── DFTBML/
    ├── dataset.p
    ├── example_dataset/
    │   └── ...
    ├── benchtop_wdir/
    │   ├── dsets/
    │   │   └── example_dataset/
    │   │       └── ...
    │   ├── results/
    │   ├── settings_files/
    │   │   ├── exp6.json
    │   │   └── refactor_default_tst.json
    │   └── tmp/
    └── analysis_dir/
        ├── analysis_files/
        └── results/
```
We need to make two edits to `exp6.json`. First, we need to change the `top_level_fold_path` field so that it points to our dataset from the level of DFTBML as our working directory, and we also need to change the `run_id` field:
```
#Other contents of dset_settings.json

    "loaded_data_fields": {
        "loaded_data": true,
        "top_level_fold_path": "benchtop_wdir/dsets/example_dataset",
        "run_check": false,

#Other contents of dset_settings.json
    
    "run_id" : "example_train_result"
}
```
The `run_id` field indicates the name of the directory containing our trained model SKF files and metadata that will appear in the `benchtop_wdir/results` directory, so it's important to set this to something meaningful. `top_level_fold_path` again points to the directory that we want to use for training. Most of the other settings can be left as is, though you may want to decrease `nepochs` from 2500 to some smaller value to save time on training.

Once this is set up, go back to the DFTBML directory level. Activate the virtual environment and run the following command in your terminal:
```bash
>> conda activate DFTBML
>> nohup python benchtop.py &
```
We use the `nohup` command here because training usually takes a long time, so running it headlessly in the background is both convenient and safer. You will notice that a file will appear called `benchtop_wdir/EXP_LOG.txt`. This is a basic log file that will indicate the start and end times of experiments, as well as any safety checks that were conducted during the course of training. 

After training, the results will show up in the `benchtop_wdir/results` directory. Our directory structure now looks as follows:
```
.
└── DFTBML/
    ├── dataset.p
    ├── example_dataset/
    │   └── ...
    ├── benchtop_wdir/
    │   ├── EXP_LOG.txt
    │   ├── dsets/
    │   │   └── example_dataset/
    │   │       ├── ...
    │   │       └── Split0/
    │   │           └── ...
    │   ├── results/
    │   │   └── example_train_result/
    │   │       └── ...
    │   ├── settings_files/
    │   │   ├── exp6.json
    │   │   └── refactor_default_tst.json
    │   └── tmp/
    │       └── exp6_TMP.txt
    └── analysis_dir/
        ├── analysis_files/
        └── results/
```
The `Split0` directory that appears inside `benchtop_wdir/dsets/example_datasets` also appears in `benchtop_wdir/results/example_train_result/` and contains some additional metadata about training (epoch times, data for loss curve visualization, etc.). The file that appears in `benchtop_wdir/tmp` is a placeholder to prevent experiments from overriding each other in the case of distributed training across multiple servers. At this point, the `example_train_result` directory is ready for analysis. 

Congratulations on successfully training DFTBML!

## Model evaluation
Now that we have successfully trained the model, the next step is to evaluate the trained model by running the resulting SKF files on a test set of molecules. In principle, this can be done using a number of different software packages for quantum chemical calculations, but all the DFTBML testing done during development used [DFTB+](https://dftbplus.org/). The instructions provided here are for evaluating the model using DFTB+.

First, we need to move our resulting SKFs from `benchtop_wdir/results` to `analysis_dir/results`. Our directory structure is now:
```
.
└── DFTBML/
    ├── dataset.p
    ├── example_dataset/
    │   └── ...
    ├── benchtop_wdir/
    │   ├── EXP_LOG.txt
    │   ├── dsets/
    │   │   └── example_dataset/
    │   │       └── ...
    │   ├── results/
    │   │   └── example_train_result/
    │   │       └── ...
    │   ├── settings_files/
    │   │   ├── exp6.json
    │   │   └── refactor_default_tst.json
    │   └── tmp/
    │       └── exp6_TMP.txt
    └── analysis_dir/
        ├── results/
        │   └── exmaple_train_result/
        │       └── ...
        └── analysis_files/
```
We no longer need to worry about `benchtop_wdir`, so all diagrams from now on will only focus on `analysis_dir`. The script we need to run to analyze our results is `analysis.py`, which can be used to either analyze one set of SKFs or a collection of SKFs. Even though we only have one set of results, the command line argument for analyzing a collection of SKFs is simpler, so we will use that functionality here.

First, move your test set into the `analysis_dir/example_train_results` directory. Your test set should similarly be in the molecule dictionary representaiton and the file should be named `test_set.p`. Our directory now looks like this:
```
.
└── DFTBML/
    ├── ...
    └── analysis_dir/
        ├── results/
        │   └── exmaple_train_result/
        │       ├── ...
        │       └── test_set.p
        └── analysis_files/
```
To run the evaluation, we need access to the dftb+ binary. To install DFTB, follow the installation instructions at https://dftbplus.org/ for your system. Then, open up the `analyze.py` script and change the `exec_path` variable to point to the `DFTB+` binary in your system. We recommend using version 21.1.

Now to run the analysis, make sure the virtual environment is activated and run the script in your terminal from the DFTBML directory level:
```bash
>> nohup python analyze.py internal Y analysis_dir/results N &
```
Note that the analyze.py script takes four arguments. They are as follows:
|Argument|Data Type|Possible Values|Description|
|---|---|---|---|
|`dset`|`string`|Either a file path or `internal`|The name of the dataset to use. Setting this argument as `internal` means the code will search for the `test_set.p` file stored in the results directory|
|`batch`|`string`| `Y` or `N`|Whether you are analyzing a collection of SKF directories (`Y`) or a single SKF directory (`N`)|
|`skf_dir`|`string`|Usually `analysis_dir/results`|Location where the result SKF directories are stored|
|`fit_fresh_ref`|`string`|`Y` or `N`|Toggle for applying a fresh reference energy fit. Used for cases of SKF files without trained reference energy parameters (e.g. the default parameter sets listed with DFTB+)|

Running the above command will perform the analysis for you. The transient input files should be written to the `dftbscratch` directory which is setup by `directory_steup.py`. 

Once the analysis is complete, you will find the results under the `analysis_files` directory. In our case, the directory structure should look like:
```
.
└── DFTBML/
    ├── ...
    └── analysis_dir/
        ├── results/
        │   └── exmaple_train_result/
        │       ├── ...
        │       └── test_set.p
        └── analysis_files/
            ├── analysis_example_train_result.txt
            └── analysis_example_train_result.p
```
The text file contains a readout of the evaluation metrics (MAE on energy and different targets) while the pickle file contains the test set molecules where each molecule dictionary is updated with the DFTB+ calculation results. This is useful if you wish to perform further analysis on your results on a per-molecule basis, such as looking at the rates of convergence or outlier occurrence. 

## Next steps
That concludes this tutorial on the DFTBML training pipeline. This was intended as a quick start guide for first-time users who just want to gain some fmailiarity with the code base and the high-level steps involved with parameterizing based on a given class of systems. Of course, there are a lot of other features built into DFTBML that are accessible through the configuration json files, so for those interested in contributing or designing more elaborate experiments, we recommend looking at the documentation provided at [WIP]. 

# Data
As shown in the main manuscript, the two major datasets used for model development, training, and benchmarking are the ANI-1ccx dataset and the COMP6 dataset. The publications describing these datasets are:
- COMP6 Dataset: https://aip.scitation.org/doi/10.1063/1.5023802
- ANI-1ccx: https://www.nature.com/articles/s41597-020-0473-z

# Known Limitations
- DFTBML currently only works with organic molecules containing C, O, N, and H, but extending to elements with higher angular momenta is underway
- Additional interactions and forces need to be implemented, such as finite temperature effects and dispersion
- Additional targets for training need to be added beyond just total energy, dipoles, and atomic charges

  hield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
