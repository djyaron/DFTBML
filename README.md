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
git clone https://github.com/djyaron/dftbtorch.git
cd dftbtorch
```
2. Verify that everything works by running [tox](https://tox.wiki/en/latest/index.html):
```
tox
```
3. Create and activate the virtual environment
```
conda create env -f environment.yml
conda activate DFTBML
```
4. Run the directory_setup.py file 
```
cd DFTBML
python directory_setup.py
```
If everything runs without error, then you are good to go.

# Pre-trained models
---
A key advantage of DFTBML is that trained models can be saved as Slater-Koster files, otherwise known as SKF files. This file format is compatible with mainstream electronic structure calculation and molecular dynamics packages such as [DFTB+](https://dftbplus.org/), [AMBER](https://ambermd.org/), and [CP2K](https://www.cp2k.org/). The SKF files for pre-trained models can be found under the Example_SKFs directory along with the experimental conditions used to generate them. 

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
Once you have a set of molecular data in the molecule dictionary representation, the next step is to set up a precomputation. This is a fairly involved process because of the inherent complexity of the internal batch representation used in DFTBML, but here we provide a simple working example that should be sufficient for most applications. 

The first step is to partition your dataset into a training and validation set. It is also good practice to set aside a disjoint test set of molecules for benchmarking the model performance later, but the test set is not involved in the precomputation process. Assuming that you have your dataset saved in the molecule dictionary representation in a file called `dataset.p`, a simple 80-20 train-valid split can be achieved as follows:
```python
import pickle, random
with open("dataset.p", "rb") as handle:
    mols = pickle.load(handle)
    random.shuffle(mols)
    index = int(len(mols) * 0.8)
    train_mols, valid_mols = mols[:index], mols[index:]
```
Once you have the two sets of molecules separated out, save them to the directory where you will perform the precomputation. In the example below, we are saving to an existing directory called `precompute_test`:
```python
with open("precompute_test/Fold0_molecs.p", "wb") as handle:
    pickle.dump(train_mols, handle)
with open("precompute_test/Fold1_molecs.p", "wb") as handle:
    pickle.dump(valid_mols, handle)
```
Note that the specific names given, `Fold0_molecs.p` and `Fold1_molecs.p`, do matter since the code searches for all pickle files conforming to the generic pattern of `Fold[0-9]+_molecs.p`. Once you have saved the pickle files to the directory, the next step is to get 

# Data
# Known Limitations
