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
|`atomic_numbers`|`np.ndarray[uint8]`|($$N_{atom}$$,)|Array of atomic numbers specifying all the element types in the molecule|
|`coordinates`|`np.ndarray[float32]`|($$N_{atom}$$, 3)|Array of atomic cartesian coordinates given in Angstroms $$\AA$$|

# Data
# Known Limitations
