# DFTBML: A Machine-Learned Density Functional Based Tight Binding Model for the Deep Learning of Chemical Hamiltonians 
---
DFTBML provides a systematic way to parameterize the Density Functional-based Tight Binding (DFTB) semiempirical quantum chemical method for different chemical systems by learning the underlying Hamiltonian parameters rather than fitting the potential energy surface directly. By training to *ab initio* data in a manner analogous to that used to train other deep learning models, DFTBML adheres to traditional machine learning protocols while also incorporating significantly more physics by computing chemical properties via quantum chemical formalisms. This approach to semiempirical machine learning brings many advantages, namely high data efficiency, low computational cost, high accuracy, and deep interpretability.

# Relevant Publications
---
- Main DFTBML paper: https://arxiv.org/abs/2210.11682
- DFTB Layer background: https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00873

# Installation and Dependencies
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

# Pre-trained models
---
A key advantage of DFTBML is that trained models can be saved as Slater-Koster files, otherwise known as SKF files. This file format is compatible with mainstream electronic structure calculation and molecular dynamics packages such as [DFTB+](https://dftbplus.org/), [AMBER](https://ambermd.org/), and [CP2K](https://www.cp2k.org/). The SKF files for pre-trained models can be found under the Example_SKFs directory along with the experimental conditions used to generate them. 

# Usage
# Data
# Known Limitations
