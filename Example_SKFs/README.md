# Pre-trained model SKF files
---
This directory contains all the SKF files used to generate the results in the main DFTBML paper. Each directory is named according to the energy target and the number of molecules used for training. For example, the directory DFTBML_CC_2500 contains the SKF files generated from training DFTBML on _ab initio_ calculations for 2500 molecules at a CCSD(T)\_CBS level of theory for the total energy. 

Each directory contains the following:
- The SKF files for the elements C, O, N, and H. There are a total of 16 SKF files per directory specifying every possible element pair. 
- A pickle file called ref_params.p. Depending on what method DFTBML is trained to, these parameters are used to perform a linear reference energy correction between the output of DFTBML and the target _ab initio_ method. This takes the form of a vector of constants for each atom type and a constant term.
- A json file containing the experimental parameters used for the experiment that generated the SKF files. 

These SKFs are ready to use directly. However, note that each SKF set was trained to minimize the residual errors against a given _ab initio_ method after applying a linear reference energy correction. The idea is that using the DFTB level energies calculated using these SKFs, applying a linear reference energy correction with the trained parameters contained in ref_params.p will correct the prediction up to the level of the _ab initio_ target without any additional cost or calculation.

For example, suppose you would like to compute the total energy of a molecule at a DFT level of theory by running DFTB using a trained SKF set. Using the trained SKF set, you will obtain a DFTB level energy for the molecule, $E_{DFTB}$. To correct this energy to the DFT level, you would perform a linear reference energy correction as follows:

$$
E_{DFT} = E_{DFTB} + \sum_{z}N_zC_z + C_0
$$

Where $N_z$ is the number of atoms of type $z$ in the molecule and $C_z$ is the associated constant for the atom type $z$. $C_0$ is an additional constant term. It is the vector of $C_z$ and the $C_0$ that are stored in the ref_params.p file of each SKF directory. 
