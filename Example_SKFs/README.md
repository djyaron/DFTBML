# Pre-trained model SKF files
---
This directory contains all the SKF files used to generate the results in the main DFTBML paper. Each directory is named according to the energy target and the number of molecules used for training. For example, the directory DFTBML_CC_2500 contains the SKF files generated from training DFTBML on _ab initio_ calculations for 2500 molecules at a CCSD(T)\_CBS level of theory for the total energy. 

Each directory contains the following:
- The SKF files for the elements C, O, N, and H. There are a total of 16 SKF files per directory specifying every possible element pair. 
- A pickle file called ref_params.p. Depending on what method DFTBML is trained to, these parameters are used to perform a linear reference energy correction between the output of DFTBML and the target _ab initio_ method. 
- A json file containing the experimental parameters used for the experiment that generated the SKF files. 

These SKFs are ready to use directly. However, note that each SKF set was trained to minimize the residual errors against a given _ab initio_ method after applying a linear reference energy correction. To idea is that using the DFTB level energies calculated using these SKFs, applying a linear reference energy correction with the trained parameters contained in ref_params.p will correct the prediction up to the level of the _ab initio_ target without any additional cost or calculation. For more information, please refer to the Experimental Details section of the main DFTBML paper. 
