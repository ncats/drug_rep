# RDKIT

This folder contains python files using the open-source bioinformatics toolkit [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).
Rdkit was used to identify the inchikey for the compounds within our Tox21 list (i.e., ) and also calculating the tanimoto molecular similarity between the compounds. 


The RDKit has a variety of built-in functionality for generating molecular fingerprints and using them to calculate molecular similarity:

from rdkit import DataStructs
ms = [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('COC')]
fps = [Chem.RDKFingerprint(x) for x in ms]
DataStructs.FingerprintSimilarity(fps[0],fps[1])