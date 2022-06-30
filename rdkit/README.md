<h3 align="center">Using RDKit for InChI Key and chemical fingerprints</h3>


This folder contains python files using the open-source bioinformatics toolkit [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html). In this project, Rdkit is used to identify the inchikey for the compounds within the Tox21 and Pharos compound list, given the SMILES identifier. Below are the key functions used.

- **MolFromSmiles**: Construct a molecule from a SMILES string; takes in a smiles string and returns a Mol object.
- **Chem.MolToInchiKey**: Given a mol object, returns the corresponding inchikey for that object. 


RDKit is also used to calculate the Tanimoto molecular similarity between the compounds. RDKit also has a variety of built-in functionality for generating molecular fingerprints and using them to calculate molecular similarity between compounds. The default metric is the Tanimato similarity index. 

- **RDKFingerprint()**: Returns an RDKit topological fingerprint for a molecule; takes in a Mol object and returns the chemical fingerprint for that molecule (i.e., returns a DataStructs.ExplicitBitVect with _fpSize_ bits).


