from rdkit import Chem
from rdkit import DataStructs

#find the mol using the SMILES identifiers
ms = [Chem.MolFromSmiles('Cl.Cc4ncnc4C[C@H]3CCc2c(C)c1ccccc1n2C3=O', sanitize = False), Chem.MolFromSmiles('FC(F)(F)C(=O)O.CCN(CC)C(=O)c1cc(c(cc1N(CC)CCN(C)C)N2CCC(CC2)c3ccccc3)S(=O)(=O)Cc4ccccc4')]

#find the bit fingerprint for all structures
fps = [Chem.RDKFingerprint(x) for x in ms]

#print the Tonimoto similarity index
print(DataStructs.FingerprintSimilarity(fps[0],fps[1]))

"""
Goal: create an array of smiles --> use MolFromSmiles() to loop over
the array of smiles to create molfiles! --> use molfiles to find fingerprint
THEN, use fingerprint to find similarity!

Important functions:

Chem.MolFromSmiles --> input smiles, output mol
Chem.RDKFingerprint --> input mol, output fingerprint
DataStructs.FingerprintSimilarity --> input fingerprint, output Tonimoto similarity



"""
