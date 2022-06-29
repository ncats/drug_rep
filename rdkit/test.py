from rdkit import Chem
from rdkit import DataStructs
import numpy
import csv

'''
mol = Chem.MolFromSmiles('OCc1cc(ccc1O)C(O)CNCCCCCCOCCCCc2ccccc2.O=C(O)c2ccc1ccccc1c2O')
inchikey_salt = Chem.MolToInchiKey(mol)
print(inchikey_salt)

mol_parent = Chem.MolFromSmiles('OCc1cc(ccc1O)C(O)CNCCCCCCOCCCCc2ccccc2')
inchikey_parent = Chem.MolToInchiKey(mol_parent)
print(inchikey_parent)
'''

mol_parent = Chem.MolFromSmiles('CC(C)(C)C(O)C(=CC1=C(Cl)C=C(Cl)C=C1)N1C=NC=N1')
inchikey_parent = Chem.MolToInchiKey(mol_parent)
print(inchikey_parent)

mol_parent = Chem.MolFromSmiles('[c-]1cccc1')
inchikey_parent = Chem.MolToInchiKey(mol_parent)
print(inchikey_parent)
