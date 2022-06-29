from rdkit import Chem
from rdkit import DataStructs
import numpy
import csv


#goal: create an array of smiles
import csv

#initialize an array that will contain smiles
smiles = []
rows = [] #save output with inchikey

#open file for reading
#with open('./data/cluster_and_smiles.csv') as csvDataFile:
with open('./data/cluster_and_smiles_05_10.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    next(csvReader) #skip the first line/header

    #loop over each row of the smiles.csv
    for row in csvReader:
        #smiles.append(row[2])
        mol = Chem.MolFromSmiles(row[2])
        if(mol == None):
            inchikey = "NA"
        else:
            inchikey = Chem.MolToInchiKey(mol)
        row.append(inchikey)
        rows.append(row)


#To save the result to another file:
#numpy.savetxt("./results/smi_to_inchikey.txt", rows)

#with open('./results/smi_to_inchikey.csv', 'w') as outputFile:
with open('./results/smi_to_inchikey_05_10.csv', 'w') as outputFile:

    csvWriter = csv.writer(outputFile)

    #create column names
    colname = ['cas', 'cluster', 'smiles', 'inchikey']

    csvWriter.writerow(colname)
    csvWriter.writerows(rows)


'''
#find mol using SMILES identifiers for EACH compound!
mol = [Chem.MolFromSmiles(x) for x in smiles]
print(len(mol))
#print(mol)

sample_mol = mol[1:5] #list of mol
print(sample_mol)

for mol in sample_mol:
    print(Chem.MolToInchiKey(mol))



#test individually

test1 = Chem.MolFromSmiles(smiles[1])
print(test1)

test = Chem.MolToSmiles(test1)
print(test)
'''

#using the mol objects from above, find the canonical smiles!
#smiles_can = [Chem.MolToSmiles(x) for x in mol]
#print(smiles_can)

#test = None
#print(test)

#find inchi using the 'mol'
#inchi = [Chem.MolToInchi(x) for x in mol]

#Chem.MolToInchiKey()

#sanitize=False
#Chem.MolToSmiles(Chem.MolFromSmiles('C1=CC=CN=C1'))
