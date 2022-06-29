from rdkit import Chem
from rdkit import DataStructs

#goal: create an array of smiles
import csv

#initialize an array that will contain smiles
smiles = []

#open file for reading
with open('./data/smiles.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader) #skip the first line

    #loop over each row of the smiles.csv
    for row in csvReader:
        smiles.append(row[0]) #append the 1st element of each row to the SMILES Array

#find mol using SMILES identifiers for EACH compound!
ms = [Chem.MolFromSmiles(x, sanitize=False) for x in smiles]
print(len(ms))

#find the fingerprint(bit array) for all structures
fps = [Chem.RDKFingerprint(x) for x in ms]
print(len(fps))

#testing to see if the fingerprint are generated correctly
print(DataStructs.FingerprintSimilarity(fps[7027],fps[7029]))

#NOTE: in python index starts at 0
