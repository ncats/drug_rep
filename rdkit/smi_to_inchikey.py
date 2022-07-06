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
with open('./data/cluster_and_smiles.csv') as csvDataFile:

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


with open('./results/smi_to_inchikey.csv', 'w') as outputFile:

    csvWriter = csv.writer(outputFile)

    #create column names
    colname = ['cas', 'cluster', 'smiles', 'inchikey']

    csvWriter.writerow(colname)
    csvWriter.writerows(rows)
    
