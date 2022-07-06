from rdkit import Chem
from rdkit import DataStructs
import numpy
import csv


smiles = []
rows = [] #save output with inchikey

#open file for reading
with open('./data/pharos_symbol.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    next(csvReader) #skip the first line/header

    #loop over each row of the smiles.csv
    for row in csvReader:
        mol = Chem.MolFromSmiles(row[1])
        if(mol == None):
            inchikey = "NA"
        else:
            inchikey = Chem.MolToInchiKey(mol)
        row.append(inchikey)
        rows.append(row)

with open('./results/pharos_to_inchikey.csv', 'w') as outputFile:
    csvWriter = csv.writer(outputFile)

    #create column names
    colname = ['smiles', 'symbol', 'inchikey']

    csvWriter.writerow(colname)
    csvWriter.writerows(rows)
