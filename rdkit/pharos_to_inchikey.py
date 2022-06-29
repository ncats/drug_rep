from rdkit import Chem
from rdkit import DataStructs
import numpy
import csv

'''
mol = Chem.MolFromSmiles("N[C@H](C(=O)O)CC#N")
inchikey = Chem.MolToInchiKey(mol)
print(inchikey)
'''


smiles = []
rows = [] #save output with inchikey

#open file for reading
#with open('./data/pharos_symbol.csv') as csvDataFile:
with open('./data/pharos_symbol_05_10.csv') as csvDataFile:


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


#To save the result to another file:
#numpy.savetxt("./results/smi_to_inchikey.txt", rows)

#with open('./results/pharos_to_inchikey.csv', 'w') as outputFile:
with open('./results/pharos_to_inchikey_05_10.csv', 'w') as outputFile:
    csvWriter = csv.writer(outputFile)

    #create column names
    colname = ['smiles', 'symbol', 'inchikey']

    csvWriter.writerow(colname)
    csvWriter.writerows(rows)
