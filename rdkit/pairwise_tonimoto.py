from rdkit import Chem
from rdkit import DataStructs
import numpy as np
import statistics
import csv

smiles = []

with open('./data/smiles.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    next(csvReader) #skip the first line

    for row in csvReader:
        smiles.append(row[0])

#find mol using SMILES identifiers
ms = [Chem.MolFromSmiles(x, sanitize=False) for x in smiles]

#find the bit fingerprint for all structures
fps = [Chem.RDKFingerprint(x) for x in ms]

#testing
#print(DataStructs.FingerprintSimilarity(fps[52],fps[52]))
#print(DataStructs.FingerprintSimilarity(fps[52],fps[54]))

"""
range(start, stop, step)

for n in range(6):
    print(n)
RESULT: 0 1 2 3 4 5

for n in range(3, 6):
    print(n)
RESULT: 3 4 5
"""


#write function to calculate the Upper/TRIANGULAR Tanimoto similarity matrix
def tanimoto(fp):
    similarity = []

    #return the Tanimoto similarities between one vector and a
    #sequence of other
    for i in range(1, len(fp)): #len(fp)
        similarities = DataStructs.BulkTanimotoSimilarity(fp[i], fp[:i])
        similarity.extend(similarities)
    return similarity

#extend(): add the elements of a list to the end of the current list
test1 = tanimoto(fps)
print(len(test1))

#find the mean and median of ALL the pairwise Tanimoto similarities
mean = statistics.mean(test1)
median = statistics.median(test1)
print("The average Tanimoto similarity index is: ", mean)
print("The median Tanimoto similarity index is: ", median)


# The codechunks below creates a nxn matrix of ALL the pairwise similarity
"""
#test matrix --> see if this method works first!
b = np.zeros((5,5))
b[2,3] = 5
for i in range(5):
    for j in range(5):
        b[i,j] = DataStructs.FingerprintSimilarity(fps[i],fps[j])

np.savetxt("test1.txt", b, fmt='%.2f')


#Complete matrix
def tani_matrix(fp):
    n = len(fp) #how many compounds are there?
    matrix = np.zeros((n,n)) #initial nxn matrix with all zeroes
    for i in range(n):
        for j in range(n):
            matrix[i,j] = DataStructs.FingerprintSimilarity(fp[i],fp[j])
    return matrix

print(len(tani_matrix(fps)))
print(len(tani_matrix(fps)[0]))


#save the complete nxn matrix as .txt
#np.savetxt("complete_matrix.txt", tani_matrix(fps), fmt='%.2f')
"""
