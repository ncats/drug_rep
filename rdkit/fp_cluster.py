from rdkit import Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt
import statistics
import numpy
import csv

#clusters = array of array
#cluster #2 --> 3 compound[smiles1, smiles2, smiles3]

cluster = [] #intermediate placeholder of SMILES for all the compound within each cluster!
smiles = []
smile_clusters = [] #array of array!!!

#open file for reading
with open('./data/cluster_and_smiles.csv') as csvDataFile:
    csvData = csv.reader(csvDataFile)

    next(csvData) #skip the first line

    current_cluster = 1

    #loop over each row of the cluster_smiles file
    for row in csvData:
        cluster_n = int(row[1]) #cluster_n = cluster # (i.e., 1, 2... 144)
        smiles = row[2] #second element of file is the SMILES

        if(cluster_n == current_cluster):
            cluster.append(smiles)
        else: #moves on to another cluster!
            smile_clusters.append(cluster)
            current_cluster = cluster_n
            cluster = [] #empty the intermediate placeholder!!
            cluster.append(smiles)
    smile_clusters.append(cluster)

print(len(smile_clusters)) #this should equal to the # of unique clusters!!

#find the fingerprint for each compound within each clusters
fp_cluster = []

for cluster in smile_clusters:
    ms = [Chem.MolFromSmiles(x, sanitize=False) for x in cluster]
    fps = [Chem.RDKFingerprint(x) for x in ms]
    fp_cluster.append(fps)



def tanimoto_distance_matrix(fp_list):
    """Calculate DISTANCE matrix for fingerprint list"""
    dissimilarity_matrix = []

    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def intra_tanimoto(fps_clusters):
    """Function to compute Tanimoto similarity for all pairs of fingerprints in each cluster"""
    intra_similarity = []

    # Calculate intra similarity per cluster
    for clus in fps_clusters:
        # Tanimoto distance matrix function converted to similarity matrix (1-distance)
        intra_similarity.append([1 - x for x in tanimoto_distance_matrix(clus)])
    return intra_similarity

intra_sim = intra_tanimoto(fp_cluster)

# DEBUG - check if the intra_sim contains the tanimoto similarity
# for all pairwise comparison in a cluster
# print(intra_sim[2])

#print the mean tanimoto similarity index for each cluster
result = []
for x in intra_sim:
    result.append(numpy.mean(x))
#print(result)

#To save the result to another file:
numpy.savetxt("./results/cluster_average.txt", result, fmt='%.3f')

mean = statistics.mean(result)
median = statistics.median(result)
print(mean)
print(median)

'''
r = plt.violinplot(intra_sim[:10],showmeans=True, showextrema=False, showmedians=True)
r["cmeans"].set_color("red")
plt.show()
'''


# Create Violin plots to show intra-cluster similarity
fig, ax = plt.subplots(figsize=(10, 5))
indices = list(range(1,30,1))
#print(indices)
ax.set_xlabel("Cluster index")
ax.set_ylabel("Similarity")
ax.set_xticks(indices)
ax.set_xticklabels(indices)
ax.set_yticks(numpy.arange(0, 1.0, 0.2))
ax.set_title("Intra-cluster Tanimoto similarity")
r = ax.violinplot(intra_sim[1:30], indices, showmeans=True, showmedians=True, showextrema=False)
r["cmeans"].set_color("red")
plt.show()
