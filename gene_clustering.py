import numpy, scipy

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,dendrogram

from epigenetics_data_processing import EpigeneticsData


#def compute_genes_similarities()

#def create_distance_matrix()


geneId_to_expressionProfile = EpigeneticsData.geneId_to_expressionProfile

geneExpressions = []

geneIds = geneId_to_expressionProfile.keys()
for geneId in geneIds:
    geneExpressions += [geneId_to_expressionProfile[geneId]]

print geneIds
print geneExpressions


data_dist = pdist(geneExpressions, metric='euclidean')
data_link = linkage(data_dist)

print data_dist

dendrogram(data_link, labels=geneIds)

plt.ylabel("Distances")
plt.xlabel("Genes")
plt.show()
