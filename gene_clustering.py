import numpy as np
import matplotlib.pyplot as plt
from epigenetics_data_processing import EpigeneticsData
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,dendrogram


def find_closest_clusters(distance_matrix):

    min = float('inf')
    min_index = 0
    max_index = 0

    matrix_shape = distance_matrix.shape

    for index_i in range(matrix_shape[0]):
        for index_j in range(matrix_shape[1]):
            if (distance_matrix[index_i][index_j] < min) & (index_i != index_j):
                min = distance_matrix[index_i][index_j]
                if(index_i < index_j):
                    min_index = index_i
                    max_index = index_j

    return min_index, max_index


def hierarchical_clustering(geneId_to_expressionProfile, min_num_clusters):
    geneIds = geneId_to_expressionProfile.keys()
    geneExpressions = []
    for geneId in geneIds:
        geneExpressions += [geneId_to_expressionProfile[geneId]]

    distance_matrix = squareform(pdist(geneExpressions, metric='correlation'))

    # create independent clusters, such that each gene is part of a single cluster
    clusters = list()
    for index in range(len(geneIds)):
        clusters.append([geneIds[index]])
    num_clusters = len(geneIds)

    while(num_clusters > min_num_clusters):
        min_index, max_index = find_closest_clusters(distance_matrix)

        # combine clusters
        clusters[min_index] += clusters[max_index]
        del clusters[max_index]

        # update distance matrix
        for index_i in range(num_clusters):
            distance_matrix[min_index][index_i] = \
                (distance_matrix[min_index][index_i] + distance_matrix[max_index][index_i])/2
            distance_matrix[index_i][min_index] = distance_matrix[min_index][index_i]
        distance_matrix = np.delete(distance_matrix, max_index, 0)
        distance_matrix = np.delete(distance_matrix, max_index, 1)

        num_clusters -= 1

    geneId_to_cluster = dict()
    for index_i in range(num_clusters):
        print(len(clusters[index_i]))
        for gene in clusters[index_i]:
            geneId_to_cluster[gene] = index_i

    print clusters
    print geneId_to_cluster


def plot_dendogram(geneId_to_expressionProfile):
    geneExpressions = []
    geneIds = geneId_to_expressionProfile.keys()
    for geneId in geneIds:
        geneExpressions += [geneId_to_expressionProfile[geneId]]

    distance_matrix = pdist(geneExpressions, metric='correlation')
    dendrogram(linkage(distance_matrix, method='average'), labels=geneIds)


geneId_to_expressionProfile = EpigeneticsData.geneId_to_expressionProfile

geneExpressions = []

geneIds = geneId_to_expressionProfile.keys()
for geneId in geneIds:
    geneExpressions += [geneId_to_expressionProfile[geneId]]


data_dist = pdist(geneExpressions, metric='correlation')
#print(squareform(data_dist))
data_link = linkage(data_dist, method='weighted')
dendrogram(data_link, labels=geneIds)

hierarchical_clustering(geneId_to_expressionProfile, 3)





plt.ylabel("Distances")
plt.xlabel("Genes")
plt.show()
