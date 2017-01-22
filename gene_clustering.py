import numpy as np
import matplotlib.pyplot as plt
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


        # update distance matrix
        min_cluster_size = len(clusters[min_index])
        max_cluster_size = len(clusters[max_index])

        for index_i in range(num_clusters):
            #distance_matrix[min_index][index_i] = \
                #(distance_matrix[min_index][index_i] + distance_matrix[max_index][index_i])/2

            # average (UPGMA)
            distance_matrix[min_index][index_i] = \
                ((distance_matrix[min_index][index_i] * min_cluster_size) +
                 (distance_matrix[max_index][index_i] * max_cluster_size))/(min_cluster_size + max_cluster_size)

            distance_matrix[index_i][min_index] = distance_matrix[min_index][index_i]

        distance_matrix = np.delete(distance_matrix, max_index, 0)
        distance_matrix = np.delete(distance_matrix, max_index, 1)

        # combine clusters
        clusters[min_index] += clusters[max_index]
        del clusters[max_index]

        num_clusters -= 1

    gene_id_to_cluster_id = dict()
    for index_i in range(num_clusters):
        for gene in clusters[index_i]:
            gene_id_to_cluster_id[gene] = index_i

    plot_dendogram(geneId_to_expressionProfile)

    return gene_id_to_cluster_id, clusters


def plot_dendogram(geneId_to_expressionProfile):
    geneExpressions = []
    geneIds = geneId_to_expressionProfile.keys()
    for geneId in geneIds:
        geneExpressions += [geneId_to_expressionProfile[geneId]]

    distance_matrix = pdist(geneExpressions, metric='correlation')
    dendrogram(linkage(distance_matrix, method='average'), labels=geneIds)

    plt.ylabel("Distances")
    plt.xlabel("Genes")
    plt.show()







