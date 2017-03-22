import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, euclidean, correlation
from scipy.cluster.hierarchy import linkage,dendrogram

epsilon = 0.00001

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


def hierarchical_clustering(gene_id_to_expression_levels, min_num_clusters):

    print 'using hierarchical clustering'

    gene_ids = gene_id_to_expression_levels.keys()
    gene_expressions = []
    for geneId in gene_ids:
        gene_expressions += [gene_id_to_expression_levels[geneId]]

    distance_matrix = squareform(pdist(gene_expressions, metric='correlation'))

    # create independent clusters, such that each gene is part of a single cluster
    clusters = list()
    for index in range(len(gene_ids)):
        clusters.append([gene_ids[index]])
    num_clusters = len(gene_ids)

    while(num_clusters > min_num_clusters):
        min_index, max_index = find_closest_clusters(distance_matrix)

        # update distance matrix
        min_cluster_size = len(clusters[min_index])
        max_cluster_size = len(clusters[max_index])

        for index_i in range(num_clusters):
            """Weighted
            distance_matrix[min_index][index_i] = \
                (distance_matrix[min_index][index_i] + distance_matrix[max_index][index_i])/2 """

            """ Average (UPGMA) """
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
            gene_id_to_cluster_id[gene] = index_i

    return gene_id_to_cluster_id, clusters


def plot_dendogram(gene_id_to_expression_levels):
    gene_expressions = []
    gene_ids = gene_id_to_expression_levels.keys()
    for gene_id in gene_ids:
        gene_expressions += [gene_id_to_expression_levels[gene_id]]

    distance_matrix = pdist(gene_expressions, metric='correlation')
    dendrogram(linkage(distance_matrix, method='weighted'), labels=gene_ids)

    plt.ylabel("Distance", fontsize=24)
    plt.xlabel("Gene Id", fontsize=24)
    plt.show()





