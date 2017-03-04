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


def centers_convergence(new_centers, centers):
    convergence = True

    for centers_index in range(len(centers)):
        for elements_index in range(len(centers[0])):
            if (math.fabs(new_centers[centers_index][elements_index] - centers[centers_index][elements_index])) > epsilon:
                convergence = False

    return convergence


def compute_closest_center(centers, gene_expression_levels):
    min_distance = 300000
    closest_center = -1

    for index in range(len(gene_expression_levels)):
        gene_expression_levels[index] = float(gene_expression_levels[index])

    for index in range(len(centers)):
        distance = correlation(centers[index], gene_expression_levels)
        if(distance < min_distance - epsilon):
            min_distance = distance
            closest_center = index
    return closest_center


def compute_cluster_assignments(gene_ids, gene_id_to_expression_levels, centers):
    gene_id_to_cluster_id = dict()
    cluster_assignments = [[] for i in range(len(centers))]
    for gene_id in gene_ids:
        closest_center = compute_closest_center(centers, gene_id_to_expression_levels[gene_id])
        gene_id_to_cluster_id[gene_id] = closest_center
        cluster_assignments[closest_center].append(gene_id)

    return gene_id_to_cluster_id, cluster_assignments


def compute_new_centers(clusters, gene_id_to_expression_levels):

    gene_expression_levels = gene_id_to_expression_levels.values()
    size_of_center = len(gene_expression_levels[0])
    centers = np.zeros(shape=(len(clusters), size_of_center))
    for index in range(len(clusters)):
        for gene_id in clusters[index]:
            centers[index] = np.add(centers[index], gene_id_to_expression_levels[gene_id])
        centers[index] = np.divide(centers[index], len(clusters[index]))

    return centers


def k_means_clustering(gene_id_to_expression_levels, num_clusters):

    gene_ids = gene_id_to_expression_levels.keys()
    gene_expressions = []
    for geneId in gene_ids:
        gene_expressions += [gene_id_to_expression_levels[geneId]]

    gene_centers = np.random.choice(gene_ids, num_clusters)

    centers = np.zeros(shape=(num_clusters, len(gene_expressions[0])), dtype='float32')
    for index in range(num_clusters):
        centers[index] = gene_id_to_expression_levels[gene_centers[index]]

    gene_id_to_clusters_id, cluster_assignments = \
        compute_cluster_assignments(gene_ids, gene_id_to_expression_levels, centers)
    new_centers = compute_new_centers(cluster_assignments, gene_id_to_expression_levels)

    while not centers_convergence(new_centers, centers):
        centers = new_centers
        gene_id_to_clusters_id, cluster_assignments = \
            compute_cluster_assignments(gene_ids, gene_id_to_expression_levels, centers)
        new_centers = compute_new_centers(cluster_assignments, gene_id_to_expression_levels)

    gene_id_to_clusters_id, cluster_assignments = \
        compute_cluster_assignments(gene_ids, gene_id_to_expression_levels, new_centers)

    return gene_id_to_clusters_id, cluster_assignments


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





