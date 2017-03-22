import numpy as np
import math
from scipy.spatial.distance import pdist, squareform, euclidean, correlation

epsilon = 0.00001


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
    print "using k-means clustering"

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

