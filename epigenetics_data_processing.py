import math
import numpy as np
from gene_clustering import hierarchical_clustering
from epigenetics_datasets import *

# Set the gene entropy threshold for selecting the gene
""" To obtain a cluster with 128 genes set gene_entropy_treshold = 6.1 and max_num_genes = 250"""
gene_entropy_threshold = 6.2
max_num_genes = 256
# Number of k folds
k = 6


def compute_gene_entropy(gene_expressions):
    """
    Computes the entropy of the gene using the formula:
            entropy = sum_i (- g_i * log(g_i))
    where g_i is the expression level of the gene in experiment i

    The entropy of the gene is useful in determining which genes change their values a lot over the stages of
    embryonic development.

    :param (list) gene_expressions: an array containing the gene expression profile
    :return (float): gene_entropy: a float representing the entropy of the gene expression profile

    """
    gene_entropy = 0.0
    for gene_expression in gene_expressions:
        if float(gene_expression) > 0.0:
            gene_entropy -= float(gene_expression) * math.log(float(gene_expression), 2)

    return gene_entropy


def extract_embryo_id_to_embryo_stage(data_file):
    """
    Create a dictionary from an embryo_id to the corresponding development stage.
    The data is extracted from the input file.

    :param (file) data_file
    :return (dictionary): embryo_id_to_embryo_stage
    """
    embryo_id_to_embryo_stage = dict()
    data_file.readline()
    for line in data_file:
        line_elements = line.split()
        embryo_id_to_embryo_stage[line_elements[0]] = line_elements[1]
    return embryo_id_to_embryo_stage


def extract_embryo_stage_to_embryo_ids(data_file):
    """
    Create a dictionary that maps the embryo development stage to a list of corresponding embryo_ids
    whose gene expression profile was measured at this stage.
    The data is extracted from the input file.

    :param (file) data_file
    :return (dictionary): embryo_stage_to_embryo_ids
    """

    embryo_stage_to_embryo_ids = dict()
    data_file.readline()
    for line in data_file:
        line_elements = line.split()
        if line_elements[1] in embryo_stage_to_embryo_ids.keys():
            embryo_stage_to_embryo_ids[line_elements[1]] += [line_elements[0]]
        else:
            embryo_stage_to_embryo_ids[line_elements[1]] = [line_elements[0]]
    return embryo_stage_to_embryo_ids


def extract_gene_id_to_gene_entropy_and_expression_profile(data_file):
    """
    Creates two dictionaries: one dictionary that maps the gene_id to its corresponding gene entropy and
                             one dictionary that maps the gene_id to its corresponding expression_profile
    The data is extracted from the input file.

    :param (file) data_file
    :return (dictionary, dictionary): gene_id_to_gene_entropy, gene_id_to_expression_profile
    """
    gene_id_to_gene_entropy = dict()
    gene_id_to_expression_profile = dict()

    data_file.readline()
    num_genes = 0
    for line in data_file:
        line_elements = line.split()
        gene_entropy = compute_gene_entropy(compute_probability_distribution(line_elements[1:]))
        gene_id_to_gene_entropy[line_elements[0]] = gene_entropy

        if (gene_entropy > gene_entropy_threshold) & (num_genes < max_num_genes):
            num_genes += 1
            gene_id_to_expression_profile[line_elements[0]] = line_elements[1:]

    return gene_id_to_gene_entropy, gene_id_to_expression_profile


def extract_embryo_id_to_gene_expressions(data_file, gene_id_to_gene_entropy, gene_entropy_threshold, max_num_genes):
    """
    Creates a dictionary that maps each embryo_id to the corresponding list of gene expression levels.
    The order of the genes whose expression levels are in the list is the same for every embryo.

    A training example for the feedforward neural network and the recurrent neural network consists of an embryo
    and the input data to the networks is represented by the corresponding gene expressions.

    The data is extracted from the input file.

    :param (file) data_file
    :param (dictionary) gene_id_to_gene_entropy: the entropy of each gene.
    :param (float) gene_entropy_threshold: the minimum entropy a gene needs to have in order to be selected to be part
                                           of the input data to the neural networks
    :param (integer) max_num_genes: the maximum number of genes whose expression levels can be used as
                                    inputs to the neural networks
    :return (dictionary): embryo_id_to_gene_expressions
    """

    embryo_id_to_gene_expressions = dict()

    """ Read the first line of the input file and create an entry in the dictionary for each embryo_id. """
    embryo_ids = (data_file.readline()).split()
    embryo_ids = embryo_ids[1:]

    for embryo_id in embryo_ids:
        embryo_id_to_gene_expressions[embryo_id] = []

    num_genes = 0
    for line in data_file:
        line_elements = line.split()
        if (gene_id_to_gene_entropy[line_elements[0]] > gene_entropy_threshold) & (num_genes < max_num_genes) & \
                (len(line_elements) == len(embryo_ids) + 1):
            num_genes += 1
            for index in range(len(embryo_ids)):
                embryo_id_to_gene_expressions[embryo_ids[index]] += [line_elements[index + 1]]

    for embryo_id in embryo_ids:
        embryo_id_to_gene_expressions[embryo_id] = \
            compute_probability_distribution(embryo_id_to_gene_expressions[embryo_id])

    print len(embryo_id_to_gene_expressions['GSM896803'])

    return embryo_id_to_gene_expressions


def extract_embryo_id_to_gene_expressions_clusters(data_file, gene_id_to_cluster_id):
    """
    Creates a dictionary that maps each embryo_id to the corresponding dictionary that contains the mapping from
    the cluster_id to the gene expression levels in the cluster. For example, by considering 2 embryos and 2 clusters,
    where the first cluster consists of gene_1 and gene_2 and the second cluster consists of gene_3 and gene_4,
    the embryo_id_to_gene_expressions_clusters_dictionary will have the form:

    {'embryo_1': {'0': [normalized_expressions_level_gene_1, normalized_expression_level_gene_2],
    '1': [normalized_expression_level_gene_3, normalized_expression_level_gene_4]},
    'embryo_2': {'0': [normalized_expression_level_gene_1, normalized_expression_level_gene_2],
    '1': [normalized_expression_level_gene_3, normalized_expression_level_gene_4]}

    The order of the genes whose expression levels are in the list is the same for every cluster.

    A training example for the superlayered neural network consists of an embryo
    and the input data to each feedforward neural netowkr in the superlayered architecture is represented by
    the corresponding gene expressions for the corresponding cluster.

    The data is extracted from the input file.

    :param (file) data_file
    :param (dictionary) gene_id_to_cluster_id: the cluster assignments for each gene
    :return (dictionary): embryo_id_to_gene_expressions_clusters
    """

    embryo_id_to_gene_expressions_clusters = dict()
    gene_ids = gene_id_to_cluster_id.keys()
    max_cluster_id = max(gene_id_to_cluster_id.values()) + 1

    """ Read the first line of the input file and create an entry in the dictionary for each embryo_id.
        Then, for each embryo_id, create an entry in their dictionary for each cluster_id."""
    embryo_ids = (data_file.readline()).split()
    embryo_ids = embryo_ids[1:]

    for embryo_id in embryo_ids:
        embryo_id_to_gene_expressions_clusters[embryo_id] = dict()
        for cluster_id in range(max_cluster_id):
            embryo_id_to_gene_expressions_clusters[embryo_id][cluster_id] = []


    for line in data_file:
        line_elements = line.split()
        if (line_elements[0] in gene_ids) & (len(line_elements) == len(embryo_ids) + 1):
            cluster_id = gene_id_to_cluster_id[line_elements[0]]
            for index in range(len(embryo_ids)):
                embryo_id_to_gene_expressions_clusters[embryo_ids[index]][cluster_id] += \
                    [line_elements[index + 1]]

    for embryo_id in embryo_ids:
        for cluster_id in range(max_cluster_id):
            embryo_id_to_gene_expressions_clusters[embryo_id][cluster_id] = \
                compute_probability_distribution(embryo_id_to_gene_expressions_clusters[embryo_id][cluster_id])

    return embryo_id_to_gene_expressions_clusters


def create_one_hot_encoding(embryo_stages):
    """
    Creates a dictionary that contains a mapping from each embryo_stage to a distinct one hot encoding. This
    represents the output label for each training exampel.

    :param (dictionary) embryo_stages: the embryonic development stages that the neural networks are predicting
    :return (dictionary): embryo_stage_to_one_hot_encoding
    """
    embryo_stage_to_one_hot_encoding = dict()

    for index in range(len(embryo_stages)):
        one_hot_encoding = [0.0]*len(embryo_stages)
        one_hot_encoding[index] = 1.0
        embryo_stage_to_one_hot_encoding[embryo_stages[index]] = one_hot_encoding

    return embryo_stage_to_one_hot_encoding


def compute_clusters_size(gene_clusters):
    """
    Computs the number of genes in each cluster.

    :param (list) gene_clusters: a list containing the clusters
    :return (list): clusters_size: a list containing the cluster sizes
    """
    clusters_size = []
    for index in range(len(gene_clusters)):
        clusters_size.append(len(gene_clusters[index]))
    return clusters_size


def extract_training_validation_test_embryo_ids(embryo_stage_to_embryo_ids):
    """
    Deprecated: use k-fold cross validation instead

    :param embryo_stage_to_embryo_ids:
    :return: training_embryo_ids
    :return: validation_embryo_ids
    :return: test_embryo_ids
    """
    training_embryo_ids = []
    validation_embryo_ids = []
    test_embryo_ids = []

    embryoStages = embryo_stage_to_embryo_ids.keys()
    for embryoStage in embryoStages:
        embryoIds = embryo_stage_to_embryo_ids[embryoStage]
        if len(embryoIds) < 6:
            test_embryo_ids += [embryoIds[0]]
            validation_embryo_ids += [embryoIds[1]]
            training_embryo_ids += embryoIds[2:]
        else:
            test_embryo_ids += embryoIds[0:2]
            validation_embryo_ids += embryoIds[2:4]
            training_embryo_ids += embryoIds[4:]

    return training_embryo_ids, validation_embryo_ids, test_embryo_ids


"""
Class that extracts the data to perform supervised learning using the neural network architectures.
"""

class EpigeneticsData(object):

    gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
    embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')

    embryo_id_to_embryo_stage = extract_embryo_id_to_embryo_stage(embryo_stage_file)
    embryo_stage_file.seek(0)
    embryo_stage_to_embryo_ids = extract_embryo_stage_to_embryo_ids(embryo_stage_file)

    geneId_to_gene_entropy, geneId_to_expressionProfile = \
        extract_gene_id_to_gene_entropy_and_expression_profile(gene_expressions_file)

    gene_expressions_file.seek(0)
    embryo_id_to_gene_expressions = extract_embryo_id_to_gene_expressions(
        gene_expressions_file, geneId_to_gene_entropy, gene_entropy_threshold, max_num_genes)

    gene_expressions_file.seek(0)
    gene_id_to_gene_cluster, gene_clusters = hierarchical_clustering(geneId_to_expressionProfile, 2)
    embryo_id_to_gene_expressions_clusters = extract_embryo_id_to_gene_expressions_clusters(
        gene_expressions_file, gene_id_to_gene_cluster)

    gene_expressions_file.close()
    embryo_stage_file.close()

    embryoIds = embryo_id_to_embryo_stage.keys()
    input_data_size = len(embryo_id_to_gene_expressions[embryoIds[0]])

    clusters_size = compute_clusters_size(gene_clusters)
    print clusters_size


    embryo_stages = embryo_stage_to_embryo_ids.keys()
    embryo_stage_to_one_hot_encoding = create_one_hot_encoding(embryo_stages)
    output_size = len(embryo_stages)

    training_embryoIds, validation_embryoIds, test_embryoIds = \
        extract_training_validation_test_embryo_ids(embryo_stage_to_embryo_ids)

    training_embryoIds += test_embryoIds

    training_dataset = create_training_dataset(
        training_embryoIds, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

    training_dataset_cluster = create_training_dataset_with_clusters(
        training_embryoIds, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

    #print training_dataset_cluster

    validation_dataset = create_validation_dataset(
        validation_embryoIds, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

    validation_dataset_clusters = create_validation_dataset_with_clusters(
        validation_embryoIds, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

    #print validation_dataset_clusters


    test_dataset = create_test_dataset(
        test_embryoIds, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

    k_fold_embryoIds = create_k_fold_embryo_ids(k, embryo_stage_to_embryo_ids)

    k_fold_datasets = create_k_fold_datasets(
        k, k_fold_embryoIds, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)


    k_fold_datasets_with_clusters = create_k_fold_datasets_with_clusters(
        k, k_fold_embryoIds, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

