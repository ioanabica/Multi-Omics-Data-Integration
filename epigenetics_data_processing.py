import math
import numpy as np
from gene_clustering import hierarchical_clustering

# Set the gene entropy threshold for selecting the gene
gene_entropy_threshold = 6.0
max_num_genes = 8
# Number of k folds
k = 6


def compute_probability_distribution(gene_expressions):
    """
    Normalizes the gene expressions profile to obtain a probability distribution which will be used as the input
    to the neural network architectures.

    :param (list) gene_expressions :  The un-normalized gene expression profile for a training example
    :return (list): normalized_gene_expressions: The normalized gene expression profile for a training
             example
    """

    gene_expressions_sum = 0.0
    for gene_expression in gene_expressions:
        gene_expressions_sum += float(gene_expression)
    normalized_gene_expressions = range(len(gene_expressions))
    for index in range(len(gene_expressions)):
        normalized_gene_expressions[index] = float(gene_expressions[index])/gene_expressions_sum

    return normalized_gene_expressions


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
    Create a dictionary that maps the embryo development stage to a list of corresponding embryos
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

    print len(embryo_id_to_gene_expressions['GSM896803'])

    return embryo_id_to_gene_expressions


def extract_embryo_id_to_gene_expressions_clusters(data_file, gene_id_to_cluster_id):
    """
    Creates a dictionary that maps each embryo_id to the corresponding list of gene expression levels .
    The order of the genes whose expression levels are in the list is the same for every embryo.

    A training example for the superlayered neural network consists of an embryo
    and the input data to the networks is represented by the corresponding gene expressions for each gene cluster.

    The data is extracted from the input file.

    :param (file) data_file
    :param (dictionary) gene_id_to_cluster_id:
    :return (dictionary): embryo_id_to_gene_expressions
    """

    embryo_id_to_gene_expressions_clusters = dict()
    gene_ids = gene_id_to_cluster_id.keys()
    max_cluster_id = max(gene_id_to_cluster_id.values()) + 1
    print "max cluster"
    print max_cluster_id

    """ Read the first line of the input file and create an entry in the dictionary for each embryo_id. """
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
                embryo_id_to_gene_expressions_clusters[embryo_ids[index]][cluster_id] += [line_elements[index + 1]]

    print embryo_id_to_gene_expressions_clusters

    return embryo_id_to_gene_expressions_clusters


def create_oneHotEncoding(embryoStages):
    """

    :param embryoStages:
    :return:
    """
    embryoStage_to_oneHotEncoding = dict()

    for index in range(len(embryoStages)):
        oneHotEncoding = [0.0]*len(embryoStages)
        oneHotEncoding[index] = 1.0
        embryoStage_to_oneHotEncoding[embryoStages[index]] = oneHotEncoding

    return embryoStage_to_oneHotEncoding


def extract_training_validation_test_embryoIds(embryoStage_to_embryoIds):
    """

    :param embryoStage_to_embryoIds:
    :return:
    """
    training_embryoIds = []
    validation_embryoIds = []
    test_embryoIds = []
    embryoStages = embryoStage_to_embryoIds.keys()
    for embryoStage in embryoStages:
        embryoIds = embryoStage_to_embryoIds[embryoStage]
        if len(embryoIds) < 6:
            test_embryoIds += [embryoIds[0]]
            validation_embryoIds += [embryoIds[1]]
            training_embryoIds += embryoIds[2:]
        else:
            test_embryoIds += embryoIds[0:2]
            validation_embryoIds += embryoIds[2:4]
            training_embryoIds += embryoIds[4:]

    return training_embryoIds, validation_embryoIds, test_embryoIds


def create_training_dataset(
        training_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage):
    """

    :param training_embryoIds:
    :param input_data_size:
    :param output_size:
    :param embryoId_to_geneExpressions:
    :param embryoStage_to_oneHotEncoding:
    :param embryoId_to_embryoStage:
    :return:
    """

    training_dataset = dict()

    training_data = np.ndarray(shape=(len(training_embryoIds), input_data_size),
                               dtype=np.float32)
    training_labels = np.ndarray(shape=(len(training_embryoIds), output_size),
                                 dtype=np.float32)
    np.random.shuffle(training_embryoIds)
    index = 0
    for embryoId in training_embryoIds:
        training_data[index, :] = compute_probability_distribution(embryoId_to_geneExpressions[embryoId])
        training_labels[index, :] = embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]
        index += 1

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset

def create_validation_dataset(
        validation_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage):

    """

    :param validation_embryoIds:
    :param input_data_size:
    :param output_size:
    :param embryoId_to_geneExpressions:
    :param embryoStage_to_oneHotEncoding:
    :param embryoId_to_embryoStage:
    :return:
    """

    validation_dataset = dict()
    validation_data = np.ndarray(shape=(len(validation_embryoIds), input_data_size),
                                 dtype=np.float32)
    validation_labels = np.ndarray(shape=(len(validation_embryoIds), output_size),
                                   dtype=np.float32)

    np.random.shuffle(validation_embryoIds)
    index = 0
    for embryoId in validation_embryoIds:
        validation_data[index, :] = compute_probability_distribution(embryoId_to_geneExpressions[embryoId])
        validation_labels[index, :] = embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]
        index += 1

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_test_dataset(
        test_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage):

    """

    :param test_embryoIds:
    :param input_data_size:
    :param output_size:
    :param embryoId_to_geneExpressions:
    :param embryoStage_to_oneHotEncoding:
    :param embryoId_to_embryoStage:
    :return:
    """

    test_dataset = dict()
    # create test data
    test_data = np.ndarray(shape=(len(test_embryoIds), input_data_size),
                           dtype=np.float32)
    test_labels = np.ndarray(shape=(len(test_embryoIds), output_size),
                             dtype=np.float32)

    np.random.shuffle(test_embryoIds)
    index = 0
    for embryoId in test_embryoIds:
        test_data[index, :] = compute_probability_distribution(embryoId_to_geneExpressions[embryoId])
        test_labels[index, :] = embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]
        index += 1

    test_dataset["test_data"] = test_data
    test_dataset["test_labels"] = test_labels

    return test_dataset

def create_k_fold_embryoIds(k, embryoStage_to_embryoIds):
    """

    :param k:
    :param embryoStage_to_embryoIds:
    :return:
    """
    k_fold_embryoIds = dict()
    for index in range(k):
        k_fold_embryoIds[index] = []

    embryoStages = embryoStage_to_embryoIds.keys()
    for embryoStage in embryoStages:
        embryoIds = embryoStage_to_embryoIds[embryoStage]
        if len(embryoIds) < k:
            for index in range(len(embryoIds)):
                k_fold_embryoIds[index] += [embryoIds[index]]
        else:
            group_size = len(embryoIds)/k
            for index in range(k-1):
                k_fold_embryoIds[index] += embryoIds[index*group_size:(index+1)*group_size]
            k_fold_embryoIds[k-1] += embryoIds[(k-1)*group_size:]

    return k_fold_embryoIds


def create_k_fold_datasets(
        k,
        k_fold_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage):
    """

    :param k:
    :param k_fold_embryoIds:
    :param input_data_size:
    :param output_size:
    :param embryoId_to_geneExpressions:
    :param embryoStage_to_oneHotEncoding:
    :param embryoId_to_embryoStage:
    :return:
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_embryoIds = k_fold_embryoIds[index_i]
        training_embryoIds = []
        for index_j in range(k):
            if index_j != index_i:
                training_embryoIds += k_fold_embryoIds[index_j]

        training_dataset = create_training_dataset(
            training_embryoIds,
            input_data_size,
            output_size,
            embryoId_to_geneExpressions,
            embryoStage_to_oneHotEncoding,
            embryoId_to_embryoStage)

        validation_dataset = create_validation_dataset(
            validation_embryoIds,
            input_data_size,
            output_size,
            embryoId_to_geneExpressions,
            embryoStage_to_oneHotEncoding,
            embryoId_to_embryoStage)

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets

"""
Class that extracts the epigenetics dataset
"""

class EpigeneticsData(object):

    gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
    embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')

    embryo_id_to_embryo_stage = extract_embryo_id_to_embryo_stage(embryo_stage_file)
    embryo_stage_file.seek(0)
    embryoStage_to_embryoIds = extract_embryo_stage_to_embryo_ids(embryo_stage_file)

    geneId_to_geneEntropy, geneId_to_expressionProfile = extract_gene_id_to_gene_entropy_and_expression_profile(gene_expressions_file)
    gene_expressions_file.seek(0)
    embryoId_to_geneExpressions = extract_embryo_id_to_gene_expressions(
        gene_expressions_file, geneId_to_geneEntropy, gene_entropy_threshold, max_num_genes)

    gene_expressions_file.seek(0)
    gene_id_to_gene_cluster, gene_clusters = hierarchical_clustering(geneId_to_expressionProfile, 3)
    embryoId_to_geneExpressions_clusters = extract_embryo_id_to_gene_expressions_clusters(
        gene_expressions_file, gene_id_to_gene_cluster)

    embryoIds = embryo_id_to_embryo_stage.keys()
    input_data_size = len(embryoId_to_geneExpressions[embryoIds[0]])

    gene_expressions_file.close()
    embryo_stage_file.close()




    embryoStages = embryoStage_to_embryoIds.keys()
    embryoStage_to_oneHotEncoding = create_oneHotEncoding(embryoStages)
    output_size = len(embryoStages)

    training_embryoIds, validation_embryoIds, test_embryoIds = \
        extract_training_validation_test_embryoIds(embryoStage_to_embryoIds)


    training_embryoIds += test_embryoIds

    training_dataset = create_training_dataset(
        training_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryo_id_to_embryo_stage)

    validation_dataset = create_validation_dataset(
        validation_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryo_id_to_embryo_stage)

    test_dataset = create_test_dataset(
        test_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryo_id_to_embryo_stage)

    k_fold_embryoIds = create_k_fold_embryoIds(k, embryoStage_to_embryoIds)
    print k_fold_embryoIds

    k_fold_datasets = create_k_fold_datasets(
        k,
        k_fold_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryo_id_to_embryo_stage)


