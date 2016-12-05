import math
import numpy as np

# Set the gene entropy threshold for selecting the gene
gene_entropy_threshold = 6.2
# Number of k folds
k = 6


def compute_probability_distribution(gene_expressions):
    """
    Normalize the gene expressions to obtain a probability distribution

    :param gene_expressions:
    :return:
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

    :param gene_expressions:
    :return:
    """
    gene_entropy = 0.0
    for gene_expression in gene_expressions:
        if float(gene_expression) > 0.0:
            gene_entropy -= float(gene_expression) * math.log(float(gene_expression), 2)

    return gene_entropy


def extract_embryoId_to_embryoStage(file):
    """
    Create a dictionary from an embryoId to the corresponding development stage

    :param file:
    :return:
    """
    embryoId_to_embryoStage = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        embryoId_to_embryoStage[line_elements[0]] = line_elements[1]
    return embryoId_to_embryoStage


def extract_embryoStage_to_embryoIds(file):
    """
    Create a dictionary from an embryo development stage to a list of the corresponding embroyIds

    :param file:
    :return:
    """
    embryoStage_to_embryoIds = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        if line_elements[1] in embryoStage_to_embryoIds.keys():
            embryoStage_to_embryoIds[line_elements[1]] += [line_elements[0]]
        else:
            embryoStage_to_embryoIds[line_elements[1]] = [line_elements[0]]
    return embryoStage_to_embryoIds


def extract_geneId_to_geneEntorpy(file):
    """

    :param file:
    :return:
    """
    geneId_to_geneEntropy = dict()
    file.readline()
    for line in file:
        line_elements = line.split()
        gene_entropy = compute_gene_entropy(compute_probability_distribution(line_elements[1:]))
        geneId_to_geneEntropy[line_elements[0]] = gene_entropy

    return geneId_to_geneEntropy

def extract_embryoId_to_geneExpressions (file, geneId_to_geneEntropy):
    """

    :param file:
    :param geneId_to_geneEntropy:
    :return:
    """
    embryoId_to_geneExpressions = dict()

    #read first line and create an entry in the dictionary for each embryoId
    embryoIds = (file.readline()).split()
    embryoIds = embryoIds[1:]

    for embryoId in embryoIds:
        embryoId_to_geneExpressions[embryoId] = []

    for line in file:
        line_elements = line.split()
        if (geneId_to_geneEntropy[line_elements[0]] > gene_entropy_threshold) & \
                (len(line_elements) == len(embryoIds) + 1):
            for index in range(len(embryoIds)):
                embryoId_to_geneExpressions[embryoIds[index]] += [line_elements[index+1]]

    print len(embryoId_to_geneExpressions['GSM896803'])
    return embryoId_to_geneExpressions

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
    print "Training Data"
    index = 0
    for embryoId in training_embryoIds:
        print embryoId
        training_data[index, :] = compute_probability_distribution(embryoId_to_geneExpressions[embryoId])
        print embryoId_to_geneExpressions[embryoId]
        training_labels[index, :] = embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]
        print training_labels[index, :]
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

def create_k_folds_embryoIds(k, embryoStage_to_embryoIds):
    """

    :param k:
    :param embryoStage_to_embryoIds:
    :return:
    """
    k_folds_embryoIds = dict()
    for index in range(k):
        k_folds_embryoIds[index] = []

    embryoStages = embryoStage_to_embryoIds.keys()
    for embryoStage in embryoStages:
        embryoIds = embryoStage_to_embryoIds[embryoStage]
        np.random.shuffle(embryoIds)
        if len(embryoIds) < k:
            for index in range(len(embryoIds)):
                k_folds_embryoIds[index] += [embryoIds[index]]
        else:
            group_size = len(embryoIds)/k
            for index in range(k-1):
                k_folds_embryoIds[index] += embryoIds[index*group_size:(index+1)*group_size]
            k_folds_embryoIds[k-1] += embryoIds[(k-1)*group_size:]

    return k_folds_embryoIds


def create_k_folds_datasets(
        k,
        k_folds_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage):
    """

    :param k:
    :param k_folds_embryoIds:
    :param input_data_size:
    :param output_size:
    :param embryoId_to_geneExpressions:
    :param embryoStage_to_oneHotEncoding:
    :param embryoId_to_embryoStage:
    :return:
    """

    k_folds_datasets = dict()
    for index in range(k):
        k_folds_datasets[index] = dict()

    for index_i in range(k):
        validation_embryoIds = k_folds_embryoIds[index_i]
        training_embryoIds = []
        for index_j in range(k):
            if index_j != index_i:
                training_embryoIds += k_folds_embryoIds[index_j]
        print index_i
        print validation_embryoIds
        print training_embryoIds

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

        k_folds_datasets[index_i]["training_dataset"] = training_dataset
        k_folds_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_folds_datasets

"""
Class that extracts the epigenetics dataset
"""

class EpigeneticsData(object):

    gene_expressions_file = open('data/epigenetics_data/human_early_embryo_gene_expression.txt', 'r')
    embryo_stage_file = open('data/epigenetics_data/human_early_embryo_stage.txt', 'r')

    embryoId_to_embryoStage = extract_embryoId_to_embryoStage(embryo_stage_file)
    embryo_stage_file.seek(0)
    embryoStage_to_embryoIds = extract_embryoStage_to_embryoIds(embryo_stage_file)

    geneId_to_geneEntropy = extract_geneId_to_geneEntorpy(gene_expressions_file)
    gene_expressions_file.seek(0)
    embryoId_to_geneExpressions = extract_embryoId_to_geneExpressions(gene_expressions_file, geneId_to_geneEntropy)

    embryoIds = embryoId_to_embryoStage.keys()
    input_data_size = len(embryoId_to_geneExpressions[embryoIds[0]])

    gene_expressions_file.close()
    embryo_stage_file.close()

    embryoStages = embryoStage_to_embryoIds.keys()
    embryoStage_to_oneHotEncoding = create_oneHotEncoding(embryoStages)
    print embryoStage_to_oneHotEncoding
    output_size = len(embryoStages)

    training_embryoIds, validation_embryoIds, test_embryoIds = \
        extract_training_validation_test_embryoIds(embryoStage_to_embryoIds)


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

    test_dataset = create_test_dataset(
        test_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage)

    k_folds_embryoIds = create_k_folds_embryoIds(k, embryoStage_to_embryoIds)
    print k_folds_embryoIds

    k_folds_datasets = create_k_folds_datasets(
        k,
        k_folds_embryoIds,
        input_data_size,
        output_size,
        embryoId_to_geneExpressions,
        embryoStage_to_oneHotEncoding,
        embryoId_to_embryoStage)


