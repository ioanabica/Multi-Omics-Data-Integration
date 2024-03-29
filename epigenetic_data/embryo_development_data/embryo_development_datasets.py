import numpy as np


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

    if gene_expressions_sum != 0:
        for index in range(len(gene_expressions)):
            normalized_gene_expressions[index] = float(gene_expressions[index])/gene_expressions_sum

    return normalized_gene_expressions

def normalise_data(gene_expressions):
    """
    Normalizes the gene expressions profile to obtain a probability distribution which will be used as the input
    to the neural network architectures.

    :param (list) gene_expressions :  The un-normalized gene expression profile for a training example
    :return (list): normalized_gene_expressions: The normalized gene expression profile for a training
             example
    """

    gene_expressions = np.array(gene_expressions)
    mean = np.mean(gene_expressions)
    variance = np.var(gene_expressions)
    if variance!=0:
        gene_expressions = (gene_expressions - mean) / variance

    #max = np.max(gene_expressions)
    #min = np.min(gene_expressions)
    #gene_expressions = (gene_expressions - min) / (max-min)

    return gene_expressions


def extract_training_validation_test_embryo_ids(embryoStage_to_embryoIds):
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
        training_embryo_ids, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """
    Creates a training dataset that contain the training data and the training labels.

    :param (list) training_embryo_ids:
    :param (integer) input_data_size:
    :param (integer) output_size
    :param (dictionary) embryo_id_to_gene_expressions:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """

    training_dataset = dict()

    training_data = np.ndarray(shape=(len(training_embryo_ids), input_data_size),
                               dtype=np.float32)
    training_labels = np.ndarray(shape=(len(training_embryo_ids), output_size),
                                 dtype=np.float32)
    np.random.shuffle(training_embryo_ids)
    index = 0
    for embryoId in training_embryo_ids:
        training_data[index, :] = normalise_data(embryo_id_to_gene_expressions[embryoId])
        training_labels[index, :] = embryo_stage_to_one_hot_encoding[embryo_id_to_embryo_stage[embryoId]]
        index += 1

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset


def create_validation_dataset(
        validation_embryo_ids, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """

    :param (list) validation_embryo_ids:
    :param (integer) input_data_size:
    :param (integer) output_size
    :param (dictionary) embryo_id_to_gene_expressions:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """

    validation_dataset = dict()
    validation_data = np.ndarray(shape=(len(validation_embryo_ids), input_data_size),
                                 dtype=np.float32)
    validation_labels = np.ndarray(shape=(len(validation_embryo_ids), output_size),
                                   dtype=np.float32)

    np.random.shuffle(validation_embryo_ids)
    index = 0
    for embryoId in validation_embryo_ids:
        validation_data[index, :] = normalise_data(embryo_id_to_gene_expressions[embryoId])
        validation_labels[index, :] = embryo_stage_to_one_hot_encoding[embryo_id_to_embryo_stage[embryoId]]
        index += 1

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_test_dataset(
        test_embryoIds, input_data_size, output_size,
        embryoId_to_geneExpressions, embryoStage_to_oneHotEncoding, embryoId_to_embryoStage):

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
        test_data[index, :] = normalise_data(embryoId_to_geneExpressions[embryoId])
        test_labels[index, :] = embryoStage_to_oneHotEncoding[embryoId_to_embryoStage[embryoId]]
        index += 1

    test_dataset["test_data"] = test_data
    test_dataset["test_labels"] = test_labels

    return test_dataset


def create_training_dataset_with_clusters(
        training_embryo_ids, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """

    :param (list) training_embryo_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) embryo_id_to_gene_expressions_clusters:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """
    training_dataset = dict()

    training_data = dict()

    for cluster_id in range(len(clusters_size)):
        training_data[cluster_id] = np.ndarray(shape=(len(training_embryo_ids), clusters_size[cluster_id]),
                                               dtype=np.float32)

    training_labels = np.ndarray(shape=(len(training_embryo_ids), output_size),
                                 dtype=np.float32)

    np.random.shuffle(training_embryo_ids)
    index = 0

    for embryoId in training_embryo_ids:
        for cluster_id in range(len(clusters_size)):
            training_data[cluster_id][index, :] = \
                normalise_data(embryo_id_to_gene_expressions_clusters[embryoId][cluster_id])
        training_labels[index, :] = embryo_stage_to_one_hot_encoding[embryo_id_to_embryo_stage[embryoId]]
        index += 1

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset


def create_validation_dataset_with_clusters(
        validation_embryo_ids, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """

    :param (list) validation_embryo_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) embryo_id_to_gene_expressions_clusters:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """
    validation_dataset = dict()

    validation_data = dict()

    for cluster_id in range(len(clusters_size)):
        validation_data[cluster_id] = np.ndarray(shape=(len(validation_embryo_ids), clusters_size[cluster_id]),
                                                 dtype=np.float32)

    validation_labels = np.ndarray(shape=(len(validation_embryo_ids), output_size),
                                   dtype=np.float32)

    np.random.shuffle(validation_embryo_ids)
    index = 0
    for embryoId in validation_embryo_ids:
        for cluster_id in range(len(clusters_size)):
            validation_data[cluster_id][index, :] = \
                normalise_data(embryo_id_to_gene_expressions_clusters[embryoId][cluster_id])
        validation_labels[index, :] = embryo_stage_to_one_hot_encoding[embryo_id_to_embryo_stage[embryoId]]
        index += 1

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_test_dataset_with_clusters(
        test_embryo_ids, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """

    :param (list) test_embryo_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) embryo_id_to_gene_expressions_clusters:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """
    testing_dataset = dict()

    testing_data = dict()

    for cluster_id in range(len(clusters_size)):
        testing_data[cluster_id] = np.ndarray(shape=(len(test_embryo_ids), clusters_size[cluster_id]),
                                              dtype=np.float32)

    training_labels = np.ndarray(shape=(len(test_embryo_ids), output_size),
                                 dtype=np.float32)

    np.random.shuffle(test_embryo_ids)
    index = 0
    for embryoId in test_embryo_ids:
        for cluster_id in range(len(clusters_size)):
            testing_data[cluster_id][index, :] = \
                normalise_data(embryo_id_to_gene_expressions_clusters[embryoId][cluster_id])
        training_labels[index, :] = embryo_stage_to_one_hot_encoding[embryo_id_to_embryo_stage[embryoId]]
        index += 1

    testing_dataset["test_data"] = testing_data
    testing_dataset["test_labels"] = training_labels

    return testing_dataset


def create_k_fold_embryo_ids(k, embryo_stage_to_embryo_ids):
    """
    Separates the embryo_ids into k folds. (k-1) folds will be used to training and one fold for validation.

    :param k: number of folds
    :param embryo_stage_to_embryo_ids:
    :return:
    """
    k_fold_embryoIds = dict()
    for index in range(k):
        k_fold_embryoIds[index] = []

    embryoStages = embryo_stage_to_embryo_ids.keys()
    for embryoStage in embryoStages:
        embryoIds = embryo_stage_to_embryo_ids[embryoStage]
        if len(embryoIds) < k:
            for index in range(len(embryoIds)):
                k_fold_embryoIds[index] += [embryoIds[index]]
        else:
            group_size = len(embryoIds)/k
            for index in range(k-1):
                k_fold_embryoIds[index] += embryoIds[index*group_size:(index+1)*group_size]
            k_fold_embryoIds[k-1] += embryoIds[(k-1)*group_size:]

    keys = k_fold_embryoIds.keys()
    for key in keys:
        embryoIds = k_fold_embryoIds[key]
        np.random.shuffle(embryoIds)
        k_fold_embryoIds[key] = embryoIds

    return k_fold_embryoIds


def create_k_fold_datasets(
        k, k_fold_embryo_ids, input_data_size, output_size,
        embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """
    Creates the datasets corresponding to each fold.

    :param k:
    :param k_fold_embryo_ids:
    :param (integer) input_data_size:
    :param (integer) output_size:
    :param (dictionary) embryo_id_to_gene_expressions:
    :param (dictionary) embryo_stage_to_one_hot_encoding:
    :param (dictionary) embryo_id_to_embryo_stage:
    :return:
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_embryo_ids = k_fold_embryo_ids[index_i]
        training_embryo_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_embryo_ids += k_fold_embryo_ids[index_j]

        training_dataset = create_training_dataset(
            training_embryo_ids, input_data_size, output_size,
            embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

        validation_dataset = create_validation_dataset(
            validation_embryo_ids, input_data_size, output_size,
            embryo_id_to_gene_expressions, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets


def create_k_fold_datasets_with_clusters(
        k, k_fold_embryo_ids, clusters_size, output_size,
        embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage):
    """
    Creates the datasets corresponding to each fold.

    :param k:
    :param k_fold_embryo_ids:
    :param clusters_size:
    :param output_size:
    :param embryo_id_to_gene_expressions_clusters:
    :param embryo_stage_to_one_hot_encoding:
    :param embryo_id_to_embryo_stage:
    :return:
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_embryo_ids = k_fold_embryo_ids[index_i]
        training_embryo_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_embryo_ids += k_fold_embryo_ids[index_j]

        training_dataset = create_training_dataset_with_clusters(
            training_embryo_ids, clusters_size, output_size,
            embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

        validation_dataset = create_validation_dataset_with_clusters(
            validation_embryo_ids, clusters_size, output_size,
            embryo_id_to_gene_expressions_clusters, embryo_stage_to_one_hot_encoding, embryo_id_to_embryo_stage)

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets


