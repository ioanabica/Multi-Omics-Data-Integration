import numpy as np


def compute_probability_distribution(input_values):
    """
    Normalizes the gene expressions profile to obtain a probability distribution which will be used as the input
    to the neural network architectures.

    :param (list) input_values :  The un-normalized gene expression profile for a training example
    :return (list): normalized_input_values: The normalized gene expression profile for a training
             example
    """

    input_values_sum = 0.0

    for input_value in input_values:
        input_values_sum += float(input_value)
    normalized_input_values = range(len(input_values))

    if input_values_sum != 0:
        for index in range(len(input_values)):
            normalized_input_values[index] = float(input_values[index])/input_values_sum

    return normalized_input_values


def __extract_training_validation_test_patient_ids(labels_to_patient_ids):
    training_patient_ids = []
    validation_patient_ids = []
    test_patient_ids = []

    labels = labels_to_patient_ids.keys()
    for label in labels:
        patient_ids = labels_to_patient_ids[label]

        num_training_patients = len(patient_ids) * 70/100
        num_validation_patients = len(patient_ids) * 15/100

        training_patient_ids += patient_ids[:num_training_patients]
        validation_patient_ids += patient_ids[num_training_patients + 1 : num_training_patients + num_validation_patients]
        test_patient_ids += patient_ids[num_training_patients + num_validation_patients:]

    return training_patient_ids, validation_patient_ids, test_patient_ids


def create_training_dataset(
        training_patient_ids, input_data_size, output_size,
        patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label):
    """
    Creates a training dataset that contain the training data and the training labels.

    :param (list) training_patient_ids:
    :param (integer) input_data_size:
    :param (integer) output_size
    :param (dictionary) patient_id_to_input_values:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """

    training_dataset = dict()

    training_data = np.ndarray(shape=(len(training_patient_ids), input_data_size),
                               dtype=np.float32)
    training_labels = np.ndarray(shape=(len(training_patient_ids), output_size),
                                 dtype=np.float32)
    np.random.shuffle(training_patient_ids)
    index = 0
    for patient_id in training_patient_ids:
        training_data[index, :] = compute_probability_distribution(patient_id_to_input_values[patient_id])
        training_labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset


def create_validation_dataset(
        validation_patient_ids, input_data_size, output_size,
        patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label):
    """

    :param (list) validation_patient_ids:
    :param (integer) input_data_size:
    :param (integer) output_size
    :param (dictionary) patient_id_to_input_values:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """

    validation_dataset = dict()
    validation_data = np.ndarray(shape=(len(validation_patient_ids), input_data_size),
                                 dtype=np.float32)
    validation_labels = np.ndarray(shape=(len(validation_patient_ids), output_size),
                                   dtype=np.float32)

    np.random.shuffle(validation_patient_ids)
    index = 0
    for patient_id in validation_patient_ids:
        validation_data[index, :] = compute_probability_distribution(patient_id_to_input_values[patient_id])
        validation_labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_test_dataset(
        test_patient_ids, input_data_size, output_size,
        patient_id_to_geneExpressions, label_to_oneHotEncoding, patient_id_to_label):

    """

    :param test_patient_ids:
    :param input_data_size:
    :param output_size:
    :param patient_id_to_geneExpressions:
    :param label_to_oneHotEncoding:
    :param patient_id_to_label:
    :return:
    """

    test_dataset = dict()
    # create test data
    test_data = np.ndarray(shape=(len(test_patient_ids), input_data_size),
                           dtype=np.float32)
    test_labels = np.ndarray(shape=(len(test_patient_ids), output_size),
                             dtype=np.float32)

    np.random.shuffle(test_patient_ids)
    index = 0
    for patient_id in test_patient_ids:
        test_data[index, :] = compute_probability_distribution(patient_id_to_geneExpressions[patient_id])
        test_labels[index, :] = label_to_oneHotEncoding[patient_id_to_label[patient_id]]
        index += 1

    test_dataset["test_data"] = test_data
    test_dataset["test_labels"] = test_labels

    return test_dataset


def create_training_dataset_with_clusters(
        training_patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label):
    """

    :param (list) training_patient_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) patient_id_to_input_values_clusters:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """
    training_dataset = dict()

    training_data = dict()

    for cluster_id in range(len(clusters_size)):
        training_data[cluster_id] = np.ndarray(shape=(len(training_patient_ids), clusters_size[cluster_id]),
                                               dtype=np.float32)

    training_labels = np.ndarray(shape=(len(training_patient_ids), output_size),
                                 dtype=np.float32)

    np.random.shuffle(training_patient_ids)
    index = 0
    for patient_id in training_patient_ids:
        for cluster_id in range(len(clusters_size)):
            training_data[cluster_id][index, :] = \
                compute_probability_distribution(patient_id_to_input_values_clusters[patient_id][cluster_id])
        training_labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset


def create_validation_dataset_with_clusters(
        validation_patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label):
    """

    :param (list) validation_patient_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) patient_id_to_input_values_clusters:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """
    validation_dataset = dict()

    validation_data = dict()

    for cluster_id in range(len(clusters_size)):
        validation_data[cluster_id] = np.ndarray(shape=(len(validation_patient_ids), clusters_size[cluster_id]),
                                                 dtype=np.float32)

    validation_labels = np.ndarray(shape=(len(validation_patient_ids), output_size),
                                   dtype=np.float32)

    np.random.shuffle(validation_patient_ids)
    index = 0
    for patient_id in validation_patient_ids:
        for cluster_id in range(len(clusters_size)):
            validation_data[cluster_id][index, :] = \
                compute_probability_distribution(patient_id_to_input_values_clusters[patient_id][cluster_id])
        validation_labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_test_dataset_with_clusters(
        test_patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label):
    """

    :param (list) test_patient_ids:
    :param (list) clusters_size:
    :param (integer) output_size:
    :param (dictionary) patient_id_to_input_values_clusters:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """
    testing_dataset = dict()

    testing_data = dict()

    for cluster_id in range(len(clusters_size)):
        testing_data[cluster_id] = np.ndarray(shape=(len(test_patient_ids), clusters_size[cluster_id]),
                                              dtype=np.float32)

    training_labels = np.ndarray(shape=(len(test_patient_ids), output_size),
                                 dtype=np.float32)

    np.random.shuffle(test_patient_ids)
    index = 0
    for patient_id in test_patient_ids:
        for cluster_id in range(len(clusters_size)):
            testing_data[cluster_id][index, :] = \
                compute_probability_distribution(patient_id_to_input_values_clusters[patient_id][cluster_id])
        training_labels[index, :] = label_to_one_hot_encoding[patient_id_to_label[patient_id]]
        index += 1

    testing_dataset["test_data"] = testing_data
    testing_dataset["test_labels"] = training_labels

    return testing_dataset


def create_k_fold_patient_ids(k, label_to_patient_ids):
    """
    Separates the patient_ids into k folds. (k-1) folds will be used to training and one fold for validation.

    :param k: number of folds
    :param label_to_patient_ids:
    :return:
    """
    k_fold_patient_ids = dict()
    for index in range(k):
        k_fold_patient_ids[index] = []

    labels = label_to_patient_ids.keys()
    for label in labels:
        patient_ids = label_to_patient_ids[label]
        group_size = len(patient_ids)/k
        for index in range(k-1):
            k_fold_patient_ids[index] += patient_ids[index*group_size:(index+1)*group_size]
        k_fold_patient_ids[k-1] += patient_ids[(k-1)*group_size:]

    return k_fold_patient_ids


def create_k_fold_datasets(
        k, k_fold_patient_ids, input_data_size, output_size,
        patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label):
    """
    Creates the datasets corresponding to each fold.

    :param k:
    :param k_fold_patient_ids:
    :param (integer) input_data_size:
    :param (integer) output_size:
    :param (dictionary) patient_id_to_input_values:
    :param (dictionary) label_to_one_hot_encoding:
    :param (dictionary) patient_id_to_label:
    :return:
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_patient_ids = k_fold_patient_ids[index_i]
        training_patient_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_patient_ids += k_fold_patient_ids[index_j]

        training_dataset = create_training_dataset(
            training_patient_ids, input_data_size, output_size,
            patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label)

        validation_dataset = create_validation_dataset(
            validation_patient_ids, input_data_size, output_size,
            patient_id_to_input_values, label_to_one_hot_encoding, patient_id_to_label)

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets


def create_k_fold_datasets_with_clusters(
        k, k_fold_patient_ids, clusters_size, output_size,
        patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label):
    """
    Creates the datasets corresponding to each fold.

    :param k:
    :param k_fold_patient_ids:
    :param clusters_size:
    :param output_size:
    :param patient_id_to_input_values_clusters:
    :param label_to_one_hot_encoding:
    :param patient_id_to_label:
    :return:
    """

    k_fold_datasets = dict()
    for index in range(k):
        k_fold_datasets[index] = dict()

    for index_i in range(k):
        validation_patient_ids = k_fold_patient_ids[index_i]
        training_patient_ids = []
        for index_j in range(k):
            if index_j != index_i:
                training_patient_ids += k_fold_patient_ids[index_j]

        training_dataset = create_training_dataset_with_clusters(
            training_patient_ids, clusters_size, output_size,
            patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label)


        validation_dataset = create_validation_dataset_with_clusters(
            validation_patient_ids, clusters_size, output_size,
            patient_id_to_input_values_clusters, label_to_one_hot_encoding, patient_id_to_label)

        k_fold_datasets[index_i]["training_dataset"] = training_dataset
        k_fold_datasets[index_i]["validation_dataset"] = validation_dataset

    return k_fold_datasets


