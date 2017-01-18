import numpy as np
from feedforward_neural_network import train_feedforward_neural_network
from recurrent_neural_network import train_recurrent_neural_network

num_classes = 7
num_genes = 256
training_examples_for_class = 10
validation_examples_for_class = 3

num_training_examples = num_classes * training_examples_for_class
num_validation_examples = num_classes * validation_examples_for_class


def normalize_data(data_point):
    """
    Shift the input data in interval [0, 1]

    :param data_point:
    :return:
    """
    min_element = min(data_point)
    max_element = max(data_point)

    # shift data in interval [0, 1]
    normalized_data = np.ndarray(shape=(len(data_point)), dtype=np.float32)
    for index in range(len(data_point)):
        normalized_data[index] = (data_point[index] - min_element)/(max_element - min_element)

    # create probability distribution
    #total_sum = sum(normalized_data)
    #for index in range(len(normalized_data)):
        #normalized_data[index] = normalized_data[index] / total_sum

    return normalized_data


def create_data_point(num_genes, class_id):
    mean = 0
    stddev = 1
    data_point = np.random.normal(mean, stddev, num_genes)

    shifted_mean = 5

    start_class_genes = class_id * (num_genes/num_classes)
    end_class_genes = (class_id + 1) * (num_genes/num_classes)
    data_point[start_class_genes:end_class_genes] = \
        np.random.normal(shifted_mean, stddev, num_genes/num_classes)

    return normalize_data(data_point)


def create_one_hot_encoding(class_id):
    one_hot_encoding = [0] * num_classes
    one_hot_encoding[class_id] = 1.0

    return one_hot_encoding


def create_training_dataset():
    """

    :return:
    """
    training_dataset = dict()

    training_data = np.ndarray(shape=(num_training_examples, num_genes),
                               dtype=np.float32)
    training_labels = np.ndarray(shape=(num_training_examples, num_classes),
                                 dtype=np.float32)

    for class_id in range(num_classes):
        for index in range(training_examples_for_class):
            training_data[class_id * training_examples_for_class + index, :] = \
                normalize_data(create_data_point(num_genes, class_id))
            training_labels[class_id * training_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    training_dataset["training_data"] = training_data
    training_dataset["training_labels"] = training_labels

    return training_dataset


def create_validation_dataset():
    """

    :return:
    """
    validation_dataset = dict()

    validation_data = np.ndarray(shape=(num_validation_examples, num_genes),
                               dtype=np.float32)
    validation_labels = np.ndarray(shape=(num_validation_examples, num_classes),
                                 dtype=np.float32)

    for class_id in range(num_classes):
        for index in range(validation_examples_for_class):
            validation_data[class_id * validation_examples_for_class + index, :] = \
                normalize_data(create_data_point(num_genes, class_id))
            validation_labels[class_id * validation_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


class SyntheticData(object):

    training_dataset = create_training_dataset()
    validation_dataset = create_validation_dataset()

    validation_accuracy = train_recurrent_neural_network(training_dataset, validation_dataset, num_genes,
                                                            num_classes)


