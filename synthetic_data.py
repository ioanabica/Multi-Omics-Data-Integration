import numpy as np
import math

from LSTM_recurrent_neural_network import RecurrentNeuralNetwork
from feedforward_neural_network import FeedforwardNeuralNetwork
from LSTM_for_testing import  RecurrentNeuralNetwork

num_classes = 7
num_genes = 64
num_shifted_genes = 8
training_examples_for_class = 10
validation_examples_for_class = 2

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


def create_data_point(num_genes, class_id, class_id_to_shifted_genes):
    mean = 0
    stddev = 1
    data_point = np.random.normal(mean, stddev, num_genes)

    shifted_mean = 5

    """
    start_class_genes = class_id * (num_genes/num_classes)
    end_class_genes = (class_id + 1) * (num_genes/num_classes)
    data_point[start_class_genes:end_class_genes] = \
        np.random.normal(shifted_mean, stddev, num_genes/num_classes)


    class_gene = class_id
    for index in range(num_shifted_genes):
        if(class_gene < num_genes)
            data_point[class_gene] = np.random.normal(shifted_mean, stddev, 1)
            class_gene += """


    shifted_genes = class_id_to_shifted_genes[class_id]

    for shifted_gene in shifted_genes:
        data_point[shifted_gene] = np.random.normal(shifted_mean, stddev, 1)


    return data_point


def create_one_hot_encoding(class_id):
    one_hot_encoding = [0] * num_classes
    one_hot_encoding[class_id] = 1.0

    return one_hot_encoding


def create_training_dataset(class_id_to_shifted_genes):
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
                normalize_data(create_data_point(num_genes, class_id, class_id_to_shifted_genes))
            training_labels[class_id * training_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    data_and_labels = (zip(training_data, training_labels))
    np.random.shuffle(data_and_labels)

    permutation = np.random.permutation(len(training_data))

    training_dataset["training_data"] = training_data[permutation]
    training_dataset["training_labels"] = training_labels[permutation]

    return training_dataset


def create_validation_dataset(class_id_to_shifted_genes):
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
                normalize_data(create_data_point(num_genes, class_id, class_id_to_shifted_genes))
            validation_labels[class_id * validation_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    validation_dataset["validation_data"] = validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_class_id_to_shifted_genes(num_classes, num_genes, num_shifted_genes):

    class_id_to_shifted_genes = dict()

    for index in range(num_classes):
        shifted_genes = np.random.choice(range(num_genes), num_shifted_genes, replace=False)
        class_id_to_shifted_genes[index] = shifted_genes

    return class_id_to_shifted_genes


class SyntheticData(object):

    class_id_to_shifted_genes = create_class_id_to_shifted_genes(num_classes, num_genes, num_shifted_genes)
    print class_id_to_shifted_genes

    training_dataset = create_training_dataset(class_id_to_shifted_genes)
    validation_dataset = create_validation_dataset(class_id_to_shifted_genes)

    ffnn = FeedforwardNeuralNetwork(num_genes, [256, 128, 64, 32], num_classes)
    validation_accurat = ffnn.train_and_validate(training_dataset, validation_dataset, 0.05, 0.01, 0.5)


    #rnn = RecurrentNeuralNetwork(num_genes/8, 8, [64, 128, 256], [512, 256, 128, 32], num_classes)

    # rnn = RecurrentNeuralNetwork(num_genes/4, 4, [64, 128, 256], [512, 256, 128, 32], num_classes)
    # rnn = RecurrentNeuralNetwork(num_genes/8, 8, [32, 64, 128], [256, 128, 64, 32], num_classes)

    # rnn = RecurrentNeuralNetwork(num_genes/8, 8, [32, 64, 128, 256], [256, 128, 64, 32], num_classes)

    #rnn = RecurrentNeuralNetwork(num_genes/4, 4, [16, 32, 64, 128], [256, 128, 64, 32], num_classes)

    rnn = RecurrentNeuralNetwork(num_genes / 4, 4, [64, 128], [256, 128, 64, 32], num_classes)
    print str(4) + str([32, 64]) + str([128, 64, 32, 16])

    validation_accuracy = rnn.train_and_validate(training_dataset, validation_dataset)

