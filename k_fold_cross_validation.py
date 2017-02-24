import numpy as np
from embryo_development_data import EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters

from feedforward_neural_network import FeedforwardNeuralNetwork
from LSTM_recurrent_neural_network import RecurrentNeuralNetwork
from superlayered_neural_network import SuperlayeredNeuralNetwork

from gene_clustering import plot_dendogram, k_means_clustering

import matplotlib.pyplot as plt

num_folds = 3


def cross_validate_MLP():
    keep_probability = 0.5
    epsilon = 1e-3
    learning_rate = 0.05
    weight_decay = 0.01

    epigeneticsData = EmbryoDevelopmentData(num_folds, 256, 6.2)

    k_fold_datasets = epigeneticsData.get_k_fold_datasets()
    input_data_size = epigeneticsData.input_data_size
    output_size = epigeneticsData.output_size

    ffnn = FeedforwardNeuralNetwork(input_data_size, [256, 128, 64, 32], output_size)

    keys = k_fold_datasets.keys()

    validation_accuracies = list()
    training_accuracies = list()
    losses = list()

    for key in keys:
        print "key number" + str(key)
        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, training_accuracy, loss = ffnn.train_and_validate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)
        validation_accuracies.append(validation_accuracy)
        training_accuracies.append(training_accuracy)
        losses.append(loss)

    return validation_accuracies, training_accuracies, losses


def cross_validate_RNN():
    MLP_keep_probability = 0.5
    LSTMs_keep_probability = 0.5

    # Training parameters
    RNN_learning_rate = 0.0001
    RNN_weight_decay = 0.01
    epigeneticsData = EmbryoDevelopmentData(num_folds, 256, 6.25)

    k_fold_datasets = epigeneticsData.get_k_fold_datasets()
    input_data_size = epigeneticsData.input_data_size
    output_size = epigeneticsData.output_size

    rnn = RecurrentNeuralNetwork(16, 16, [64, 128], [128, 64], output_size)

    validation_accuracies = list()
    training_accuracies = list()
    losses = list()

    keys = k_fold_datasets.keys()

    for key in keys:
        print "key number" + str(key)
        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, training_accuracy, loss = rnn.train_and_validate(
            training_dataset, validation_dataset,
            RNN_learning_rate, RNN_weight_decay, LSTMs_keep_probability, MLP_keep_probability)
        validation_accuracies.append(validation_accuracy)
        training_accuracies.append(training_accuracy)
        losses.append(loss)

    return validation_accuracies, training_accuracies, losses


def cross_validate_superlayeredNN():

    epigeneticsData = EmbryoDevelopmentDataWithClusters(2, num_folds, 256, 6)
    k_fold_datasets_with_clusters = epigeneticsData.get_k_fold_datasets()
    clusters_size = epigeneticsData.clusters_size
    print(clusters_size)
    output_size = epigeneticsData.output_size


    superlayered_nn = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32],
        output_size)

    keys = k_fold_datasets_with_clusters.keys()


    validation_accuracies = list()
    training_accuracies = list()
    losses = list()

    for key in keys:
        training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
        print len(training_dataset["training_data"][0])
        validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]
        print len(validation_dataset["validation_data"][0])
        validation_accuracy, training_accuracy, loss = superlayered_nn.train_and_validate(training_dataset, validation_dataset)
        validation_accuracies.append(validation_accuracy)
        training_accuracies.append(training_accuracy)
        losses.append(loss)
        print "key number" + str(key)

    return validation_accuracies, training_accuracies, losses

#epi_data = EmbryoDevelopmentDataWithClusters(2, 4, 16, 6.3)
#plot_dendogram(epi_data.geneId_to_expression_levels)

SNN_validation_accuracy, SNN_training_accuracies, SNN_losses = cross_validate_superlayeredNN()
print SNN_validation_accuracy
print np.mean(SNN_validation_accuracy)

MLP_validation_accuracy, MLP_training_accuracies, MLP_losses = cross_validate_MLP()
print MLP_validation_accuracy
print np.mean(MLP_validation_accuracy)



RNN_validation_accuracy, RNN_training_accuracies, RNN_losses = cross_validate_RNN()
print RNN_validation_accuracy
print np.mean(RNN_validation_accuracy)



def plot_validation_accuracy(MLP_validation_accuracy, SNN_validation_accuracy, RNN_validation_accuracy):
    plt.figure(1)

    print "MLP accuracy"
    print np.mean(MLP_validation_accuracy)

    print "SNN accuracy"
    print np.mean(SNN_validation_accuracy)

    print "RNN accuracy"
    print np.mean(RNN_validation_accuracy)

    MLP, = plt.plot(range(num_folds), MLP_validation_accuracy, 'r-', label='Multilayer Perceptron NN')
    SNN, = plt.plot(range(num_folds), SNN_validation_accuracy, 'g-', label='Superlayered NN')
    RNN, = plt.plot(range(num_folds), RNN_validation_accuracy, 'b-', label='Recurrent NN')
    plt.legend(handles=[MLP, SNN, RNN], loc='lower right')

    axes = plt.gca()
    axes.set_ylim([0, 110])

    plt.title('K-fold Cross-Validation', fontsize=20)
    plt.xlabel('Fold Number', fontsize=18)
    plt.ylabel('Test Accuracy', fontsize=20)

    plt.show()


#plot_validation_accuracy(MLP_validation_accuracy, SNN_validation_accuracy, RNN_validation_accuracy)


def plot_losses(MLP_losses, SNN_losses, RNN_losses):
    plt.figure(1)

    plt.subplot(121)
    plt.plot(MLP_losses[0], 'r')

    plt.title('Multilayer Perceptron Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)


    plt.subplot(122)
    plt.plot(SNN_losses[0], 'g')

    plt.title('Superlayered Neural Network Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.show()

    plt.figure(2)

    plt.plot(RNN_losses[0], 'b')

    plt.title('Recurrent Neural Network Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.show()

#plot_losses(MLP_losses, SNN_losses, RNN_losses)


"""


for key in keys:
    print "key number" + str(key)
    training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]

    new_training_dataset = dict()
    new_training_dataset["training_data"] = training_dataset["training_data"][0]
    new_training_dataset["training_labels"] = training_dataset["training_labels"]
    print len(new_training_dataset["training_data"])
    print len(new_training_dataset["training_data"][0])

    validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]
    new_validation_dataset = dict()
    new_validation_dataset["validation_data"] = validation_dataset["validation_data"][0]
    new_validation_dataset["validation_labels"] = validation_dataset["validation_labels"]

    print len(new_validation_dataset["validation_data"])
    print len(new_validation_dataset["validation_data"][0])

    accuracy = rnn.train_and_validate(
        new_training_dataset, new_validation_dataset)
    validation_accuracy.append(accuracy)

print validation_accuracy
print numpy.mean(validation_accuracy)


"""
