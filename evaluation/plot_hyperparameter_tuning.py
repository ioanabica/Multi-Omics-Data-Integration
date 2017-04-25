import numpy as np
import math
import matplotlib.pyplot as plt

from epigenetic_data.cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, \
    CancerPatientsDataDNAMethylationLevels

from neural_network_models.multilayer_perceptron import MultilayerPerceptron
from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from hyperparameters_tuning import choose_keep_probability, choose_learning_rate, choose_weight_decay

keep_probability_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
learning_rate_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1]
weight_decay_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]


def weight_decay_plots(network, k_fold_datasets_hyperparameters_tuning, learning_rate, title, figname):

    keys = k_fold_datasets_hyperparameters_tuning.keys()
    print keys

    lw = 2

    for key in keys:
        print "Plotting for key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_weight_decay(network, k_fold_datasets, learning_rate, 1)

        plt.errorbar(weight_decay_values, average, yerr=std, lw=lw, label='Fold %d' % (key))

    plt.xlabel('Weight decay values')
    plt.ylabel('Error rate')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(figname)
    plt.show()


def keep_probability_plots(network, k_fold_datasets_hyperparameters_tuning, learning_rate, title, figname):

    keys = k_fold_datasets_hyperparameters_tuning.keys()

    lw = 2

    for key in keys:
        print "key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_keep_probability(network, k_fold_datasets, learning_rate, 0)

        plt.errorbar(keep_probability_values, average, yerr=std, lw=lw, label='Fold %d' % key)

    plt.xlabel('Keep probability values')
    plt.ylabel('Error rate')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(figname)


def learning_rate_plots(network, k_fold_datasets_hyperparameters_tuning, title, figname):

    keys = k_fold_datasets_hyperparameters_tuning.keys()
    lw = 2

    for key in keys:
        print "key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_learning_rate(network, k_fold_datasets, 0, 1)

        plt.errorbar(learning_rate_values, average, yerr=std, lw=lw, label='Fold %d' % key)

    plt.xlabel('Learning rate values')
    plt.ylabel('Error rate')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(figname)



def plot_keep_probability(epigenetic_data, epigenetic_data_with_clusters):

    _, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    RNN = RecurrentNeuralNetwork(
            input_sequence_length=input_data_size / 4, input_step_size=4,
            LSTM_units_state_size=[16, 64], hidden_units=[32, 16],
            output_size=output_size)
    print "Got here"
    keep_probability_plots(MLP, k_fold_datasets_hyperparameters_tuning, 0.05,
                           "Error rate of MLP agains different keep probabilities", 'mlp_kp.eps')
    keep_probability_plots(RNN, k_fold_datasets_hyperparameters_tuning, 0.0001,
                           "Error rate of RNN again different keep probabilities", 'rnn_kp.eps')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    keep_probability_plots(SNN, k_fold_datasets_hyperparameters_tuning, 0.05,
                           "Error rate of MLP agains different keep probabilities", 'snn_kp.eps')


def plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters):

    _, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    RNN = RecurrentNeuralNetwork(
            input_sequence_length=input_data_size / 4, input_step_size=4,
            LSTM_units_state_size=[16, 64], hidden_units=[32, 16],
            output_size=output_size)
    print "Got here"
    weight_decay_plots(MLP, k_fold_datasets_hyperparameters_tuning, 0.01,
                           "Error rate of MLP agains different keep probabilities", 'mlp_wd.eps')
    weight_decay_plots(RNN, k_fold_datasets_hyperparameters_tuning, 0.0001,
                           "Error rate of RNN again different keep probabilities", 'rnn_wd.eps')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    weight_decay_plots(SNN, k_fold_datasets_hyperparameters_tuning, 0.05,
                           "Error rate of MLP agains different keep probabilities", 'snn_wd.eps')


def plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters):

    _, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    RNN = RecurrentNeuralNetwork(
            input_sequence_length=input_data_size / 4, input_step_size=4,
            LSTM_units_state_size=[16, 64], hidden_units=[32, 16],
            output_size=output_size)

    learning_rate_plots(MLP, k_fold_datasets_hyperparameters_tuning,
                           "Error rate of MLP agains different keep probabilities", 'mlp_ln.eps')
    learning_rate_plots(RNN, k_fold_datasets_hyperparameters_tuning,
                           "Error rate of RNN again different keep probabilities", 'rnn_ln.eps')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    learning_rate_plots(SNN, k_fold_datasets_hyperparameters_tuning, "Error rate of MLP agains different keep probabilities", 'snn_ln.eps')


epigenetic_data = CancerPatientsData(num_folds=10, num_folds_hyperparameters_tuning=10)
epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=10, num_folds_hyperparameters_tuning=10)

plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters)
plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters)
plot_keep_probability(epigenetic_data, epigenetic_data_with_clusters)