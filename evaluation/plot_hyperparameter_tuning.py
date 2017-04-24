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



def plot_keep_probability(epigenetic_data, epigenetic_data_with_clusters):

    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    print input_data_size
    print output_size

    """ Multilayer Perceptron """

    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()

    best_keep_probability, mean_error_rates, std_error_rates = \
        choose_keep_probability(MLP, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0)

    plt.errorbar(keep_probability_values, mean_error_rates, yerr=std_error_rates,
             color='green', label='MLP', lw=3)

    """ Recurrent Neural Network """

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[128, 32],
        output_size=output_size)
    best_keep_probability, mean_error_rates, std_error_rates = \
        choose_keep_probability(recurrent_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.0001, 0)
    plt.errorbar(keep_probability_values, mean_error_rates, yerr=std_error_rates,
             color='red', label='RNN', lw=3)

    "Superlayered Neural Network"

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32], output_size)

    best_keep_probability, mean_error_rates, std_error_rates = \
        choose_keep_probability(superlayered_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0)
    plt.errorbar(keep_probability_values, mean_error_rates, yerr=std_error_rates,
             color='blue', label='SNN', lw=3)


    """ Plot details """

    plt.xlabel('Keep Probability', size=24)
    plt.ylabel('Error Rate', size=24)
    plt.legend()
    plt.savefig('keep_probability.png')
    plt.savefig('keep_probability.eps')


def plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters):

    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    print input_data_size
    print output_size

    """ Multilayer Perceptron """

    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()

    _, mean_error_rates, std_error_rates = \
        choose_learning_rate(MLP, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0.01)

    plt.errorbar(learning_rate_values, mean_error_rates, yerr=std_error_rates,
             color='green', label='MLP', lw=3)

    """ Recurrent Neural Network """

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[32, 32],
        output_size=output_size)
    _, mean_error_rates, std_error_rates = \
        choose_learning_rate(recurrent_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.0001, 0.001)
    plt.errorbar(learning_rate_values, mean_error_rates, yerr=std_error_rates,
             color='red', label='RNN', lw=3)

    "Superlayered Neural Network"

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32], output_size)

    _, mean_error_rates, std_error_rates = \
        choose_learning_rate(superlayered_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0.01)
    plt.errorbar(learning_rate_values, mean_error_rates, yerr=std_error_rates,
             color='blue', label='SNN', lw=3)


    """ Plot details """

    plt.xlabel('Learning Rate', size=24)
    plt.ylabel('Error Rate', size=24)
    plt.legend()
    plt.savefig('learning_rate.png')
    plt.savefig('learning_rate.eps')


def plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters):


    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    print input_data_size
    print output_size

    """ Multilayer Perceptron """

    MLP = MultilayerPerceptron(input_data_size, [64, 32, 16, 8], output_size)
    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()

    _, mean_error_rates, std_error_rates = \
        choose_weight_decay(MLP, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0)

    plt.errorbar(weight_decay_values, mean_error_rates, yerr=std_error_rates,
             color='green', label='MLP', lw=3)

    """ Recurrent Neural Network """

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[16, 64], hidden_units=[32, 16],
        output_size=output_size)
    _, mean_error_rates, std_error_rates = \
        choose_weight_decay(recurrent_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.0001, 0)
    plt.errorbar(weight_decay_values, mean_error_rates, yerr=std_error_rates,
             color='red', label='RNN', lw=3)

    "Superlayered Neural Network"

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]],
        [32, 16], output_size)

    _, mean_error_rates, std_error_rates = \
        choose_weight_decay(superlayered_neural_network, k_fold_datasets_hyperparameters_tuning[0], 0.05, 0)
    plt.errorbar(weight_decay_values, mean_error_rates, yerr=std_error_rates,
             color='blue', label='SNN', lw=3)


    """ Plot details """

    plt.xlabel('Keep Probability', size=24)
    plt.ylabel('Error Rate', size=24)
    plt.legend()
    plt.savefig('weight_decay.png')
    plt.savefig('weight_decay.eps')




print "GOT here"
epigenetic_data = CancerPatientsData(num_folds=10, num_folds_hyperparameters_tuning=10)
epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=10, num_folds_hyperparameters_tuning=10)

plot_keep_probability(epigenetic_data, epigenetic_data_with_clusters)
plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters)
plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters)