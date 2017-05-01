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
learning_rate_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.75, 0.1, 0.5]

weight_decay_values = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]


def weight_decay_plots(network, k_fold_datasets_hyperparameters_tuning, learning_rate, filename):

    keys = k_fold_datasets_hyperparameters_tuning.keys()

    f = open(filename, "w")
    for key in keys:
        print "Plotting for key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_weight_decay(network, k_fold_datasets, learning_rate, 1)
        f.write(str(average))
        f.write('\n')
        f.write(str(std))
    f.close()



def keep_probability_plots(network, k_fold_datasets_hyperparameters_tuning, learning_rate, filename):

    keys = k_fold_datasets_hyperparameters_tuning.keys()

    f = open(filename, "w")

    for key in keys:
        print "key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_keep_probability(network, k_fold_datasets, learning_rate, 0)
        f.write(str(average))
        f.write('\n')
        f.write(str(std))
    f.close()



def learning_rate_plots(network, k_fold_datasets_hyperparameters_tuning, filename):

    keys = k_fold_datasets_hyperparameters_tuning.keys()

    f = open(filename, "w")

    for key in keys:
        print "key number" + str(key)
        k_fold_datasets = k_fold_datasets_hyperparameters_tuning[key]
        _, average, std = choose_learning_rate(network, k_fold_datasets, 0, 1)

        f.write(str(average))
        f.write('\n')
        f.write(str(std))

    f.close()




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
    keep_probability_plots(MLP, k_fold_datasets_hyperparameters_tuning, 0.05, 'mlp_kp')
    keep_probability_plots(RNN, k_fold_datasets_hyperparameters_tuning, 0.0001, 'rnn_kp')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    keep_probability_plots(SNN, k_fold_datasets_hyperparameters_tuning, 0.05, 'snn_kp')


def plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters):

    _, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    RNN = RecurrentNeuralNetwork(
            input_sequence_length=input_data_size / 4, input_step_size=4,
            LSTM_units_state_size=[16, 64], hidden_units=[32, 16],
            output_size=output_size)

    weight_decay_plots(MLP, k_fold_datasets_hyperparameters_tuning, 0.05, 'mlp_wd')
    weight_decay_plots(RNN, k_fold_datasets_hyperparameters_tuning, 0.0001, 'rnn_wd')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    weight_decay_plots(SNN, k_fold_datasets_hyperparameters_tuning, 0.05, 'snn_wd')


def plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters):

    _, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    MLP = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
    RNN = RecurrentNeuralNetwork(
            input_sequence_length=input_data_size / 4, input_step_size=4,
            LSTM_units_state_size=[32, 128], hidden_units=[128, 32],
            output_size=output_size)

    learning_rate_plots(MLP, k_fold_datasets_hyperparameters_tuning, 'mlp_ln')
    learning_rate_plots(RNN, k_fold_datasets_hyperparameters_tuning, 'rnn_ln')


    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data_with_clusters.get_k_fold_datasets()
    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size
    SNN = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[128, 64, 32, 16], [128, 64, 32, 16]], [32, 16], output_size)

    learning_rate_plots(SNN, k_fold_datasets_hyperparameters_tuning, 'snn_ln ')


epigenetic_data = CancerPatientsData(num_folds=10, num_folds_hyperparameters_tuning=10)
epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=10, num_folds_hyperparameters_tuning=10)

print np.log10(weight_decay_values)
print np.log10(learning_rate_values)

#plot_keep_probability(epigenetic_data, epigenetic_data_with_clusters)
#plot_weight_decay(epigenetic_data, epigenetic_data_with_clusters)
#plot_learning_rate(epigenetic_data, epigenetic_data_with_clusters)



def plots_for_kp(filenme, title):
    fig = plt.figure(figsize=(8, 10), dpi=150)
    ax = plt.subplot(111)
    ax.set_color_cycle(['r', 'g', 'b', 'm', 'y', 'c', 'pink', 'orange', 'indigo', 'k'])

    f = open(filenme, 'r')

    for key in range(10):
        average = f.readline()
        average = average[:-1]
        std = f.readline()

        average = average.split(',')
        average = [float(element) for element in average]

        std = std.split(',')
        std = [float(element) for element in std]

        plt.errorbar(keep_probability_values, average, yerr=std, label='Fold #%d' % key, ls='-')

    plt.xlabel('Keep probability', size=18)
    plt.ylabel('Error rate', size=18)

    plt.xticks(np.arange(0.0, 1.2, 0.10))
    plt.yticks(np.arange(0.00, 0.202, 0.025))
    plt.legend(loc='upper right', prop={'size': 10}, ncol=2)

    chartBox = ax.get_position()
    #ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    #ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.9), shadow=False, ncol=1)
    plt.title(title, size=18, y=1.04)
    plt.show()



def plots_for_wd(filenme, title):
    fig = plt.figure(figsize=(8, 10), dpi=150)
    ax = plt.subplot(111)
    ax.set_color_cycle(['r', 'g', 'b', 'm', 'y', 'c', 'pink', 'orange', 'indigo', 'k'])

    f = open(filenme, 'r')

    weight_decay_values = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    weight_decay_values = np.log10(weight_decay_values)

    for key in range(10):
        average = f.readline()
        average = average[:-1]
        std = f.readline()

        average = average.split(',')
        average = [float(element) for element in average]

        std = std.split(',')
        std = [float(element) for element in std]

        plt.errorbar(weight_decay_values, average, yerr=std, label='Fold #%d' % key, ls='-')

    plt.xlabel('log$_{10}$(Weight decay)', size=16)
    plt.ylabel('Error rate', size=16)

    plt.xticks(np.arange(-3.5, 0.55, 0.5))
    plt.yticks(np.arange(0.00, 0.202, 0.025))
    plt.legend(loc='upper right', prop={'size': 10}, ncol=5)

    chartBox = ax.get_position()
    #ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    #ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.9), shadow=False, ncol=1)
    plt.title(title, size=18, y=1.04)
    plt.show()


plots_for_wd('mlp_wd', "Error rate of MLP against different weight decay values")


    #plots_for_kp('snn_kp', "Error rate of SNN against different keep probability values")
#plots_for_kp('rnn_kp', "Error rate of RNN against different keep probability values")

#plots_for_kp('mlp_kp', "Error rate of MLP against different keep probability values")