import numpy as np
import math
from evaluation_metrics import *


from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster
from epigenetic_data.cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, \
    CancerPatientsDataDNAMethylationLevels

from neural_network_models.multilayer_perceptron import MultilayerPerceptron
from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN


def evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster):


    evaluate_superlayered_neural_network(epigenetic_data_with_clusters)
    evaluate_feed_forward_neural_network(epigenetic_data)
    evaluate_recurrent_neural_network(epigenetic_data)



def evaluate_feed_forward_neural_network(epigenetic_data):


    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Feed Forward Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_size
    print "input data size"
    print input_data_size
    output_size = epigenetic_data.output_size

    feed_forward_neural_network = MultilayerPerceptron(input_data_size, [256, 128, 64, 32], output_size)
    average_accuracy, confussion_matrix, ROC_points = nested_cross_validation_on_MLP(feed_forward_neural_network, epigenetic_data)
    print epigenetic_data.label_to_one_hot_encoding


    plot_ROC_curves(ROC_points)
    plot_confussion_matrix_as_heatmap(confussion_matrix)


def evaluate_superlayered_neural_network(epigenetic_data_with_clusters):

    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Superlayered Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    clusters_size = epigenetic_data_with_clusters.clusters_size
    output_size = epigenetic_data_with_clusters.output_size

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32], output_size)"""

    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [128, 32], output_size)

    average_accuracy, confussion_matrix, ROC_points = nested_cross_validation_on_SNN(
        superlayered_neural_network, epigenetic_data_with_clusters)

    label_to_one_hot_encoding = epigenetic_data_with_clusters.label_to_one_hot_encoding
    class_id_to_symbol_id = compute_class_id_to_class_symbol(label_to_one_hot_encoding)
    class_symbol_to_evaluation_matrix = compute_evaluation_metrics_for_each_class(
        confussion_matrix, class_id_to_symbol_id)

    print class_symbol_to_evaluation_matrix

    plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confussion Matrix for SNN')
    plot_ROC_curves(ROC_points)


def evaluate_recurrent_neural_network(epigenetic_data):

    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Recurrent Neural Network-------------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size

       # Architecture for cancer data

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size/4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[64, 32],
        output_size=output_size)

    """
    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=16, input_step_size=8,
        LSTM_units_state_size=[64, 256], hidden_units=[256, 64],
        output_size=output_size)"""

    average_accuracy, confussion_matrix, ROC_points = nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data)
    print epigenetic_data.label_to_one_hot_encoding

    #plot_ROC_curves(ROC_points)
    plot_confussion_matrix_as_heatmap(confussion_matrix)


def get_embryo_development_data():

    noise_mean = 0
    noise_stddev = 1

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    epigenetic_data = EmbryoDevelopmentData(
        num_folds=5, num_folds_hyperparameters_tuning=3, max_num_genes=64, gene_entropy_threshold=6.3)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = EmbryoDevelopmentDataWithClusters(
        num_clusters=2, clustering_algorithm='hierarchical',
        num_folds=2, num_folds_hyperparameters_tuning=3,
        max_num_genes=300, gene_entropy_threshold=6.0)
    epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = EmbryoDevelopmentDataWithSingleCluster(
        num_clusters=2, clustering_algorithm='hierarchical',
        num_folds=5, num_folds_hyperparameters_tuning=3,
        max_num_genes=100, gene_entropy_threshold=6.0)
    epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster

def get_cancer_data():

    noise_mean = 0
    noise_stddev = 0.03

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    epigenetic_data = CancerPatientsData(num_folds=5, num_folds_hyperparameters_tuning=3)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=5, num_folds_hyperparameters_tuning=3)
    #epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = CancerPatientsDataDNAMethylationLevels(num_folds=5, num_folds_hyperparameters_tuning=3)
    #epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster


epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_embryo_development_data()
evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster)


epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_cancer_data()
evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster)


