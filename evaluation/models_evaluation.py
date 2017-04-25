import numpy as np
import math
from evaluation_metrics import paired_t_test_binary_classification, compute_average_performance_metrics_for_binary_classification, \
    compute_class_id_to_class_symbol, compute_evaluation_metrics_for_each_class, \
    compute_performance_metrics_for_multiclass_classification, plot_mean_ROC_curves
from plot_confussion_matrices import plot_confussion_matrix_as_heatmap_for_cancer_data, plot_confussion_matrix_as_heatmap
from gene_clustering.hierarchical_clustering import plot_dendogram


from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster
from epigenetic_data.cancer_data.cancer_data import CancerPatientsData, CancerPatientsDataWithModalities, \
    CancerPatientsDataDNAMethylationLevels

from neural_network_models.multilayer_perceptron import MultilayerPerceptron
from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN


def evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters):


    rnn_confussion_matrix, rnn_ROC_points, rnn_performance_metrics = evaluate_recurrent_neural_network(epigenetic_data)
    mlp_confussion_matrix, mlp_ROC_points, mlp_performance_metrics = evaluate_feed_forward_neural_network(epigenetic_data)
    snn_confussion_matrix, snn_ROC_points, snn_performance_metrics = evaluate_superlayered_neural_network(
        epigenetic_data_with_clusters)

    p_values_mlp_rnn = paired_t_test_binary_classification(mlp_performance_metrics, rnn_performance_metrics)
    p_values_mlp_snn = paired_t_test_binary_classification(mlp_performance_metrics, snn_performance_metrics)
    p_values_rnn_snn = paired_t_test_binary_classification(rnn_performance_metrics, snn_performance_metrics)


    print "Comparing MLP vs RNN"
    print p_values_mlp_rnn

    print "Comparing MLP vs SNN"
    print p_values_mlp_snn

    print "Comparing RNN vs SNN"
    print p_values_rnn_snn

    plot_mean_ROC_curves(mlp_ROC_points, rnn_ROC_points, snn_ROC_points)

    plot_confussion_matrix_as_heatmap_for_cancer_data(mlp_confussion_matrix, "Confusion Matrix for MLP")
    plot_confussion_matrix_as_heatmap_for_cancer_data(rnn_confussion_matrix, "Confusion Matrix for RNN")
    plot_confussion_matrix_as_heatmap_for_cancer_data(snn_confussion_matrix, "Confusion Matrix for SNN")


def evaluate_feed_forward_neural_network(epigenetic_data):


    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Feed Forward Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_size
    print "input data size"
    print input_data_size
    output_size = epigenetic_data.output_size

    """feed_forward_neural_network = MultilayerPerceptron(input_data_size, [256, 128, 64, 32], output_size)"""
    """feed_forward_neural_network = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)"""

    feed_forward_neural_network = MultilayerPerceptron(input_data_size, [64, 32, 16, 8], output_size)
    confussion_matrix, ROC_points, performance_metrics = nested_cross_validation_on_MLP(feed_forward_neural_network, epigenetic_data)

    label_to_one_hot_encoding = epigenetic_data.label_to_one_hot_encoding
    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for MLP')

    #plot_ROC_curves(ROC_points)

    return confussion_matrix, ROC_points, performance_metrics


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
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [64, 32], output_size)

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)"""

    confussion_matrix, ROC_points, performance_metrics = nested_cross_validation_on_SNN(
        superlayered_neural_network, epigenetic_data_with_clusters)

    label_to_one_hot_encoding = epigenetic_data_with_clusters.label_to_one_hot_encoding
    class_id_to_symbol_id = compute_class_id_to_class_symbol(label_to_one_hot_encoding)
    class_symbol_to_evaluation_matrix = compute_evaluation_metrics_for_each_class(
        confussion_matrix, class_id_to_symbol_id)

    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for SNN')
    #plot_ROC_curves(ROC_points)

    return confussion_matrix, ROC_points, performance_metrics


def evaluate_recurrent_neural_network(epigenetic_data):

    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Recurrent Neural Network-------------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_size
    print input_data_size
    output_size = epigenetic_data.output_size

    """"  Architecture for cancer data """

    """recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size/2, input_step_size=2,
        LSTM_units_state_size=[32, 128], hidden_units=[32, 32],
        output_size=output_size)"""

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[128, 32],
        output_size=output_size)

    """recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 2, input_step_size=2,
        LSTM_units_state_size=[32, 64], hidden_units=[32, 32],
        output_size=output_size)"""

    """ Architecture for embryo development data
    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=16, input_step_size=16,
        LSTM_units_state_size=[64, 128], hidden_units=[32, 32],
        output_size=output_size) """

    confussion_matrix, ROC_points, performance_metrics = \
        nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data)

    label_to_one_hot_encoding = epigenetic_data.label_to_one_hot_encoding
    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for RNN')

    return confussion_matrix, ROC_points, performance_metrics


def get_embryo_development_data():

    noise_mean = 0
    noise_stddev = 1

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    """epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=256, gene_entropy_threshold=6.2) """
    epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=16, gene_entropy_threshold=5.8)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = EmbryoDevelopmentDataWithClusters(
        num_clusters=2, clustering_algorithm='k-means',
        num_folds=5, num_folds_hyperparameters_tuning=3,
        max_num_genes=500, gene_entropy_threshold=6.0)
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

    epigenetic_data = CancerPatientsData(num_folds=10, num_folds_hyperparameters_tuning=3)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=10, num_folds_hyperparameters_tuning=3)
    #epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = CancerPatientsDataDNAMethylationLevels(num_folds=10, num_folds_hyperparameters_tuning=3)
    #epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster


#epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_embryo_development_data()
#plot_dendogram(epigenetic_data.geneId_to_expression_levels)


epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_cancer_data()
evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters)


#print "Evaluate for single modality"
#evaluate_neural_network_models(epigenetic_data_for_single_cluster, epigenetic_data_with_clusters)
