import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from evaluation_metrics import paired_t_test_binary_classification, compute_average_performance_metrics_for_binary_classification, \
    compute_class_id_to_class_symbol, compute_evaluation_metrics_for_each_class, \
    compute_performance_metrics_for_multiclass_classification, plot_mean_ROC_curves, paired_t_test_multiclass_classification, \
    plot_ROC_curves, plot_mean_ROC_curves_for_two_models
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


    mlp_confussion_matrix, mlp_ROC_points, mlp_performance_metrics = evaluate_feed_forward_neural_network(
        epigenetic_data)

    rnn_confussion_matrix, rnn_ROC_points, rnn_performance_metrics = evaluate_recurrent_neural_network(epigenetic_data)

    snn_confussion_matrix, snn_ROC_points, snn_performance_metrics = evaluate_superlayered_neural_network(
        epigenetic_data_with_clusters)



    """Evaluation metrics for cancer data



    p_values_mlp_rnn = paired_t_test_binary_classification(mlp_performance_metrics, rnn_performance_metrics)
    p_values_mlp_snn = paired_t_test_binary_classification(mlp_performance_metrics, snn_performance_metrics)
    p_values_rnn_snn = paired_t_test_binary_classification(rnn_performance_metrics, snn_performance_metrics)

    print "Comparing MLP vs RNN"
    print p_values_mlp_rnn

    print "Comparing MLP vs SNN"
    print p_values_mlp_snn

    print "Comparing RNN vs SNN"
    print p_values_rnn_snn

    #plot_mean_ROC_curves(mlp_ROC_points, rnn_ROC_points, snn_ROC_points)

    #plot_confussion_matrix_as_heatmap_for_cancer_data(mlp_confussion_matrix, "Confusion Matrix for MLP")
    #plot_confussion_matrix_as_heatmap_for_cancer_data(rnn_confussion_matrix, "Confusion Matrix for RNN")
    #plot_confussion_matrix_as_heatmap_for_cancer_data(snn_confussion_matrix, "Confusion Matrix for SNN")"""


    """Evaluaiton for embryo data
    print "Comparing MLP vs RNN"
    print "Micro"
    p_values_mlp_rnn_micro = paired_t_test_multiclass_classification(
        mlp_performance_metrics['micro'], rnn_performance_metrics['micro'])

    print "Comparing MLP vs SNN"
    print "Micro"
    p_values_mlp_snn_micro = paired_t_test_multiclass_classification(
        mlp_performance_metrics['micro'], snn_performance_metrics['micro'])

    print "Comparing RNN vs SNN"
    print "Micro"
    p_values_rnn_snn_micro = paired_t_test_multiclass_classification(
        rnn_performance_metrics['micro'], snn_performance_metrics['micro'])

    print "Comparing MLP vs RNN"
    print "Macro"
    p_values_mlp_rnn_macro = paired_t_test_multiclass_classification(
        mlp_performance_metrics['macro'], rnn_performance_metrics['macro'])

    print "Comparing MLP vs SNN"
    print "Macro"
    p_values_mlp_snn_macro = paired_t_test_multiclass_classification(
        mlp_performance_metrics['macro'], snn_performance_metrics['macro'])

    print "Comparing RNN vs SNN"
    print "Micro"
    p_values_rnn_snn_macro = paired_t_test_multiclass_classification(
        rnn_performance_metrics['macro'], snn_performance_metrics['macro'])"""

    return mlp_performance_metrics, rnn_performance_metrics, snn_performance_metrics



def evaluate_feed_forward_neural_network(epigenetic_data):


    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Feed Forward Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_size
    print "input data size"
    print input_data_size
    output_size = epigenetic_data.output_size

    """feed_forward_neural_network = MultilayerPerceptron(input_data_size, [256, 128, 64, 32], output_size)"""
    """feed_forward_neural_network = MultilayerPerceptron(128, [256, 128, 64, 32], output_size)"""

    feed_forward_neural_network = MultilayerPerceptron(input_data_size, [128, 64, 32, 16], output_size)
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

    superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[256, 128, 64, 32], [256, 128, 64, 32]],
        [128, 32], output_size)

    """superlayered_neural_network = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [256, 64], output_size)"""

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
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[128, 32],
        output_size=output_size)"""

    """"  Architecture for cancer data with single modaliy"""
    """recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 2, input_step_size=2,
        LSTM_units_state_size=[32, 64], hidden_units=[32, 32],
        output_size=output_size)"""

    """ Architecture for embryo development data
    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=16, input_step_size=16,
        LSTM_units_state_size=[64, 128], hidden_units=[32, 32],
        output_size=output_size)"""

    """recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=8, input_step_size=16,
        LSTM_units_state_size=[32, 128], hidden_units=[64, 32],
        output_size=output_size)"""

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size/4, input_step_size=4,
        LSTM_units_state_size=[64, 128], hidden_units=[128, 64],
        output_size=output_size)

    confussion_matrix, ROC_points, performance_metrics = \
        nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data)

    label_to_one_hot_encoding = epigenetic_data.label_to_one_hot_encoding
    #plot_confussion_matrix_as_heatmap(confussion_matrix, label_to_one_hot_encoding, 'Confusion Matrix for RNN')

    return confussion_matrix, ROC_points, performance_metrics


def get_embryo_development_data(clustering_algorithm):

    noise_mean = 0
    noise_stddev = 1

    print "Noise Characteristics"

    """epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=256, gene_entropy_threshold=6.2) """
    epigenetic_data = EmbryoDevelopmentData(
        num_folds=6, num_folds_hyperparameters_tuning=3, max_num_genes=128, gene_entropy_threshold=6.3)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = EmbryoDevelopmentDataWithClusters(
        num_clusters=2, clustering_algorithm=clustering_algorithm,
        num_folds=6, num_folds_hyperparameters_tuning=3,
        max_num_genes=500, gene_entropy_threshold=6.0)
    epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)
    print epigenetic_data_with_clusters.clusters_size

    epigenetic_data_for_single_cluster = EmbryoDevelopmentDataWithSingleCluster(
        num_clusters=2, clustering_algorithm=clustering_algorithm,
        num_folds=6, num_folds_hyperparameters_tuning=3,
        max_num_genes=320, gene_entropy_threshold=6)
    print"Gene entropy threshold for epigenetic data with single cluster"
    print epigenetic_data_with_clusters.gene_entropy_threshold
    epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)
    print epigenetic_data_with_clusters.clusters_size

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster


def get_cancer_data(noise_mean=0, noise_stddev=0):

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    epigenetic_data = CancerPatientsData(num_folds=10, num_folds_hyperparameters_tuning=3)
    if noise_stddev != 0:
        epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = CancerPatientsDataWithModalities(num_folds=10, num_folds_hyperparameters_tuning=3)
    if noise_stddev != 0:
        epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = CancerPatientsDataDNAMethylationLevels(num_folds=10, num_folds_hyperparameters_tuning=3)
    if noise_stddev != 0:
        epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster


def compare_clustering_on_SNN():
    print "K-means clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('k-means')

    _, _, K_snn_performance_metrics = evaluate_superlayered_neural_network(
        epigenetic_data_with_clusters)

    print "Hierarchical Clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('hierarchical')
    _, _, H_snn_performance_metrics = evaluate_superlayered_neural_network(
        epigenetic_data_with_clusters)


    print "Compare Clustering algorithm on SNN micro"
    p_values_snn_micro = paired_t_test_multiclass_classification(
        K_snn_performance_metrics['micro'], H_snn_performance_metrics['micro'])

    print "Compare Clustering algorithm on SNN macro"
    p_values_snn_macro = paired_t_test_multiclass_classification(
        K_snn_performance_metrics['macro'], H_snn_performance_metrics['macro'])


def compare_clustering_on_MLP():
    print "K-means clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('k-means')

    _, _, K_mlp_performance_metrics = evaluate_feed_forward_neural_network(
        epigenetic_data_for_single_cluster)

    print "Hierarchical Clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('hierarchical')

    _, _, H_mlp_performance_metrics = evaluate_feed_forward_neural_network(
        epigenetic_data_for_single_cluster)

    print "Compare Clustering algorithm on MLP micro"
    p_values_snn_micro = paired_t_test_multiclass_classification(
        K_mlp_performance_metrics['micro'], H_mlp_performance_metrics['micro'])

    print "Compare Clustering algorithm on MLP macro"
    p_values_snn_macro = paired_t_test_multiclass_classification(
        K_mlp_performance_metrics['macro'], H_mlp_performance_metrics['macro'])



def compare_clustering_algorithms():

    print "K-means clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('k-means')
    K_mlp_performance_metrics, K_rnn_performance_metrics, K_snn_performance_metrics = evaluate_neural_network_models(
        epigenetic_data_for_single_cluster, epigenetic_data_with_clusters)

    print "Hierarchical Clustering"
    _, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = \
        get_embryo_development_data('hierarchical')
    H_mlp_performance_metrics, H_rnn_performance_metrics, H_snn_performance_metrics = evaluate_neural_network_models(
        epigenetic_data_for_single_cluster, epigenetic_data_with_clusters)


    print "Compare Clustering algorithm on MLP micro"
    p_values_mlp_micro = paired_t_test_multiclass_classification(
        H_mlp_performance_metrics['micro'], K_mlp_performance_metrics['micro'])

    print "Compare Clustering algorithm on RNN micro"
    p_values_rnn_micro = paired_t_test_multiclass_classification(
        H_rnn_performance_metrics['micro'], K_mlp_performance_metrics['micro'])

    print "Compare Clustering algorithm on SNN micro"
    p_values_snn_micro = paired_t_test_multiclass_classification(
        H_snn_performance_metrics['micro'], H_snn_performance_metrics['micro'])

    print "Compare Clustering algorithm on MLP macro"
    p_values_mlp_macro = paired_t_test_multiclass_classification(
        H_mlp_performance_metrics['macro'], K_mlp_performance_metrics['macro'])

    print "Compare Clustering algorithm on RNN macro"
    p_values_rnn_macro = paired_t_test_multiclass_classification(
        H_rnn_performance_metrics['macro'], K_rnn_performance_metrics['macro'])

    print "Compare Clustering algorithm on SNN macro"
    p_values_snn_macro = paired_t_test_multiclass_classification(
        H_snn_performance_metrics['macro'], K_snn_performance_metrics['macro'])


def compare_RNN_on_different_modalities():

    epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_cancer_data()

    input_data_size = epigenetic_data.input_size
    output_size = epigenetic_data.output_size
    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 4, input_step_size=4,
        LSTM_units_state_size=[32, 128], hidden_units=[128, 32],
        output_size=output_size)

    confussion_matrix, ROC_points, performance_metrics = \
        nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data)

    input_data_size = epigenetic_data_for_single_cluster.input_size
    output_size = epigenetic_data_for_single_cluster.output_size

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size / 2, input_step_size=2,
        LSTM_units_state_size=[32, 64], hidden_units=[32, 32],
        output_size=output_size)

    DNA_confussion_matrix, DNA_ROC_points, DNA_performance_metrics = \
        nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data_for_single_cluster, single_modality=True)

    p_values = paired_t_test_binary_classification(performance_metrics, DNA_performance_metrics)

    print "Comparing RNN on dna methylation"
    print p_values

    plot_mean_ROC_curves_for_two_models(ROC_points, DNA_ROC_points)


def compare_MLP_on_different_modalities():

    epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_cancer_data()

    mlp_confussion_matrix, mlp_ROC_points, mlp_performance_metrics = evaluate_feed_forward_neural_network(
        epigenetic_data)

    DNA_mlp_confussion_matrix, DNA_mlp_ROC_points, DNA_mlp_performance_metrics = evaluate_feed_forward_neural_network(
        epigenetic_data_for_single_cluster)

    p_values = paired_t_test_binary_classification(mlp_performance_metrics, DNA_mlp_performance_metrics)

    print "Comparing MLP on dna methylation"
    print p_values

    plot_mean_ROC_curves_for_two_models(mlp_ROC_points, DNA_mlp_ROC_points)


#compare_RNN_on_different_modalities()

def plot_noise_resistance():

    stddev = [0.1 * i for i in range(20, 40)]
    print stddev


    mlp_points = []
    mlp_std = []

    mlp_points_MCC = []
    mlp_std_MCC = []

    rnn_points = []
    rnn_std = []

    rnn_points_MCC = []
    rnn_std_MCC = []

    snn_points = []
    snn_std = []

    snn_points_MCC = []
    snn_std_MCC = []

    for std in stddev:
        print "Computing for " + str(std)
        epigenetic_data, epigenetic_data_with_clusters, _ = get_cancer_data(noise_mean=0, noise_stddev=std)
        mlp_perf, rnn_perf, snn_perf = evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters)

        mlp_avg = compute_average_performance_metrics_for_binary_classification(mlp_perf)
        mlp_points += [mlp_avg['average']['accuracy']]
        mlp_std += [mlp_avg['std']['accuracy']]

        mlp_points_MCC += [mlp_avg['average']['MCC']]
        mlp_std_MCC += [mlp_avg['std']['MCC']]

        print "MLP Accuracy"
        print mlp_points
        print mlp_std

        print "MLP MCC"
        print mlp_points_MCC
        print mlp_std_MCC

        rnn_avg = compute_average_performance_metrics_for_binary_classification(rnn_perf)
        rnn_points += [rnn_avg['average']['accuracy']]
        rnn_std += [rnn_avg['std']['accuracy']]

        rnn_points_MCC += [rnn_avg['average']['MCC']]
        rnn_std_MCC += [rnn_avg['std']['MCC']]

        print "RNN Accuracy"
        print rnn_points
        print rnn_std

        print "RNN MCC"
        print rnn_points_MCC
        print rnn_std_MCC

        snn_avg = compute_average_performance_metrics_for_binary_classification(snn_perf)
        snn_points += [snn_avg['average']['accuracy']]
        snn_std += [snn_avg['std']['accuracy']]

        snn_points_MCC += [snn_avg['average']['MCC']]
        snn_std_MCC += [snn_avg['std']['MCC']]

        print 'SNN Accuracy'
        print snn_points
        print snn_std

        print 'SNN MCC'
        print snn_points_MCC
        print snn_std_MCC






plot_noise_resistance()

#compare_clustering_on_MLP()
#compare_clustering_algorithms()


#epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_embryo_development_data('k')
#plot_dendogram(epigenetic_data_with_clusters.geneId_to_expression_levels)



#evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters)


#compare_MLP_on_different_modalities()

#epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_cancer_data()
#evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters)


#print "Evaluate for single modality"
#evaluate_neural_network_models(epigenetic_data_for_single_cluster, epigenetic_data_with_clusters)


