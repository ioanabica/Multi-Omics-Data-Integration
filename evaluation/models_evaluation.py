import numpy as np
import math

from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster
from epigenetic_data.cancer_data.cancer_data import CancerData, CancerDataWithClusters, CancerDataWithDNAMethylationLevels

from neural_network_models.feedforward_neural_network import FeedforwardNeuralNetwork
from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN


def evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster):

    average_accuracy, confussion_matrix = evaluate_recurrent_neural_network(epigenetic_data)
    print average_accuracy, confussion_matrix

    average_accuracy, confussion_matrix = evaluate_superlayered_neural_network(epigenetic_data_with_clusters)
    print average_accuracy, confussion_matrix

    average_accuracy, confussion_matrix = evaluate_feed_forward_neural_network(epigenetic_data)
    print average_accuracy, confussion_matrix


def evaluate_feed_forward_neural_network(epigenetic_data):


    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Feed Forward Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_data_size
    print "input data size"
    print input_data_size
    output_size = epigenetic_data.output_size

    feed_forward_neural_network = FeedforwardNeuralNetwork(input_data_size, [256, 128, 64, 32], output_size)
    average_accuracy, confussion_matrix = nested_cross_validation_on_MLP(feed_forward_neural_network, epigenetic_data)

    return average_accuracy, confussion_matrix


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

    average_accuracy, confussion_matrix = nested_cross_validation_on_SNN(
        superlayered_neural_network, epigenetic_data_with_clusters)

    return average_accuracy, confussion_matrix


def evaluate_recurrent_neural_network(epigenetic_data):

    print "-------------------------------------------------------------------------------"
    print "-------------------Evaluating Recurrent Neural Network-------------------------"
    print "-------------------------------------------------------------------------------"

    input_data_size = epigenetic_data.input_data_size
    output_size = epigenetic_data.output_size

    """
        Architecture for cancer data

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=input_data_size/4, input_step_size=4,
        LSTMs_state_size=[32, 128], hidden_units=[64, 32],
        output_size=output_size)"""

    recurrent_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=16, input_step_size=16,
        LSTMs_state_size=[64, 128], hidden_units=[128, 64],
        output_size=output_size)

    average_accuracy, confussion_matrix = nested_cross_validation_on_RNN(recurrent_neural_network, epigenetic_data)

    return average_accuracy, confussion_matrix


def get_embryo_development_data():

    noise_mean = 0
    noise_stddev = 0.01

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    epigenetic_data = EmbryoDevelopmentData(
        num_folds=5, num_folds_hyperparameters_tuning=3, max_num_genes=256, gene_entropy_threshold=6.25)
    #epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = EmbryoDevelopmentDataWithClusters(
        num_clusters=2, clustering_algorithm='hierarchical_clustering',
        num_folds=5, num_folds_hyperparameters_tuning=3,
        max_num_genes=120, gene_entropy_threshold=6.3)
    #epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = EmbryoDevelopmentDataWithSingleCluster(
        num_clusters=2, clustering_algorithm='hierarchical_clustering',
        num_folds=5, num_folds_hyperparameters_tuning=3,
        max_num_genes=180, gene_entropy_threshold=6.3)
    #epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster

def get_cancer_data():

    noise_mean = 0
    noise_stddev = 0.03

    print "Noise Characteristics"
    print noise_mean
    print noise_stddev

    epigenetic_data = CancerData(num_folds = 4, num_folds_hyperparameters_tuning=3)
    epigenetic_data.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_with_clusters = CancerDataWithClusters(num_folds=4, num_folds_hyperparameters_tuning=3)
    epigenetic_data_with_clusters.add_Gaussian_noise(noise_mean, noise_stddev)

    epigenetic_data_for_single_cluster = CancerDataWithDNAMethylationLevels(num_folds = 4, num_folds_hyperparameters_tuning=3)
    epigenetic_data_for_single_cluster.add_Gaussian_noise(noise_mean, noise_stddev)

    return epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster


epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster = get_embryo_development_data()
evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters, epigenetic_data_for_single_cluster)


def compute_class_id_to_class_symbol(label_to_one_hot_encoding):
    class_id_to_class_symbol = dict()
    labels = label_to_one_hot_encoding.keys()
    for label in labels:
        class_id = np.argmax(label_to_one_hot_encoding[label])
        class_id_to_class_symbol[class_id] = label
    return class_id_to_class_symbol


def compute_evaluation_metrics_for_each_class(confussion_matrix, class_id_to_class_symbol):

    confussion_matrix = np.array(confussion_matrix)

    class_symbol_to_evaluation_metrics = dict()
    class_ids = class_id_to_class_symbol.keys()

    for class_id in class_ids:
        class_symbol = class_id_to_class_symbol[class_id]
        class_symbol_to_evaluation_metrics[class_symbol] = dict()

    sum_over_rows = confussion_matrix.sum(axis=1)
    sum_over_columns = confussion_matrix.sum(axis=0)


    for class_id in class_ids:
        class_symbol = class_id_to_class_symbol[class_id]

        true_positives = confussion_matrix[class_id][class_id]
        false_positives = sum_over_columns[class_id] - true_positives


        false_negatives = sum_over_rows[class_id] - true_positives
        true_negatives = sum_over_columns.sum() - sum_over_columns[class_id] - \
                         sum_over_rows[class_id] + confussion_matrix[class_id][class_id]

        precision = true_positives / (true_positives + false_positives)

        recall =  true_positives / (true_positives + false_negatives)

        f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

        MCC = (true_positives * true_negatives - false_positives * false_negatives) / \
              (math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * \
                         (true_negatives + false_positives) * (true_negatives + false_negatives)))

        class_symbol_to_evaluation_metrics[class_symbol]['true_positives'] = true_positives
        class_symbol_to_evaluation_metrics[class_symbol]['false_positives'] = false_positives

        class_symbol_to_evaluation_metrics[class_symbol]['true_negatives'] = true_negatives
        class_symbol_to_evaluation_metrics[class_symbol]['false_negatives'] = false_negatives

        class_symbol_to_evaluation_metrics[class_symbol]['precision'] = precision
        class_symbol_to_evaluation_metrics[class_symbol]['recall'] = recall
        class_symbol_to_evaluation_metrics[class_symbol]['f1_score'] = f1_score
        class_symbol_to_evaluation_metrics[class_symbol]['MCC'] = MCC

    return class_symbol_to_evaluation_metrics








