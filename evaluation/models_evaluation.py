from epigenetic_data.embryo_development_data.embryo_development_data import \
    EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters, EmbryoDevelopmentDataWithSingleCluster
from epigenetic_data.cancer_data.cancer_data import CancerData, CancerDataWithClusters

from neural_network_models.feedforward_neural_network import FeedforwardNeuralNetwork
from neural_network_models.recurrent_neural_network_modified import RecurrentNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

from evaluation.nested_cross_validation import nested_cross_validation_on_MLP, nested_cross_validation_on_SNN, nested_cross_validation_on_RNN


def evaluate_neural_network_models(epigenetic_data, epigenetic_data_with_clusters):

    average_accuracy, confussion_matrix = evaluate_feed_forward_neural_network(epigenetic_data)
    average_accuracy, confussion_matrix = evaluate_recurrent_neural_network(epigenetic_data)
    average_accuracy, confussion_matrix = evaluate_superlayered_neural_network(epigenetic_data_with_clusters)


def evaluate_feed_forward_neural_network(epigenetic_data):

    input_data_size = epigenetic_data.input_data_size
    output_size = epigenetic_data.output_size

    feed_forward_neural_network = FeedforwardNeuralNetwork(input_data_size, [256, 128, 64, 32], output_size)
    average_accuracy, confussion_matrix = nested_cross_validation_on_MLP(feed_forward_neural_network, epigenetic_data)

    return average_accuracy, confussion_matrix


def evaluate_superlayered_neural_network(epigenetic_data_with_clusters):

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

    input_data_size = epigenetic_data.input_data_size
    output_size = epigenetic_data.output_size

    feed_forward_neural_network = RecurrentNeuralNetwork(
        input_sequence_length=16, input_step_size=8,
        LSTMs_state_size=[32, 128], hidden_units=[32],
        output_size=output_size)

    average_accuracy, confussion_matrix = nested_cross_validation_on_RNN(feed_forward_neural_network, epigenetic_data)

    return average_accuracy, confussion_matrix




"""epigenetic_data = EmbryoDevelopmentData(
        num_folds=3, num_folds_hyperparameters_tuning=3, max_num_genes=256, gene_entropy_threshold=6.1)

average_accuracy, confussion_matrix = evaluate_recurrent_neural_network(epigenetic_data)
print average_accuracy, confussion_matrix"""



epigenetic_data = EmbryoDevelopmentDataWithSingleCluster(
        num_clusters=2, clustering_algorithm='k-means',
        num_folds=5, num_folds_hyperparameters_tuning=3,
        max_num_genes=250, gene_entropy_threshold=6.1)

average_accuracy, confussion_matrix = evaluate_recurrent_neural_network(epigenetic_data)

print average_accuracy, confussion_matrix
