import numpy as np

from gene_clustering import hierarchical_clustering
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

num_classes = 7
num_clusters = 2
#num_genes = 64
num_shifted_genes = 16
training_examples_for_class = 10
validation_examples_for_class = 2

cluster_1_num_genes = 256
cluster_2_num_genes = 256

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


def create_data_point(class_id, class_id_to_shifted_genes, num_genes, mean):
    stddev = 1
    data_point = np.random.normal(mean, stddev, num_genes)
    shifted_mean = 5

    shifted_genes = class_id_to_shifted_genes[class_id]

    for shifted_gene in shifted_genes:
        data_point[shifted_gene] = np.random.normal(mean+shifted_mean, stddev, 1)

    return data_point[:num_genes]


def create_one_hot_encoding(class_id):
    one_hot_encoding = [0] * num_classes
    one_hot_encoding[class_id] = 1.0

    return one_hot_encoding


def create_training_dataset(class_id_to_shifted_genes, cluster_id_to_class_id_to_gene_expressions_mean_value, clusters_size):

    training_dataset = dict()
    training_dataset["training_data"] = dict()

    cluster_1_num_genes = clusters_size[0]
    cluster_2_num_genes = clusters_size[1]

    cluster_1_training_data = np.ndarray(shape=(num_training_examples, cluster_1_num_genes),
                               dtype=np.float32)
    cluster_2_training_data = np.ndarray(shape=(num_training_examples, cluster_2_num_genes),
                               dtype=np.float32)

    training_labels = np.ndarray(shape=(num_training_examples, num_classes),
                                 dtype=np.float32)

    for class_id in range(num_classes):
        for index in range(training_examples_for_class):

            cluster_1_training_data[class_id * training_examples_for_class + index, :] = \
                normalize_data(create_data_point(class_id, class_id_to_shifted_genes,
                    cluster_1_num_genes, cluster_id_to_class_id_to_gene_expressions_mean_value[0][class_id]))

            cluster_2_training_data[class_id * training_examples_for_class + index, :] = \
                normalize_data(create_data_point(class_id, class_id_to_shifted_genes,
                    cluster_2_num_genes, cluster_id_to_class_id_to_gene_expressions_mean_value[1][class_id]))

            training_labels[class_id * training_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    permutation = np.random.permutation(len(training_labels))

    training_dataset["training_data"][0] = cluster_1_training_data[permutation]
    training_dataset["training_data"][1] = cluster_2_training_data[permutation]
    training_dataset["training_labels"] = training_labels[permutation]

    return training_dataset


def create_validation_dataset(class_id_to_shifted_genes, cluster_id_to_class_id_to_gene_expressions_mean_value, clusters_size):

    validation_dataset = dict()
    validation_dataset["validation_data"] = dict()

    cluster_1_num_genes = clusters_size[0]
    cluster_2_num_genes = clusters_size[1]

    cluster_1_validation_data = np.ndarray(shape=(num_validation_examples, cluster_1_num_genes),
                               dtype=np.float32)
    cluster_2_validation_data = np.ndarray(shape=(num_validation_examples, cluster_2_num_genes),
                               dtype=np.float32)

    validation_labels = np.ndarray(shape=(num_validation_examples, num_classes),
                                 dtype=np.float32)

    for class_id in range(num_classes):
        for index in range(validation_examples_for_class):
            cluster_1_validation_data[class_id * validation_examples_for_class + index, :] = \
                normalize_data(create_data_point(class_id, class_id_to_shifted_genes,
                    cluster_1_num_genes, cluster_id_to_class_id_to_gene_expressions_mean_value[0][class_id]))

            cluster_2_validation_data[class_id * validation_examples_for_class + index, :] = \
                normalize_data(create_data_point(class_id, class_id_to_shifted_genes,
                    cluster_2_num_genes, cluster_id_to_class_id_to_gene_expressions_mean_value[1][class_id]))

            validation_labels[class_id * validation_examples_for_class + index, :] = create_one_hot_encoding(class_id)

    validation_dataset["validation_data"][0] = cluster_1_validation_data
    validation_dataset["validation_data"][1] = cluster_2_validation_data
    validation_dataset["validation_labels"] = validation_labels

    return validation_dataset


def create_expression_profile(
        gene_id, class_id_to_shifted_genes, class_id_to_gene_expression_mean_value):

    class_ids = class_id_to_shifted_genes.keys()
    expression_profile = list()
    stddev = 1
    shifted_mean = 5

    for class_id in class_ids:
        gene_expression_level = np.random.normal(class_id_to_gene_expression_mean_value[class_id], stddev)
        if gene_id in class_id_to_shifted_genes[class_id]:
            gene_expression_level = np.random.normal(
                class_id_to_gene_expression_mean_value[class_id] + shifted_mean, stddev)
        expression_profile.append(gene_expression_level)

    return expression_profile


def create_data_for_clustering(
        class_id_to_shifted_genes,
        cluster_id_to_class_id_to_gene_expression_mean_value,
        clusters_size):

    gene_id_to_expression_profile = dict()

    cluster_1_num_genes = clusters_size[0]
    cluster_2_num_genes = clusters_size[1]

    total_num_genes = cluster_1_num_genes + cluster_2_num_genes

    for gene_id in range(total_num_genes):
        if gene_id < cluster_1_num_genes:
            gene_id_to_expression_profile[gene_id] = create_expression_profile(
                gene_id, class_id_to_shifted_genes, cluster_id_to_class_id_to_gene_expression_mean_value[0])
        else:
            gene_id_to_expression_profile[gene_id] = create_expression_profile(
                gene_id - cluster_1_num_genes,
                class_id_to_shifted_genes,
                cluster_id_to_class_id_to_gene_expression_mean_value[1])

    return gene_id_to_expression_profile


def create_cluster_id_to_class_id_to_gene_expressions_mean_value(num_clusters, num_classes):

    cluster_id_to_class_id_to_gene_expression_mean_value = dict()

    for cluster_index in range(num_clusters):
        class_id_gene_expressions_mean_value = dict()
        for class_index in range(num_classes):
            class_id_gene_expressions_mean_value[class_index] = np.random.random_integers(0, 60)
        cluster_id_to_class_id_to_gene_expression_mean_value[cluster_index] = class_id_gene_expressions_mean_value

    return cluster_id_to_class_id_to_gene_expression_mean_value


def create_class_id_to_shifted_genes(num_classes, num_genes, num_shifted_genes):

    class_id_to_shifted_genes = dict()

    for index in range(num_classes):
        shifted_genes = np.random.choice(range(num_genes), num_shifted_genes, replace=False)
        class_id_to_shifted_genes[index] = shifted_genes

    return class_id_to_shifted_genes


class SyntheticDataWithClusters(object):

    class_id_to_shifted_genes = create_class_id_to_shifted_genes(
        num_classes, min(cluster_1_num_genes, cluster_2_num_genes), num_shifted_genes)
    print class_id_to_shifted_genes

    cluster_id_to_class_id_to_gene_expression_mean_value = create_cluster_id_to_class_id_to_gene_expressions_mean_value(
        num_clusters, num_classes)

    training_dataset_with_clusters = create_training_dataset(
        class_id_to_shifted_genes,
        cluster_id_to_class_id_to_gene_expression_mean_value,
        [cluster_1_num_genes, cluster_2_num_genes])

    validation_dataset_with_clusters = create_validation_dataset(
        class_id_to_shifted_genes,
        cluster_id_to_class_id_to_gene_expression_mean_value,
        [cluster_1_num_genes, cluster_2_num_genes])

    data_for_clustering = create_data_for_clustering(
        class_id_to_shifted_genes,
        cluster_id_to_class_id_to_gene_expression_mean_value,
        [cluster_1_num_genes, cluster_2_num_genes])


    superlayered_nn = SuperlayeredNeuralNetwork(
        [cluster_1_num_genes, cluster_2_num_genes],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [128, 32],
        num_classes)


    learning_rate = 0.05
    weight_decay = 0.05
    keep_probability = 0.5


    validation_accuraty, confussion_matrix = superlayered_nn.train_and_evaluate(
        training_dataset_with_clusters, validation_dataset_with_clusters, learning_rate, weight_decay, keep_probability)


    print confussion_matrix