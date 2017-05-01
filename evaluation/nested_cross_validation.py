import matplotlib.pyplot as plt
import numpy as np
from evaluation_metrics import *
import matplotlib.pyplot as plt

from hyperparameters_tuning import choose_hyperparameters, choose_hyperparameters_for_RNN


def nested_cross_validation_on_MLP(network, epigenetic_data):

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    output_size = epigenetic_data.output_size
    keys = k_fold_datasets.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()
    ROC_points = dict()

    macro_average = dict()
    micro_average = dict()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data.label_to_one_hot_encoding)
    performance_metrics = dict()

    """ Outer cross-validation """

    for key in keys:
        print "key number" + str(key)

        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.05
        weight_decay = 0.01
        keep_probability = 0.75

        print "Learning rate" + str(learning_rate)
        print "Weight decay" + str(weight_decay)
        print "Keep_probability" + str(keep_probability)

        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, ffnn_confussion_matrix, MLP_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            ffnn_confussion_matrix, class_id_to_class_symbol)

        print performance_metrics[key]

        """micro_average[key] = compute_micro_average(performance_metrics[key])
        macro_average[key] = compute_macro_average(performance_metrics[key])

        micro_average[key]['accuracy'] = validation_accuracy
        macro_average[key]['accuracy'] = validation_accuracy"""

        print ffnn_confussion_matrix
        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, ffnn_confussion_matrix)
        ROC_points[key] = MLP_ROC_points

    average_validation_accuracy = np.mean(validation_accuracy_list)
    average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)

    """performance_metrics['micro'] = micro_average
    performance_metrics['macro'] = macro_average

    print "Micro"
    average_micro = compute_performance_metrics_for_multiclass_classification(micro_average)
    print "Macro"
    average_macro = compute_performance_metrics_for_multiclass_classification(macro_average)"""

    return confussion_matrix, ROC_points, performance_metrics


def nested_cross_validation_on_SNN(network, epigenetic_data_with_clusters):

    k_fold_datasets_with_clusters, k_fold_datasets_hyperparameters_tuning = \
        epigenetic_data_with_clusters.get_k_fold_datasets()

    output_size = epigenetic_data_with_clusters.output_size

    keys = k_fold_datasets_with_clusters.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()
    ROC_points = dict()

    macro_average = dict()
    micro_average = dict()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data_with_clusters.label_to_one_hot_encoding)
    performance_metrics = dict()

    """ Outer cross-validation """

    for key in keys:

        """ Inner cross-validation """
        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.05
        weight_decay = 0.001
        keep_probability = 0.75

        print learning_rate
        print weight_decay
        print keep_probability

        training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
        validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]

        validation_accuracy, snn_confussion_matrix, snn_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, weight_decay, keep_probability)

        print snn_confussion_matrix

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            snn_confussion_matrix, class_id_to_class_symbol)

        print performance_metrics[key]

        """micro_average[key] = compute_micro_average(performance_metrics[key])
        macro_average[key] = compute_macro_average(performance_metrics[key])

        micro_average[key]['accuracy'] = validation_accuracy
        macro_average[key]['accuracy'] = validation_accuracy"""

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, snn_confussion_matrix)
        ROC_points[key] = snn_ROC_points

    average_validation_accuracy = np.mean(validation_accuracy_list)
    average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)

    """
    print "Micro"
    performance_metrics['micro'] = micro_average
    print "Macro"
    performance_metrics['macro'] = macro_average

    print "Micro"
    average_micro = compute_performance_metrics_for_multiclass_classification(micro_average)
    print "Macro"
    average_macro = compute_performance_metrics_for_multiclass_classification(macro_average)"""

    return confussion_matrix, ROC_points, performance_metrics


def nested_cross_validation_on_RNN(network, epigenetic_data, single_modality=False):

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    output_size = epigenetic_data.output_size

    keys = k_fold_datasets.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()

    class_id_to_class_symbol = compute_class_id_to_class_symbol(epigenetic_data.label_to_one_hot_encoding)

    performance_metrics = dict()
    macro_average = dict()
    micro_average = dict()


    ROC_points = dict()

    """ Outer cross-validation """

    for key in keys:
        print "key number" + str(key)

        #learning_rate, weight_decay, keep_probability = choose_hyperparameters_for_RNN(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        # Hyperparameters for Embryo Development data
        #learning_rate = 0.0001
        #weight_decay = 0.001
        #keep_probability = 0.7

        learning_rate = 0.0001
        weight_decay = 0.001
        keep_probability = 0.7

        if single_modality:
            learning_rate = 0.0005
            weight_decay = 0.001
            keep_probability = 1

        print "Learning rate" + str(learning_rate)
        print "Weight decay" + str(weight_decay)
        print "Keep_probability" + str(keep_probability)

        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, rnn_confussion_matrix, rnn_ROC_points = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)
        print rnn_confussion_matrix

        performance_metrics[key] = compute_evaluation_metrics_for_each_class(
            rnn_confussion_matrix, class_id_to_class_symbol)
        print performance_metrics[key]

        """micro_average[key] = compute_micro_average(performance_metrics[key])
        macro_average[key] = compute_macro_average(performance_metrics[key])

        micro_average[key]['accuracy'] = validation_accuracy
        macro_average[key]['accuracy'] = validation_accuracy

        performance_metrics['micro'] = micro_average
        performance_metrics['macro'] = macro_average"""

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, rnn_confussion_matrix)
        ROC_points[key] = rnn_ROC_points

    average_performance_metrics = compute_average_performance_metrics_for_binary_classification(performance_metrics)
    average_validation_accuracy = np.mean(validation_accuracy_list)


    """print "micro"
    average_micro = compute_performance_metrics_for_multiclass_classification(micro_average)
    print "macro"
    average_macro = compute_performance_metrics_for_multiclass_classification(macro_average)

    performance_metrics['micro'] = micro_average
    performance_metrics['macro'] = macro_average"""

    return confussion_matrix, ROC_points, performance_metrics
