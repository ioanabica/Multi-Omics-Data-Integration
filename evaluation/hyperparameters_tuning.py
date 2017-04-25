#from embryo_development_data import EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters
import matplotlib.pyplot as plt
import numpy as np

from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

#learning_rate_values = [0.01, 0.02, 0.03, 0.04, 0.05]
#learning_rate_values_for_RNN = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

#weight_decay_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

keep_probability_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
learning_rate_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1]
weight_decay_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]




def choose_hyperparameters(network, k_fold_datasets):

    keep_probability = choose_keep_probability(network, k_fold_datasets, 0.05, 0.01)
    weight_decay = choose_weight_decay(network, k_fold_datasets, 0.05, keep_probability)
    learning_rate = choose_learning_rate(network, k_fold_datasets, weight_decay, keep_probability)

    return learning_rate, weight_decay, keep_probability


def choose_hyperparameters_for_RNN(network, k_fold_datasets):

    keep_probability = choose_keep_probability(network, k_fold_datasets, 0.01, 0.001)
    weight_decay = choose_weight_decay(network, k_fold_datasets, 0.001, keep_probability)
    learning_rate = choose_learning_rate_for_RNN(network, k_fold_datasets, weight_decay, keep_probability)

    return learning_rate, weight_decay, keep_probability


def choose_keep_probability(network, k_fold_datasets, fixed_learning_rate, fixed_weight_decay):
    max_validation_accuracy = 0
    best_keep_probability = 0

    print "Choosing dropout"

    mean_error_rates = []
    std_error_rates = []

    for keep_probability in keep_probability_values:
        print "Computing for keep probability " + str(keep_probability)
        error_rates, average_validation_accuracy = inner_cross_validation(
            network, k_fold_datasets, fixed_learning_rate, fixed_weight_decay, keep_probability)

        average_error_rate = np.mean(error_rates)
        std_error_rate = np.std(error_rates)

        mean_error_rates += [average_error_rate]
        std_error_rates += [std_error_rate]

        if average_validation_accuracy > max_validation_accuracy:
            best_keep_probability = keep_probability

    return best_keep_probability, mean_error_rates, std_error_rates


def choose_weight_decay(network, k_fold_datasets, fixed_learning_rate, fixed_keep_probability):

    max_validation_accuracy = 0
    best_weight_decay = 0

    "Choosing weight decay"

    mean_error_rates = []
    std_error_rates = []

    for weight_decay in weight_decay_values:
        print "Computing for weight decay" + str(weight_decay)
        error_rates, average_validation_accuracy = inner_cross_validation(
            network, k_fold_datasets, fixed_learning_rate, weight_decay, fixed_keep_probability)

        average_error_rate = np.mean(error_rates)
        std_error_rate = np.std(error_rates)

        mean_error_rates += [average_error_rate]
        std_error_rates += [std_error_rate]

        if average_validation_accuracy > max_validation_accuracy:
            best_weight_decay = weight_decay

    return best_weight_decay, mean_error_rates, std_error_rates


def choose_learning_rate(network, k_fold_datasets, fixed_weight_decay, fixed_keep_probability):

    max_validation_accuracy = 0
    best_learning_rate = 0

    mean_error_rates = []
    std_error_rates = []

    for learning_rate in learning_rate_values:
        print "Computing for learning rate " + str(learning_rate)
        error_rates, average_validation_accuracy = inner_cross_validation(
            network, k_fold_datasets, learning_rate, fixed_weight_decay, fixed_keep_probability)

        average_error_rate = np.mean(error_rates)
        std_error_rate = np.std(error_rates)

        mean_error_rates += [average_error_rate]
        std_error_rates += [std_error_rate]

        if average_validation_accuracy >  max_validation_accuracy:
            best_learning_rate = learning_rate

    return best_learning_rate, mean_error_rates, std_error_rates


def choose_learning_rate_for_RNN(network, k_fold_datasets, fixed_weight_decay, fixed_keep_probability):

    max_validation_accuracy = 0
    best_learning_rate = 0

    for learning_rate in learning_rate_values_for_RNN:

        average_validation_accuracy = inner_cross_validation(
            network, k_fold_datasets, learning_rate, fixed_weight_decay, fixed_keep_probability)

        if average_validation_accuracy > max_validation_accuracy:
            best_learning_rate = learning_rate

    return best_learning_rate


def inner_cross_validation(network, k_fold_datasets, learning_rate, weight_decay, keep_probability):

    keys = k_fold_datasets.keys()
    validation_accuracy_list = list()
    error_rate_list = list()

    print learning_rate
    print weight_decay
    print keep_probability

    for key in keys:
        print "key number" + str(key)
        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, _, _ = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)

        error_rate = float(1.0 - float(validation_accuracy)/100.0)
        print error_rate

        validation_accuracy_list.append(validation_accuracy)
        error_rate_list.append(error_rate)

    average_validation_accuracy = np.mean(validation_accuracy_list)

    return error_rate_list, average_validation_accuracy



def superlayeredNN_hyperparameters_tuning():

    #epigeneticsData = EmbryoDevelopmentDataWithClusters(2, 3, 256, 6)

    epigeneticsData = CancerDataWithModalities(5)
    training_dataset, validation_dataset, test_dataset = epigeneticsData.get_training_validation_test_datasets()
    clusters_size = epigeneticsData.clusters_size
    output_size = epigeneticsData.output_size

    """superlayered_nn = SuperlayeredNeuralNetwork(
            [clusters_size[0], clusters_size[1]],
            [[512, 256, 128, 64], [512, 256, 128, 64]],
            [128, 64],
            output_size)"""

    superlayered_nn = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[64, 256, 128, 16], [64, 256, 128, 16]],
        [64, 16],
        output_size)

    learning_rate = 0.02
    weight_decay = 0.01
    keep_probability = 0.50

    training_accuracy, validation_accuracy, steps_list, test_accuracy = superlayered_nn.train_validate_test(
        training_dataset, validation_dataset, test_dataset, learning_rate, weight_decay, keep_probability)

    plot_training_accuracy_and_validation_accuracy(training_accuracy, validation_accuracy, steps_list)



def plot_training_accuracy_and_validation_accuracy(training_accuracy, validation_accuracy, steps_list):
    plt.figure(1)

    print steps_list
    print validation_accuracy

    trainingPlot,  = plt.plot(steps_list, training_accuracy, 'r', label='Training Accuracy')
    validationPlot,  = plt.plot(steps_list, validation_accuracy, 'b', label='Validation Accuracy')
    # plt.legend([trainingPlot,validationPlot],['Training','Validation'])
    plt.legend(handles=[trainingPlot, validationPlot], loc='lower right')
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.ylim([0, 101])
    plt.show()



