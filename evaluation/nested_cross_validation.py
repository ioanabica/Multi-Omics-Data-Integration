import matplotlib.pyplot as plt
import numpy as np

from hyperparameters_tuning import choose_hyperparameters, choose_hyperparameters_for_RNN


def nested_cross_validation_on_MLP(network, epigenetic_data):

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    output_size = epigenetic_data.output_size
    keys = k_fold_datasets.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()

    """ Outer cross-validation """

    for key in keys:
        print "key number" + str(key)

        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.05
        weight_decay = 0.05
        keep_probability = 0.8

        print "Learning rate" + str(learning_rate)
        print "Weight decay" + str(weight_decay)
        print "Keep_probability" + str(keep_probability)

        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, ffnn_confussion_matrix = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)

        print ffnn_confussion_matrix
        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, ffnn_confussion_matrix)

    average_validation_accuracy = np.mean(validation_accuracy_list)

    return average_validation_accuracy, confussion_matrix


def nested_cross_validation_on_SNN(network, epigenetic_data_with_clusters):

    k_fold_datasets_with_clusters, k_fold_datasets_hyperparameters_tuning = \
        epigenetic_data_with_clusters.get_k_fold_datasets()

    output_size = epigenetic_data_with_clusters.output_size

    keys = k_fold_datasets_with_clusters.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()

    """ Outer cross-validation """

    for key in keys:

        """ Inner cross-validation """
        #learning_rate, weight_decay, keep_probability = choose_hyperparameters(
            #network, k_fold_datasets_hyperparameters_tuning[key])

        learning_rate = 0.05
        weight_decay = 0.02
        keep_probability = 0.6

        training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
        validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]

        validation_accuracy, snn_confussion_matrix = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, weight_decay, keep_probability)

        print snn_confussion_matrix

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, snn_confussion_matrix)

    average_validation_accuracy = np.mean(validation_accuracy_list)

    return average_validation_accuracy, confussion_matrix


def nested_cross_validation_on_RNN(network, epigenetic_data):

    k_fold_datasets, k_fold_datasets_hyperparameters_tuning = epigenetic_data.get_k_fold_datasets()
    output_size = epigenetic_data.output_size

    keys = k_fold_datasets.keys()

    confussion_matrix = np.zeros(shape=(output_size, output_size))
    validation_accuracy_list = list()

    """ Outer cross-validation """

    for key in keys:
        print "key number" + str(key)

        #learning_rate, weight_decay, keep_probability = choose_hyperparameters_for_RNN(
            #network, k_fold_datasets_hyperparameters_tuning[key])
        learning_rate = 0.0001
        weight_decay = 0.01
        keep_probability = 0.7

        print "Learning rate" + str(learning_rate)
        print "Weight decay" + str(weight_decay)
        print "Keep_probability" + str(keep_probability)

        training_dataset = k_fold_datasets[key]["training_dataset"]

        print len(training_dataset["training_data"])
        print len(training_dataset["training_data"][0])

        validation_dataset = k_fold_datasets[key]["validation_dataset"]
        print len(validation_dataset["validation_data"])

        validation_accuracy, rnn_confussion_matrix = network.train_and_evaluate(
            training_dataset, validation_dataset,
            learning_rate, weight_decay, keep_probability)
        print rnn_confussion_matrix

        validation_accuracy_list.append(validation_accuracy)
        confussion_matrix = np.add(confussion_matrix, rnn_confussion_matrix)

    average_validation_accuracy = np.mean(validation_accuracy_list)

    return average_validation_accuracy, confussion_matrix


def plot_validation_accuracy(MLP_validation_accuracy, SNN_validation_accuracy, RNN_validation_accuracy):
    plt.figure(1)

    print "MLP accuracy"
    print np.mean(MLP_validation_accuracy)

    print "SNN accuracy"
    print np.mean(SNN_validation_accuracy)

    print "RNN accuracy"
    print np.mean(RNN_validation_accuracy)

    MLP, = plt.plot(range(num_folds), MLP_validation_accuracy, 'r-', label='Multilayer Perceptron NN')
    SNN, = plt.plot(range(num_folds), SNN_validation_accuracy, 'g-', label='Superlayered NN')
    RNN, = plt.plot(range(num_folds), RNN_validation_accuracy, 'b-', label='Recurrent NN')
    plt.legend(handles=[MLP, SNN, RNN], loc='lower right')

    axes = plt.gca()
    axes.set_ylim([0, 110])

    plt.title('K-fold Cross-Validation', fontsize=20)
    plt.xlabel('Fold Number', fontsize=18)
    plt.ylabel('Test Accuracy', fontsize=20)

    plt.show()

def plot_losses(MLP_losses, SNN_losses, RNN_losses):
    plt.figure(1)

    plt.subplot(121)
    plt.plot(MLP_losses[0], 'r')

    plt.title('Multilayer Perceptron Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)


    plt.subplot(122)
    plt.plot(SNN_losses[0], 'g')

    plt.title('Superlayered Neural Network Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.show()

    plt.figure(2)

    plt.plot(RNN_losses[0], 'b')

    plt.title('Recurrent Neural Network Training Loss', fontsize=20)
    plt.xlabel('Iteration number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.show()


