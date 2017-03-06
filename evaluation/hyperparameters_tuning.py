#from embryo_development_data import EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters
import matplotlib.pyplot as plt

from epigenetic_data.cancer_data.cancer_data import CancerData, CancerDataWithClusters
from neural_network_models.recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.feedforward_neural_network import FeedforwardNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork

learning_rate_values = [0.01, 0.02, 0.03, 0.04, 0.05]
learning_rate_values_for_RNN = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

weight_decay_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

keep_probability_values = [0.25, 0.35, 0.5, 0.75, 0.8]


def choose_hyperparameters(network, training_dataset, validation_dataset):

    dropout = choose_dropout(network, training_dataset, validation_dataset, 0.05, 0.01)
    weight_decay = choose_weight_decay(network, training_dataset, validation_dataset, 0.05, dropout)
    learning_rate = choose_learning_rate(network, training_dataset, validation_dataset, weight_decay, dropout)

    return learning_rate, weight_decay, dropout


def choose_hyperparameters_for_RNN(network, training_dataset, validation_dataset):

    dropout = choose_dropout(network, training_dataset, validation_dataset, 0.05, 0.01)
    weight_decay = choose_weight_decay(network, training_dataset, validation_dataset, 0.05, dropout)
    learning_rate = choose_learning_rate(network, training_dataset, validation_dataset, weight_decay, dropout)

    return learning_rate, weight_decay, dropout


def choose_dropout(network, training_dataset, validation_dataset, fixed_learning_rate, fixed_weight_decay):
    max_validation_accuracy = 0
    best_keep_probability = 0

    for keep_probability in keep_probability_values:
        validation_accuracy =  network.train_and_validate(
            training_dataset, validation_dataset, fixed_learning_rate, fixed_weight_decay, keep_probability)
        if validation_accuracy > max_validation_accuracy:
            best_keep_probability = keep_probability

    return best_keep_probability


def choose_weight_decay(network, training_dataset, validation_dataset, fixed_learning_rate, fixed_dropout):

    max_validation_accuracy = 0
    best_weight_decay = 0

    for weight_decay in weight_decay_values:
        validation_accuracy = network.train_and_validate(
            training_dataset, validation_dataset, fixed_learning_rate, weight_decay, fixed_dropout)
        if validation_accuracy > max_validation_accuracy:
            best_weight_decay = weight_decay

    return best_weight_decay


def choose_learning_rate(network, training_dataset, validation_dataset, fixed_weight_decay, fixed_dropout):

    max_validation_accuracy = 0
    best_learning_rate = 0

    for learning_rate in learning_rate_values:
        validation_accuracy = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, fixed_weight_decay, fixed_dropout)
        if validation_accuracy >  max_validation_accuracy:
            best_learning_rate = learning_rate

    return best_learning_rate


def choose_learning_rate_for_RNN(network, training_dataset, validation_dataset, fixed_weight_decay, fixed_dropout):

    max_validation_accuracy = 0
    best_learning_rate = 0

    for learning_rate in learning_rate_values_for_RNN:
        validation_accuracy = network.train_and_evaluate(
            training_dataset, validation_dataset, learning_rate, fixed_weight_decay, fixed_dropout)
        if validation_accuracy >  max_validation_accuracy:
            best_learning_rate = learning_rate

    return best_learning_rate







def MLP_hyperparameters_tuning():
    keep_probability = 0.5
    epsilon = 1e-3
    learning_rate = 0.05
    weight_decay = 0.01

    epigeneticsData = CancerData(5)

    training_dataset, validation_dataset, test_dataset = epigeneticsData.get_training_validation_test_datasets()

    input_data_size = epigeneticsData.input_data_size
    output_size = epigeneticsData.output_size

    ffnn = FeedforwardNeuralNetwork(input_data_size, [64, 128, 32, 16], output_size)

    training_accuracy, validation_accuracy, steps_list, test_accuracy = ffnn.train_and_validate(
        training_dataset, validation_dataset, test_dataset,
        learning_rate, weight_decay, keep_probability)

    plot_training_accuracy_and_validation_accuracy(training_accuracy, validation_accuracy, steps_list)

def RNN_hyperparameters_tunning():

    MLP_keep_probability = 0.5
    LSTMs_keep_probability = 0.5

    # Training parameters
    RNN_learning_rate = 0.0001
    RNN_weight_decay = 0.005
    #epigeneticsData = EmbryoDevelopmentData(1, 256, 6.25)

    epigeneticsData = CancerData(5)

    training_dataset, validation_dataset, test_dataset = epigeneticsData.get_training_validation_test_datasets()

    input_data_size = epigeneticsData.input_data_size
    output_size = epigeneticsData.output_size


    #rnn = RecurrentNeuralNetwork(16, 16, [64, 128], [128, 64], output_size)

    print input_data_size
    rnn = RecurrentNeuralNetwork(26, 2, [32, 128], [64, 16], output_size)


    training_accuracy, validation_accuracy, steps_list, test_accuracy = rnn.train_validate_test(
        training_dataset, validation_dataset, test_dataset,
        RNN_learning_rate, RNN_weight_decay, LSTMs_keep_probability, MLP_keep_probability)

    plot_training_accuracy_and_validation_accuracy(training_accuracy, validation_accuracy, steps_list)


def superlayeredNN_hyperparameters_tuning():

    #epigeneticsData = EmbryoDevelopmentDataWithClusters(2, 3, 256, 6)

    epigeneticsData = CancerDataWithClusters(5)
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

    print "got her"
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


superlayeredNN_hyperparameters_tuning()

MLP_hyperparameters_tuning()

RNN_hyperparameters_tunning()

