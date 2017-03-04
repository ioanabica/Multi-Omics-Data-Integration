#from embryo_development_data import EmbryoDevelopmentData, EmbryoDevelopmentDataWithClusters
import matplotlib.pyplot as plt

from epygenetic_data.cancer_data.cancer_data import CancerData, CancerDataWithClusters
from neural_network_models.LSTM_recurrent_neural_network import RecurrentNeuralNetwork
from neural_network_models.feedforward_neural_network import FeedforwardNeuralNetwork
from neural_network_models.superlayered_neural_network import SuperlayeredNeuralNetwork


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

    training_accuracy, validation_accuracy, steps_list, test_accuracy = ffnn.train_validate_test(
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

