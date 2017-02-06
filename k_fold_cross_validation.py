import numpy
from embryo_development_data import EmbryoDevelopmentData
#from synthetic_data import SyntheticData

from feedforward_neural_network import FeedforwardNeuralNetwork
from LSTM_recurrent_neural_network import RecurrentNeuralNetwork
from superlayered_neural_network import SuperlayeredNeuralNetwork


keep_probability = 0.5
epsilon = 1e-3

# Training parameters
learning_rate = 0.05
weight_decay = 0.01


MLP_keep_probability = 0.75
LSTMs_keep_probability = 0.75

# Training parameters
RNN_learning_rate = 0.0001
RNN_weight_decay = 0.001

epigeneticsData = EmbryoDevelopmentData(7, 128, 6.3)

k_fold_datasets = epigeneticsData.get_k_fold_datasets()

input_data_size = epigeneticsData.input_data_size
output_size = epigeneticsData.output_size

#k_fold_datasets_with_clusters = epigeneticsData.k_fold_datasets_with_clusters

keys = k_fold_datasets.keys()
validation_accuracy = list()

ffnn = FeedforwardNeuralNetwork(input_data_size, [256, 128, 64, 32], output_size)
#rnn = RecurrentNeuralNetwork(input_data_size/8, 8, [128, 256, 512], [512, 256, 128, 32], output_size)
#rnn = RecurrentNeuralNetwork(input_data_size/8, 8, [64, 128, 256], [512, 256, 128, 32], output_size)
#rnn = RecurrentNeuralNetwork(input_data_size/8, 8, [16, 32, 64, 128], [256, 128, 64, 32], output_size)

rnn = RecurrentNeuralNetwork(16, 8, [64, 256], [128, 64], output_size)

for key in keys:
    print "key number" + str(key)
    training_dataset = k_fold_datasets[key]["training_dataset"]

    print len(training_dataset["training_data"])
    print len(training_dataset["training_data"][0])

    validation_dataset = k_fold_datasets[key]["validation_dataset"]
    print len(validation_dataset["validation_data"])

    accuracy = rnn.train_and_validate(
        training_dataset, validation_dataset,
        RNN_learning_rate, RNN_weight_decay, LSTMs_keep_probability, MLP_keep_probability)
    validation_accuracy.append(accuracy)

print validation_accuracy
print numpy.mean(validation_accuracy)

"""


for key in keys:
    print "key number" + str(key)
    training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]

    new_training_dataset = dict()
    new_training_dataset["training_data"] = training_dataset["training_data"][0]
    new_training_dataset["training_labels"] = training_dataset["training_labels"]
    print len(new_training_dataset["training_data"])
    print len(new_training_dataset["training_data"][0])

    validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]
    new_validation_dataset = dict()
    new_validation_dataset["validation_data"] = validation_dataset["validation_data"][0]
    new_validation_dataset["validation_labels"] = validation_dataset["validation_labels"]

    print len(new_validation_dataset["validation_data"])
    print len(new_validation_dataset["validation_data"][0])

    accuracy = rnn.train_and_validate(
        new_training_dataset, new_validation_dataset)
    validation_accuracy.append(accuracy)

print validation_accuracy
print numpy.mean(validation_accuracy)


"""
"""

epigeneticsData = EpigeneticsData()
k_fold_datasets_with_clusters = epigeneticsData.k_fold_datasets_with_clusters
clusters_size = epigeneticsData.clusters_size
output_size = epigeneticsData.output_size

keys = k_fold_datasets_with_clusters.keys()
validation_accuracy = []

superlayered_nn = SuperlayeredNeuralNetwork(
        [clusters_size[0], clusters_size[1]],
        [[512, 256, 128, 64], [512, 256, 128, 64]],
        [128, 32],
        output_size)


for key in keys:
    training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
    print len(training_dataset["training_data"][0])
    validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]
    print len(validation_dataset["validation_data"][0])
    accuracy = superlayered_nn.train_and_validate(training_dataset, validation_dataset)
    validation_accuracy += [accuracy]
    print "key number" + str(key)


print validation_accuracy
print numpy.mean(validation_accuracy)

"""
