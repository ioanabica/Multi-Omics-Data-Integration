import numpy
from epigenetics_data_processing import EpigeneticsData
#from synthetic_data import SyntheticData

from feedforward_neural_network import FeedforwardNeuralNetwork
from LSTM_recurrent_neural_network import RecurrentNeuralNetwork



keep_probability = 0.5
epsilon = 1e-3

# Training parameters
learning_rate = 0.05
weight_decay = 0.01

epigeneticsData = EpigeneticsData()
k_fold_datasets = epigeneticsData.k_fold_datasets
input_data_size = epigeneticsData.input_data_size
output_size = epigeneticsData.output_size

keys = k_fold_datasets.keys()
validation_accuracy = []

ffnn = FeedforwardNeuralNetwork(input_data_size, [256, 128, 64, 32], output_size)
rnn = RecurrentNeuralNetwork(input_data_size/16, 16, [64, 128, 256], [512, 256, 128, 32], output_size)


for key in keys:
    print "key number" + str(key)
    training_dataset = k_fold_datasets[key]["training_dataset"]
    print len(training_dataset["training_data"])

    validation_dataset = k_fold_datasets[key]["validation_dataset"]
    print len(validation_dataset["validation_data"])

    accuracy = rnn.train_and_validate(
        training_dataset, validation_dataset)
    validation_accuracy += accuracy


print validation_accuracy
print numpy.mean(validation_accuracy)



"""
epigeneticsData = EpigeneticsData()
k_fold_datasets_with_clusters = epigeneticsData.k_fold_datasets_with_clusters
clusters_size = epigeneticsData.clusters_size
output_size = epigeneticsData.output_size

keys = k_fold_datasets_with_clusters.keys()
validation_accuracy = []

for key in keys:
    training_dataset = k_fold_datasets_with_clusters[key]["training_dataset"]
    print len(training_dataset["training_data"][0])
    validation_dataset = k_fold_datasets_with_clusters[key]["validation_dataset"]
    print len(validation_dataset["validation_data"][0])
    accuracy = train_superlayered_neural_network(training_dataset, validation_dataset, clusters_size,
                                                            output_size)
    validation_accuracy += [accuracy]
    print "key number" + str(key)


print validation_accuracy
print numpy.mean(validation_accuracy)
"""

