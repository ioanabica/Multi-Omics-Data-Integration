import numpy
from epigenetics_data_processing import EpigeneticsData
#from synthetic_data import SyntheticData
from feedforward_neural_network import train_feedforward_neural_network

#from recurrent_neural_network import train_recurrent_neural_network

#from recurrent_neural_network_using_BasicLSTMcell import train_recurrent_neural_network

from rnn_three_LSTM_units import train_recurrent_neural_network

epigeneticsData = EpigeneticsData()
k_fold_datasets = epigeneticsData.k_fold_datasets
input_data_size = epigeneticsData.input_data_size
output_size = epigeneticsData.output_size

keys = k_fold_datasets.keys()
validation_accuracy = []


for key in keys:
    training_dataset = k_fold_datasets[key]["training_dataset"]
    print len(training_dataset["training_data"])
    validation_dataset = k_fold_datasets[key]["validation_dataset"]
    print len(validation_dataset["validation_data"])
    accuracy = train_recurrent_neural_network(training_dataset, validation_dataset, input_data_size,
                                                            output_size)
    validation_accuracy += [accuracy]
    print "key number" + str(key)


print validation_accuracy
print numpy.mean(validation_accuracy)


training_dataset = EpigeneticsData.training_dataset
print len(training_dataset["training_data"])
validation_dataset = EpigeneticsData.validation_dataset
print len(validation_dataset["validation_data"])


print ("training")

validation_accuracy = train_feedforward_neural_network(training_dataset, validation_dataset, input_data_size,
                                                            output_size)



