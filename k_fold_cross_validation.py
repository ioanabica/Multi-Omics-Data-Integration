from epigenetics_data_processing import EpigeneticsData
from feedforward_neural_network import train_feed_forward_neural_network

epigeneticsData = EpigeneticsData()
k_fold_datasets = epigeneticsData.k_folds_datasets
input_data_size = epigeneticsData.input_data_size
output_size = epigeneticsData.output_size

keys = k_fold_datasets.keys()

for key in keys:
    training_dataset = k_fold_datasets[key]["training_dataset"]

    validation_dataset = k_fold_datasets[key]["validation_dataset"]

    validation_accuracy = train_feed_forward_neural_network(training_dataset, validation_dataset, input_data_size,
                                                            output_size)
    print "key number" + str(key)
    print validation_accuracy





