import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Hyperparameters
hidden_units = 32

# Training parameters
learning_rate = 0.05
batch_size = 16


def initialize_weights_and_biases(hidden_units, output_size):

    weights = dict()
    biases = dict()

    weights_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units, output_size], stddev=math.sqrt(2.0/float(hidden_units))))
    weights['hidden_layer'] = weights_hidden_layer

    biases_output_layer = tf.Variable(tf.zeros(output_size))
    biases['output_layer'] = biases_output_layer

    return weights, biases


def inference(input_data, sequence_size, weights, biases):
    """
    The recurrent neural network processes the epigenetics data as a sequence of gene expressions.

    :param input_data:
    :param weights:
    :param biases:
    :return:
    """

    # Modify input data shape for the recurrent neural network
    # Current data shape: (batch_size, input_data_size)
    # Current data shape: (batch_size, input_data_size)
    # Required data shape: (input_data_size, batch_size)

    input_data = tf.transpose(input_data, [1, 0, 2])
    input_data = tf.reshape(input_data, [-1, 1])
    input_data = tf.split(0, sequence_size, input_data)

    lstm_cell = rnn_cell.BasicLSTMCell(hidden_units, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, input_data,  dtype=tf.float32)

    # output layer
    logits = tf.matmul(outputs[-1], weights['hidden_layer']) + biases['output_layer']

    return logits


def validation_inference(input_data, sequence_size, weights, biases):
    """
    The recurrent neural network processes the epigenetics data as a sequence of gene expressions.

    :param input_data:
    :param weights:
    :param biases:
    :return:
    """

    # Modify input data shape for the recurrent neural network
    # Current data shape: (batch_size, input_data_size)
    # Current data shape: (batch_size, input_data_size)
    # Required data shape: (input_data_size, batch_size)

    input_data = tf.transpose(input_data, [1, 0, 2])
    input_data = tf.reshape(input_data, [-1, 1])
    input_data = tf.split(0, sequence_size, input_data)

    lstm_cell = rnn_cell.BasicLSTMCell(hidden_units, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, input_data,  dtype=tf.float32)

    # output layer
    logits = tf.matmul(outputs[-1], weights['hidden_layer']) + biases['output_layer']

    return logits




def create_feed_dictionary(
        placeholder_data, placeholder_labels, data, labels):
    """
    :param placeholder_data:
    :param placeholder_labels:
    :param data:
    :param labels:
    :return: a dictionary form the placeholder data to the real data
    """
    feed_dictionary = {
        placeholder_data: data,
        placeholder_labels: labels
    }
    return feed_dictionary


def compute_loss(logits, labels):
    """
    :param logits:
    :param labels:
    :return:
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)

    return loss


def compute_predictions_accuracy(predictions, labels):
    """
    :param predictions: labels given by the feedforward neural network
    :param labels: correct labels for the input data
    :return: percentage of predictions that match the correct labels
    """
    num_correct_labels = 0
    for index in range(predictions.shape[0]):
        if np.argmax(predictions[index]) == np.argmax(labels[index]):
            num_correct_labels += 1

    return (100 * num_correct_labels)/predictions.shape[0]


def train_recurrent_neural_network(training_dataset, validation_dataset, input_data_size, output_size):
    """
    Train the feed forward neural network using gradient descent by trying to minimize the loss.
    This function is used for cross validation.

    :param training_dataset: dictionary containing the training data and training labels
    :param validation_dataset: dictionary containing the validation data and validation labels
    :param input_data_size: the size of the input to the neural network
    :param output_size: the number of labels in the output
    :return: the validation accuracy of the model
    """

    training_data = training_dataset["training_data"]
    training_labels = training_dataset["training_labels"]

    training_data = np.reshape(training_data, (len(training_data), input_data_size, 1))

    validation_data = validation_dataset["validation_data"]
    validation_labels = validation_dataset["validation_labels"]

    validation_data = np.reshape(validation_data, (len(validation_data), input_data_size, 1))

    graph = tf.Graph()
    with graph.as_default():

        # create placeholders for input tensors
        tf_input_data = tf.placeholder(tf.float32, shape=(None, input_data_size, 1))
        tf_input_labels = tf.placeholder(tf.float32, shape=(None, output_size))

        weights, biases = initialize_weights_and_biases(hidden_units, output_size)

        with tf.variable_scope('inference'):
            logits = inference(tf_input_data, input_data_size, weights, biases)
        training_loss = compute_loss(logits, tf_input_labels)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(training_loss)

        training_predictions = tf.nn.softmax(logits)
        with tf.variable_scope('inference') as scope:
            scope.reuse_variables()
            validation_predictions = tf.nn.softmax(validation_inference(validation_data, input_data_size, weights, biases))

    steps = 4000
    with tf.Session(graph=graph) as session:

        # initialize weights and biases
        tf.initialize_all_variables().run()

        for step in range(steps):

            offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

            # Create a training minibatch.
            minibatch_data = training_data[offset:(offset + batch_size), :]

            minibatch_data = minibatch_data.reshape(batch_size, input_data_size, 1)

            minibatch_labels = training_labels[offset:(offset + batch_size), :]

            feed_dictionary = create_feed_dictionary(
                tf_input_data, tf_input_labels,
                minibatch_data, minibatch_labels)

            _, loss, predictions = session.run(
                [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

            if (step % 300 == 0):
                print('Minibatch loss at step %d: %f' % (step, loss))
                print('Minibatch accuracy: %.1f%%' % compute_predictions_accuracy(predictions, minibatch_labels))

        validation_feed_dictionary = create_feed_dictionary(
            tf_input_data, tf_input_labels,
            validation_data, validation_labels)

        validation_accuracy = compute_predictions_accuracy(
                  validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

        print('Validation accuracy: %.1f%%' % validation_accuracy)
        return validation_accuracy

