import math
import numpy as np
import tensorflow as tf

# Hyperparameters

# first superlayer
s1_hidden_units_1 = 256
s1_hidden_units_2 = 128
s1_hidden_units_3 = 64
s1_hidden_units_4 = 32

# second superlayer
s2_hidden_units_1 = 256
s2_hidden_units_2 = 128
s2_hidden_units_3 = 64
s2_hidden_units_4 = 32

merge_layer_size = 16

keep_probability = 0.5
epsilon = 1e-3

# Training parameters
learning_rate = 0.05
weight_decay = 0.01
batch_size = 16


def initialize_weights_and_biases_for_one_superlayer(input_data_size, hidden_units, merge_layer_size):
    """
    Initialize the weights for the neural network using He initialization and initialize the biases to zero
    :param input_data_size: number of gene used in the input layer
    :param hidden_units: array containing the number of units for each hidden layer
    :param merge_layer_size: number of classes in the output layer
    :return: weights dictionary
    :return: biases dictionary
    """

    weights = dict()
    biases = dict()

    hidden_units_1 = hidden_units[0]
    hidden_units_2 = hidden_units[1]
    hidden_units_3 = hidden_units[2]
    hidden_units_4 = hidden_units[3]

    # weights for the input layer
    weights_input_layer = tf.Variable(
        tf.truncated_normal([input_data_size, hidden_units_1],
                            stddev=math.sqrt(2.0 / float(input_data_size))))
    weights['weights_input_layer'] = weights_input_layer

    # biases for the first hidden layer
    biases_first_hidden_layer = tf.Variable(tf.zeros(hidden_units_1))
    biases['biases_first_hidden_layer'] = biases_first_hidden_layer

    # weights for first hidden layer
    weights_first_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_1, hidden_units_2],
                            stddev=math.sqrt(2.0 / float(hidden_units_1))))
    weights['weights_first_hidden_layer'] = weights_first_hidden_layer

    # biases for second hidden layer
    biases_second_hidden_layer = tf.Variable(tf.zeros(hidden_units_2))
    biases['biases_second_hidden_layer'] = biases_second_hidden_layer

    # weights for second hidden layer
    weights_second_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_2, hidden_units_3],
                            stddev=math.sqrt(2.0 / float(hidden_units_2))))
    weights['weights_second_hidden_layer'] = weights_second_hidden_layer

    # biases for third hidden layer
    biases_third_hidden_layer = tf.Variable(tf.zeros(hidden_units_3))
    biases['biases_third_hidden_layer'] = biases_third_hidden_layer

    # weights for third hidden layer
    weights_third_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_3, hidden_units_4],
                            stddev=math.sqrt(2.0 / float(hidden_units_3))))
    weights['weights_third_hidden_layer'] = weights_third_hidden_layer

    # biases for forth layer
    biases_forth_hidden_layer = tf.Variable(tf.zeros(hidden_units_4))
    biases['biases_forth_hidden_layer'] = biases_forth_hidden_layer

    # weights for forth layer
    weights_forth_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_4, merge_layer_size], stddev=math.sqrt(2.0 / float(hidden_units_4))))
    weights['weights_forth_hidden_layer'] = weights_forth_hidden_layer

    # biases for output layer
    biases_merge_layer = tf.Variable(tf.zeros(merge_layer_size))
    biases['biases_merge_layer'] = biases_merge_layer

    return weights, biases


def initialize_weights_and_biases_for_superlayered_network (
        clusters_size, s1_hidden_units, s2_hidden_units, merge_layer_size, output_size):

    weights = dict()
    biases = dict()

    weights['s1'], biases['s1'] = initialize_weights_and_biases_for_one_superlayer(
        clusters_size[0], s1_hidden_units, merge_layer_size)

    weights['s2'], biases['s2'] = initialize_weights_and_biases_for_one_superlayer(
        clusters_size[1], s2_hidden_units, merge_layer_size)

    # initialize weights for cross connections
    weights['cross_connection_from_s1'] = tf.Variable(
        tf.truncated_normal([s1_hidden_units[1], s2_hidden_units[2]],
                            stddev=math.sqrt(2.0 / float(s1_hidden_units[1]))))

    weights['cross_connection_from_s2'] = tf.Variable(
        tf.truncated_normal([s2_hidden_units[1], s1_hidden_units[2]],
                            stddev=math.sqrt(2.0 / float(s2_hidden_units[1]))))

    # biases for merge layer
    biases['merge_layer'] = tf.Variable(tf.zeros(merge_layer_size))

    # weights for merge layer
    weights['merge_layer'] = tf.Variable(
        tf.truncated_normal([merge_layer_size, output_size], stddev=math.sqrt(2.0 / float(merge_layer_size))))

    # biases for output layer
    biases_output_layer = tf.Variable(tf.zeros(output_size))
    biases['output_layer'] = biases_output_layer

    return weights, biases


def inference(s1_input_data, s2_input_data, weights, biases, keep_probability):

    s1_weights = weights['s1']
    s1_biases = biases['s1']

    s2_weights = weights['s2']
    s2_biases = biases['s2']

    """ First hidden layer """

    s1_input_to_first_hidden_layer = \
        tf.matmul(s1_input_data, s1_weights['weights_input_layer']) + \
        s1_biases['biases_first_hidden_layer']
    mean, variance = tf.nn.moments(s1_input_to_first_hidden_layer, [0])

    s1_first_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s1_input_to_first_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    s2_input_to_first_hidden_layer = \
        tf.matmul(s2_input_data, s2_weights['weights_input_layer']) + \
        s2_biases['biases_first_hidden_layer']
    mean, variance = tf.nn.moments(s2_input_to_first_hidden_layer, [0])

    s2_first_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s2_input_to_first_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    """ Second hidden layer """

    s1_input_to_second_hidden_layer = \
        tf.matmul(s1_first_hidden_layer, s1_weights['weights_first_hidden_layer']) + \
        s1_biases['biases_second_hidden_layer']
    mean, variance = tf.nn.moments(s1_input_to_second_hidden_layer, [0])

    s1_second_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s1_input_to_second_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    s2_input_to_second_hidden_layer = \
        tf.matmul(s2_first_hidden_layer, s2_weights['weights_first_hidden_layer']) + s2_biases[
            'biases_second_hidden_layer']
    mean, variance = tf.nn.moments(s2_input_to_second_hidden_layer, [0])

    s2_second_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s2_input_to_second_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    """ Third hidden layer that contains cross-connection between the superlayers """

    s1_input_to_third_hidden_layer = \
        tf.matmul(s1_second_hidden_layer, s1_weights['weights_second_hidden_layer']) + \
        tf.matmul(s2_second_hidden_layer, weights['cross_connection_from_s2']) + \
        s1_biases['biases_third_hidden_layer']
    mean, variance = tf.nn.moments(s1_input_to_third_hidden_layer, [0])

    s1_third_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s1_input_to_third_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    s2_input_to_third_hidden_layer = \
        tf.matmul(s2_second_hidden_layer, s2_weights['weights_second_hidden_layer']) + \
        tf.matmul(s2_second_hidden_layer, weights['cross_connection_from_s1']) + \
        s2_biases['biases_third_hidden_layer']
    mean, variance = tf.nn.moments(s2_input_to_third_hidden_layer, [0])

    s2_third_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s2_input_to_third_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    """ Forth_hidden_layer """

    s1_input_to_forth_hidden_layer = \
        tf.matmul(s1_third_hidden_layer, s1_weights['weights_third_hidden_layer']) + \
        s1_biases['biases_forth_hidden_layer']
    mean, variance = tf.nn.moments(s1_input_to_forth_hidden_layer, [0])

    s1_forth_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s1_input_to_forth_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    s2_input_to_forth_hidden_layer = \
        tf.matmul(s2_third_hidden_layer, s2_weights['weights_third_hidden_layer']) + s2_biases[
            'biases_forth_hidden_layer']
    mean, variance = tf.nn.moments(s2_input_to_forth_hidden_layer, [0])

    s2_forth_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(s2_input_to_forth_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    """ Merge layer """

    input_to_merge_layer = \
        tf.matmul(s1_forth_hidden_layer, s1_weights['weights_forth_hidden_layer']) + \
        tf.matmul(s2_forth_hidden_layer, s2_weights['weights_forth_hidden_layer']) + \
        biases['merge_layer']
    mean, variance = tf.nn.moments(input_to_merge_layer, [0])

    merge_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(input_to_merge_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    # output layer
    logits = tf.matmul(merge_layer, weights['merge_layer']) + biases['output_layer']

    return logits


def create_feed_dictionary(
        s1_placeholder_data, s2_placeholder_data, placeholder_labels, placeholder_keep_probability,
        s1_input_data, s2_input_data, labels, keep_probability):

    feed_dictionary = {
        s1_placeholder_data: s1_input_data,
        s2_placeholder_data: s2_input_data,
        placeholder_labels: labels,
        placeholder_keep_probability: keep_probability
    }
    return feed_dictionary


def compute_loss(logits, labels, weights):
    """
    :param logits:
    :param labels:
    :return:
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    L2_loss = tf.nn.l2_loss(weights['s1']['weights_input_layer']) + \
              tf.nn.l2_loss(weights['s1']['weights_first_hidden_layer']) + \
              tf.nn.l2_loss(weights['s1']['weights_second_hidden_layer']) + \
              tf.nn.l2_loss(weights['s1']['weights_third_hidden_layer']) + \
              tf.nn.l2_loss(weights['s1']['weights_forth_hidden_layer']) + \
              tf.nn.l2_loss(weights['s2']['weights_input_layer']) + \
              tf.nn.l2_loss(weights['s2']['weights_first_hidden_layer']) + \
              tf.nn.l2_loss(weights['s2']['weights_second_hidden_layer']) + \
              tf.nn.l2_loss(weights['s2']['weights_third_hidden_layer']) + \
              tf.nn.l2_loss(weights['s2']['weights_forth_hidden_layer'])

    loss = tf.reduce_mean(cross_entropy + L2_loss * weight_decay)

    return loss


def compute_predictions_accuracy(predictions, labels):
    """
    :param predictions: labels given by the feedforward neural network
    :param labels: correct labels for the input date
    :return: percentage of predictions that match the correct labels
    """
    num_correct_labels = 0
    for index in range(predictions.shape[0]):
        if np.argmax(predictions[index]) == np.argmax(labels[index]):
            num_correct_labels += 1

    return (100 * num_correct_labels)/predictions.shape[0]


def train_superlayered_neural_network(training_dataset, validation_dataset, clusters_size, output_size):
    """
    Train the feed forward neural network using gradient descent by trying to minimize the loss.
    This function is used for cross validation.

    :param training_dataset: dictionary containing the training data and training labels
    :param validation_dataset: dictionary containing the validation data and validation labels
    :param input_data_size: the size of the input to the neural network
    :param output_size: the number of labels in the output
    :return: the validation accuracy of the model
    """

    s1_training_data = training_dataset['training_data'][0]
    s2_training_data = training_dataset['training_data'][1]
    training_labels = training_dataset["training_labels"]

    s1_validation_data = validation_dataset['validation_data'][0]
    s2_validation_data = validation_dataset['validation_data'][1]
    validation_labels = validation_dataset["validation_labels"]


    graph = tf.Graph()
    with graph.as_default():

        # create placeholders for input tensors
        tf_s1_input_data = tf.placeholder(tf.float32, shape=(None, clusters_size[0]))
        tf_s2_input_data = tf.placeholder(tf.float32, shape=(None, clusters_size[1]))

        tf_output_labels = tf.placeholder(tf.float32, shape=(None, output_size))

        # create placeholder for the keep probability
        # dropout is used during training, but not during testing
        tf_keep_probability = tf.placeholder(tf.float32)

        s1_hidden_units = [s1_hidden_units_1, s1_hidden_units_2, s1_hidden_units_3, s1_hidden_units_4]
        s2_hidden_units = [s2_hidden_units_1, s2_hidden_units_2, s2_hidden_units_3, s2_hidden_units_4]

        weights, biases = initialize_weights_and_biases_for_superlayered_network(
            clusters_size, s1_hidden_units, s1_hidden_units, merge_layer_size, output_size)

        logits = inference(tf_s1_input_data, tf_s2_input_data, weights, biases, tf_keep_probability)
        training_loss = compute_loss(logits, tf_output_labels, weights)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)

        training_predictions = tf.nn.softmax(logits)
        validation_predictions = tf.nn.softmax(inference(
            s1_validation_data, s2_validation_data, weights, biases, tf_keep_probability))

    steps = 8000
    with tf.Session(graph=graph) as session:

        # initialize weights and biases
        tf.initialize_all_variables().run()

        for step in range(steps):

            offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

            """ Create a training minibatch for each superlayer. """
            s1_minibatch_data = s1_training_data[offset:(offset + batch_size), :]
            s2_minibatch_data = s2_training_data[offset:(offset + batch_size), :]

            minibatch_labels = training_labels[offset:(offset + batch_size), :]

            feed_dictionary = create_feed_dictionary(
                tf_s1_input_data, tf_s2_input_data, tf_output_labels, tf_keep_probability,
                s1_minibatch_data, s2_minibatch_data, minibatch_labels, keep_probability)

            _, loss, predictions = session.run(
                [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, loss))
                print('Minibatch accuracy: %.1f%%' % compute_predictions_accuracy(predictions, minibatch_labels))

        validation_feed_dictionary = create_feed_dictionary(
            tf_s1_input_data, tf_s2_input_data, tf_output_labels, tf_keep_probability,
            s1_validation_data, s2_validation_data, validation_labels, 1.0)
        validation_accuracy = compute_predictions_accuracy(
                  validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

        print('Validation accuracy: %.1f%%' % validation_accuracy)
        return validation_accuracy

