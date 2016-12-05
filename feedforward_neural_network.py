import math
import numpy as np
import tensorflow as tf

# Hyperparameters
hidden_units_1 = 256
hidden_units_2 = 128
hidden_units_3 = 64
hidden_units_4 = 32
keep_probability = 0.5
epsilon = 1e-3

# Training parameters
learning_rate = 0.05
batch_size = 16


def initialize_weights_and_biases(
        input_data_size, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, output_size):
    """
    Initialize the weights for the neural network using He initialization and initialize the biases to zero
    :param input_data_size: number of gene used in the input layer
    :param hidden_units_1: number of hidden neurons in the first layer
    :param hidden_units_2: number of hidden neurons in the second layer
    :param hidden_units_3: number of hidden neurons in the third layer
    :param hidden_units_4: number of hidden neurons in the forth layer
    :param output_size: number of classes in the output layer
    :return: weights dictionary
    :return: biases dictionary
    """

    weights = dict()
    biases = dict()

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

    #biases for second hidden layer
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
        tf.truncated_normal([hidden_units_4, output_size], stddev=math.sqrt(2.0/float(hidden_units_4))))
    weights['weights_forth_hidden_layer'] = weights_forth_hidden_layer

    # biases for output layer
    biases_output_layer = tf.Variable(tf.zeros(output_size))
    biases['biases_output_layer'] = biases_output_layer

    return weights, biases


def inference(input_data, weights, biases, keep_probability):
    """
    :param input_data:  input to the feedforward neural network for which the model is run
    :param weights: the weights for the layers of the neural network
    :param biases: the biases for the layers of the neural network
    :param keep_probability: the probability for a neuron in the hidden layers to be used during dropout
    :return: logits: the output of the feed forward neural network
    """

    # first hidden layer
    input_to_first_hidden_layer = \
        tf.matmul(input_data, weights['weights_input_layer']) + biases['biases_first_hidden_layer']
    mean, variance = tf.nn.moments(input_to_first_hidden_layer, [0])

    first_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(input_to_first_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    # second hidden layer
    input_to_second_hidden_layer = \
        tf.matmul(first_hidden_layer, weights['weights_first_hidden_layer']) + biases['biases_second_hidden_layer']
    mean, variance = tf.nn.moments(input_to_second_hidden_layer, [0])

    second_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(input_to_second_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    # third hidden layer
    input_to_third_hidden_layer = \
        tf.matmul(second_hidden_layer, weights['weights_second_hidden_layer']) + biases['biases_third_hidden_layer']
    mean, variance = tf.nn.moments(input_to_third_hidden_layer, [0])
    third_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(input_to_third_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    # forth_hidden_layer
    input_to_forth_hidden_layer = \
        tf.matmul(third_hidden_layer, weights['weights_third_hidden_layer']) + biases['biases_forth_hidden_layer']
    mean, variance = tf.nn.moments(input_to_forth_hidden_layer, [0])

    forth_hidden_layer = tf.nn.dropout(tf.nn.relu(
        tf.nn.batch_normalization(input_to_forth_hidden_layer, mean, variance, None, None, epsilon)),
        keep_probability)

    # output layer
    logits = tf.matmul(forth_hidden_layer, weights['weights_forth_hidden_layer']) + biases['biases_output_layer']

    return logits


def create_feed_dictionary(
        placeholder_data, placeholder_labels, placeholder_keep_probability, data, labels, keep_probability):
    """
    :param placeholder_data:
    :param placeholder_labels:
    :param placeholder_keep_probability:
    :param data:
    :param labels:
    :param keep_probability:
    :return: a dictionary form the placeholder data to the real data
    """
    feed_dictionary = {
        placeholder_data: data,
        placeholder_labels: labels,
        placeholder_keep_probability: keep_probability
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
    :param labels: correct labels for the input date
    :return: percentage of predictions that match the correct labels
    """
    num_correct_labels = 0
    for index in range(predictions.shape[0]):
        if np.argmax(predictions[index]) == np.argmax(labels[index]):
            num_correct_labels += 1

    return (100 * num_correct_labels)/predictions.shape[0]


def train_feed_forward_neural_network(training_dataset, validation_dataset, input_data_size, output_size):
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

    validation_data = validation_dataset["validation_data"]
    validation_labels = validation_dataset["validation_labels"]

    graph = tf.Graph()
    with graph.as_default():

        # create placeholders for input tensors
        tf_input_data = tf.placeholder(tf.float32, shape=(None, input_data_size))
        tf_input_labels = tf.placeholder(tf.float32, shape=(None, output_size))

        # create placeholder for the keep probability
        # dropout is used during training, but not during testing
        tf_keep_probability = tf.placeholder(tf.float32)

        weights, biases = initialize_weights_and_biases(
            input_data_size,
            hidden_units_1,
            hidden_units_2,
            hidden_units_3,
            hidden_units_4,
            output_size)

        logits = inference(tf_input_data, weights, biases, tf_keep_probability)
        training_loss = compute_loss(logits, tf_input_labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)

        training_predictions = tf.nn.softmax(logits)
        validation_predictions = tf.nn.softmax(inference(validation_data, weights, biases, tf_keep_probability))

    steps = 3000
    with tf.Session(graph=graph) as session:

        # initialize weights and biases
        tf.initialize_all_variables().run()

        for step in range(steps):

            offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

            # Create a training minibatch.
            minibatch_data = training_data[offset:(offset + batch_size), :]
            minibatch_labels = training_labels[offset:(offset + batch_size), :]

            feed_dictionary = create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_keep_probability,
                minibatch_data, minibatch_labels, keep_probability)

            _, loss, predictions = session.run(
                [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, loss))
                print('Minibatch accuracy: %.1f%%' % compute_predictions_accuracy(predictions, minibatch_labels))

        validation_feed_dictionary = create_feed_dictionary(
            tf_input_data, tf_input_labels, tf_keep_probability,
            validation_data, validation_labels, 1.0)
        validation_accuracy = compute_predictions_accuracy(
                  validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

        print('Validation accuracy: %.1f%%' % validation_accuracy)
        return validation_accuracy

