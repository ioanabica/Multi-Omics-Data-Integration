import math
import numpy as np
import tensorflow as tf
from epigenetics_data_processing import EpigeneticsData

# constants
hidden_units_1 = 256
hidden_units_2 = 128
hidden_units_3 = 64
hidden_units_4 = 32
learning_rate = 0.5
epsilon = 1e-3
batch_size = 16
dropout_probability = 0.5


def initialize_weights_and_biases(input_data_size, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, label_size):
    """
    Initialize the weights for the neural network using He initialization and intialize the biases to zero
    :param input_data_size: number of gene used in the input layer
    :param hidden_units_1: number of hidden neurons in the first layer
    :param hidden_units_2: number of hidden neurons in the second layer
    :param hidden_units_3: number of hidden neurons in the third layer
    :param hidden_units_4: number of hidden neurons in the forth layer
    :param label_size: number of classes in the output layer
    :return: weights_input_layer
    """

    # weights for the input layer
    weights_input_layer = tf.Variable(
        tf.truncated_normal([input_data_size, hidden_units_1],
                            stddev=math.sqrt(2.0 / float(input_data_size))))

    # biases for the first hidden layer
    biases_first_hidden_layer = tf.Variable(tf.zeros(hidden_units_1))

    # weights for first hidden layer
    weights_first_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_1, hidden_units_2],
                            stddev=math.sqrt(2.0 / float(hidden_units_1))))

    #biases for second hidden layer
    biases_second_hidden_layer = tf.Variable(tf.zeros(hidden_units_2))

    # weights for second hidden layer
    weights_second_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_2, hidden_units_3],
                            stddev=math.sqrt(2.0 / float(hidden_units_2))))

    # biases for third hidden layer
    biases_third_hidden_layer = tf.Variable(tf.zeros(hidden_units_3))

    # weights for third hidden layer
    weights_third_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_3, hidden_units_4],
                            stddev=math.sqrt(2.0 / float(hidden_units_3))))
    # biases for forth layer
    biases_forth_hidden_layer = tf.Variable(tf.zeros(hidden_units_4))

    #weights for forth layer
    weights_forth_hidden_layer = tf.Variable(
        tf.truncated_normal([hidden_units_4, label_size], stddev=math.sqrt(2.0/float(hidden_units_4))))

    # biases for output layer
    biases_output_layer = tf.Variable(tf.zeros(label_size))

    return weights_input_layer, biases_first_hidden_layer, \
           weights_first_hidden_layer, biases_second_hidden_layer, \
           weights_second_hidden_layer, biases_third_hidden_layer, \
           weights_third_hidden_layer, biases_forth_hidden_layer, \
           weights_forth_hidden_layer, biases_output_layer


def create_feed_dictionary(
        placeholder_data, placeholder_labels, data, labels, placeholder_dropout_probability, dropout_probability):
    # create a dictionary form the placeholder data to the real data
    feed_dictionary = {
        placeholder_data: data,
        placeholder_labels: labels,
        placeholder_dropout_probability: dropout_probability
    }

    return feed_dictionary


def compute_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)

    return loss


def compute_prediction_accuracy(predictions, labels):
    num_correct_labels = 0
    for index in range(predictions.shape[0]):
        if np.argmax(predictions[index]) == np.argmax(labels[index]):
            num_correct_labels += 1

    return (100 * num_correct_labels)/predictions.shape[0]


epigeneticsData = EpigeneticsData()

training_data = epigeneticsData.training_data
training_labels = epigeneticsData.training_labels

validation_data = epigeneticsData.validation_data
validation_labels = epigeneticsData.validation_labels

test_data = epigeneticsData.test_data
test_labels = epigeneticsData.test_labels

graph = tf.Graph()
with graph.as_default():

    # create placeholders for input tensors
    # the shape of the placeholders matches the batch_size and the number of gene expressions
    tf_training_data = tf.placeholder(tf.float32, shape=(None, epigeneticsData.input_data_size))
    tf_training_labels = tf.placeholder(tf.float32, shape=(None, epigeneticsData.label_size))
    tf_dropout_probability = tf.placeholder(tf.float32)

    tf_validation_data = tf.constant(validation_data)
    tf_test_data = tf.constant(test_data)

    weights_input_layer, biases_first_hidden_layer, \
    weights_first_hidden_layer, biases_second_hidden_layer, \
    weights_second_hidden_layer, biases_third_hidden_layer, \
    weights_third_hidden_layer, biases_forth_hidden_layer, \
    weights_forth_hidden_layer, biases_output_layer = initialize_weights_and_biases(
        epigeneticsData.input_data_size,
        hidden_units_1,
        hidden_units_2,
        hidden_units_3,
        hidden_units_4,
        epigeneticsData.label_size)

    def inference(input_data):
        # first hidden layer
        input_to_first_hidden_layer = \
            tf.matmul(input_data, weights_input_layer) + biases_first_hidden_layer
        mean, variance = tf.nn.moments(input_to_first_hidden_layer, [0])

        first_hidden_layer = tf.nn.relu(
            tf.nn.batch_normalization(input_to_first_hidden_layer, mean, variance, None, None, epsilon))

        # second hidden layer
        input_to_second_hidden_layer = \
            tf.matmul(first_hidden_layer, weights_first_hidden_layer) + biases_second_hidden_layer
        mean, variance = tf.nn.moments(input_to_second_hidden_layer, [0])

        second_hidden_layer = tf.nn.relu(
            tf.nn.batch_normalization(input_to_second_hidden_layer, mean, variance, None, None, epsilon))

        # third hidden layer
        input_to_third_hidden_layer = \
            tf.matmul(second_hidden_layer, weights_second_hidden_layer) + biases_third_hidden_layer
        mean, variance = tf.nn.moments(input_to_third_hidden_layer, [0])
        third_hidden_layer = tf.nn.relu(
            tf.nn.batch_normalization(input_to_third_hidden_layer, mean, variance, None, None, epsilon))

        # forth_hidden_layer
        input_to_forth_hidden_layer = \
            tf.matmul(third_hidden_layer, weights_third_hidden_layer) + biases_forth_hidden_layer
        mean, variance = tf.nn.moments(input_to_forth_hidden_layer, [0])

        forth_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_forth_hidden_layer, mean, variance, None, None, epsilon)), tf_dropout_probability)

        # output layer
        logits = tf.matmul(forth_hidden_layer, weights_forth_hidden_layer) + biases_output_layer

        return logits

    logits = inference(tf_training_data)
    training_loss = compute_loss(logits, tf_training_labels)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)

    training_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(inference(tf_validation_data))
    test_prediction = tf.nn.softmax(inference(tf_test_data))

steps = 3001
with tf.Session(graph=graph) as session:

    # initialize weights and biases
    tf.initialize_all_variables().run()

    for step in range(steps):

        offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

        # Create a training minibatch.
        minibatch_data = training_data[offset:(offset + batch_size), :]
        minibatch_labels = training_labels[offset:(offset + batch_size), :]

        feed_dictionary = create_feed_dictionary(
            tf_training_data, tf_training_labels,
            minibatch_data, minibatch_labels,
            tf_dropout_probability, dropout_probability)

        _, loss, predictions = session.run(
            [optimizer, training_loss, training_prediction], feed_dict=feed_dictionary)

        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, loss))
            print('Minibatch accuracy: %.1f%%' % compute_prediction_accuracy(predictions, minibatch_labels))

            validation_feed_dictionary = create_feed_dictionary(
                tf_training_data, tf_training_labels,
                validation_data, validation_labels,
                tf_dropout_probability, 1.0)
            print('Validation accuracy: %.1f%%' %
                 compute_prediction_accuracy(
                     validation_prediction.eval(feed_dict=validation_feed_dictionary), validation_labels))

    test_feed_dictionary = create_feed_dictionary(
        tf_training_data, tf_training_labels,
        test_data, test_labels,
        tf_dropout_probability, 1.0)
    print('Test accuracy: %.1f%%' % compute_prediction_accuracy(test_prediction.eval(feed_dict=test_feed_dictionary), test_labels))
