import math
import numpy as np
import tensorflow as tf
from data_processing import EpigeneticsData

# constants
hidden_units_1 = 64
hidden_units_2 = 32
learning_rate = 0.5
batch_size = 16

def inference(input_data, input_data_size, label_size, hidden_units_1, hidden_units_2):
    # first hidden layer
    weights_1 = tf.Variable(
        tf.truncated_normal([input_data_size, hidden_units_1],
                            stddev=1.0 / math.sqrt(float(input_data_size))))
    biases_1 = tf.Variable(tf.zeros(hidden_units_1))
    hidden_layer_1 = tf.nn.relu(tf.matmul(input_data, weights_1) + biases_1)

    # second hidden layer
    weights_2 = tf.Variable(
        tf.truncated_normal([hidden_units_1, hidden_units_2],
                            stddev=1.0 / math.sqrt(float(hidden_units_1))))
    biases_2 = tf.Variable(tf.zeros(hidden_units_2))
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights_2) + biases_2)

    # output layer
    weights_output = tf.Variable(
        tf.truncated_normal([hidden_units_2, label_size],
                            stddev=1.0 / math.sqrt(float(hidden_units_2))))
    biases_output = tf.Variable(tf.zeros(label_size))
    logits = tf.matmul(hidden_layer_2, weights_output) + biases_output

    return logits


def create_feed_dictionary(placeholder_data, placeholder_labels, data, labels):
    # create a dictionary form the placeholder data to the real data
    feed_dictionary = {
        placeholder_data: data,
        placeholder_labels: labels
    }

    return feed_dictionary


def compute_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)

    return loss


def compute_prediction_accuracy(logits, labels):
    num_correct_labels = 0
    for index in range(batch_size):
        if np.argmax(logits[index]) == np.argmax(labels[index]):
            num_correct_labels += 1

    return num_correct_labels


class FeedForwardNeuralNetwork(object):
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
        tf_training_data = tf.placeholder(tf.float32, shape=(batch_size, epigeneticsData.input_data_size))
        tf_training_labels = tf.placeholder(tf.float32, shape=(batch_size, epigeneticsData.label_size))

        print tf_training_data
        print tf_training_labels

        tf_validation_data = tf.constant(validation_data)
        tf_validation_labels = tf.constant(validation_labels)
        tf_test_data = tf.constant(test_data)
        tf_test_labels = tf.constant(test_labels)

        logits = inference(
            tf_training_data,
            epigeneticsData.input_data_size,
            epigeneticsData.label_size,
            hidden_units_1,
            hidden_units_2)
        training_loss = compute_loss(logits, tf_training_labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)
        prediction_accuracy = compute_prediction_accuracy(logits, tf_training_labels)
        print prediction_accuracy

        training_prediction = tf.nn.softmax(logits)

    steps = 3001
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(steps):
            offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

            # Create a training minibatch.
            minibatch_data = training_data[offset:(offset + batch_size), :]
            minibatch_labels = training_labels[offset:(offset + batch_size), :]

            feed_dictionary = create_feed_dictionary(
                tf_training_data, tf_training_labels, minibatch_data, minibatch_labels)

            _, loss, predictions = session.run(
                [optimizer, training_loss, training_prediction], feed_dict=feed_dictionary)

            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, loss))
