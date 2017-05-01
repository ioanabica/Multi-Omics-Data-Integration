import math
import numpy as np
import tensorflow as tf

# Hyperparameters
# hidden_units_1 = 256
# hidden_units_2 = 128
# hidden_units_3 = 64
# hidden_units_4 = 32
#keep_probability = 0.5
epsilon = 1e-3

# Training parameters
#learning_rate = 0.05
#weight_decay = 0.01
batch_size = 64


logs_path = '/tmp/tensorboard'


class MultilayerPerceptron(object):

    def __init__(self, input_size, hidden_units, output_size):
        self.input_data_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size

    def train_and_evaluate(
            self, training_dataset, test_dataset, learning_rate, weight_decay, keep_probability):
        """
        Train the feed forward neural network using gradient descent by trying to minimize the loss.

        :param training_dataset: dictionary containing the training data and training labels
        :param test_dataset: dictionary containing the validation data and validation labels
        :param learning_rate:
        :param weight_decay:
        :param keep_probability:
        :return: the validation accuracy of the model
        """

        training_data = training_dataset["training_data"]
        training_labels = training_dataset["training_labels"]

        validation_data = test_dataset["validation_data"]
        validation_labels = test_dataset["validation_labels"]

        graph = tf.Graph()
        with graph.as_default():

            # create placeholders for input tensors
            tf_input_data = tf.placeholder(tf.float32, shape=(None, self.input_data_size), name='TrainingExample')
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, self.output_size), name='OutputLabel')

            # create placeholder for the keep probability
            # dropout is used during training, but not during testing
            tf_keep_probability = tf.placeholder(tf.float32)

            self.weights, self.biases = self.initialize_weights_and_biases()

            logits = self.compute_predictions(tf_input_data, self.weights, self.biases, tf_keep_probability)
            training_loss = self.compute_loss(logits, tf_input_labels, self.weights, weight_decay)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)

            training_predictions = tf.nn.softmax(logits)
            validation_predictions = tf.nn.softmax(self.compute_predictions(validation_data, self.weights, self.biases, tf_keep_probability))


        steps = 5000

        with tf.Session(graph=graph) as session:

            # initialize weights and biases
            init = tf.global_variables_initializer()
            session.run(init)

            for step in range(steps):

                offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

                # Create a training minibatch.
                minibatch_data = training_data[offset:(offset + batch_size), :]
                minibatch_labels = training_labels[offset:(offset + batch_size), :]

                feed_dictionary = self.create_feed_dictionary(
                    tf_input_data, tf_input_labels, tf_keep_probability,
                    minibatch_data, minibatch_labels, keep_probability)

                _, loss, predictions = session.run(
                    [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

                if (step % 500 == 0):
                    print('Minibatch loss at step %d: %f' % (step, loss))
                    print('Minibatch accuracy: %.1f%%' % self.compute_predictions_accuracy(predictions, minibatch_labels))

            #plt.plot(range(steps), losses)
            #plt.show()

            validation_feed_dictionary = self.create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_keep_probability,
                validation_data, validation_labels, 1.0)

            validation_accuracy = self.compute_predictions_accuracy(
                validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

            confussion_matrix = self.compute_confussion_matrix(
                validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

            ROC_points = self.compute_ROC_points(
                validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)
            print('Validation accuracy: %.1f%%' % validation_accuracy)

        return validation_accuracy, confussion_matrix, ROC_points


    def train_and_validate(self, training_dataset, validation_dataset, test_dataset,
                           learning_rate, weight_decay, keep_probability):
        """
        Train the feed forward neural network using gradient descent by trying to minimize the loss.

        :param training_dataset: dictionary containing the training data and training labels
        :param validation_dataset: dictionary containing the validation data and validation labels
        :param learning_rate:
        :param weight_decay:
        :param keep_probability:
        :return: the validation accuracy of the model
        """

        training_data = training_dataset["training_data"]
        training_labels = training_dataset["training_labels"]

        validation_data = validation_dataset["validation_data"]
        validation_labels = validation_dataset["validation_labels"]

        test_data = test_dataset["test_data"]
        test_labels = test_dataset["test_labels"]

        print len(training_data)
        print len(validation_data)
        print len(test_data)

        graph = tf.Graph()
        with graph.as_default():

            # create placeholders for input tensors
            tf_input_data = tf.placeholder(tf.float32, shape=(None, self.input_data_size), name='TrainingExample')
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, self.output_size), name='OutputLabel')

            # create placeholder for the keep probability
            # dropout is used during training, but not during testing
            tf_keep_probability = tf.placeholder(tf.float32)
            tf.scalar_summary('dropout_keep_probability', keep_probability)

            weights, biases = self.initialize_weights_and_biases()

            logits = self.compute_predictions(tf_input_data, weights, biases, tf_keep_probability)
            training_loss = self.compute_loss(logits, tf_input_labels, weights, weight_decay)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss)

            training_predictions = tf.nn.softmax(logits)
            validation_predictions = tf.nn.softmax(
                self.compute_predictions(validation_data, weights, biases, tf_keep_probability))
            test_predictions = tf.nn.softmax(self.compute_predictions(test_data, weights, biases, tf_keep_probability))

            merged_summary = tf.merge_all_summaries()

        steps = 10000
        losses = []

        training_accuracy_list = list()
        validation_accuracy_list = list()
        steps_list = list()

        with tf.Session(graph=graph) as session:

            # initialize weights and biases
            init = tf.global_variables_initializer()
            session.run(init)

            summary_writer = tf.train.SummaryWriter(logs_path, graph)

            for step in range(steps):

                offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

                # Create a training minibatch.
                minibatch_data = training_data[offset:(offset + batch_size), :]
                minibatch_labels = training_labels[offset:(offset + batch_size), :]

                feed_dictionary = self.create_feed_dictionary(
                    tf_input_data, tf_input_labels, tf_keep_probability,
                    minibatch_data, minibatch_labels, keep_probability)

                _, loss, predictions, summary = session.run(
                    [optimizer, training_loss, training_predictions, merged_summary], feed_dict=feed_dictionary)
                losses.append(loss)

                summary_writer.add_summary(summary, step)

                if (step % 400 == 0):
                    print('Minibatch loss at step %d: %f' % (step, loss))
                    print('Minibatch accuracy: %.1f%%' % self.compute_predictions_accuracy(predictions, minibatch_labels))

                    validation_feed_dictionary = self.create_feed_dictionary(
                        tf_input_data, tf_input_labels, tf_keep_probability,
                        validation_data, validation_labels, 1.0)

                    validation_accuracy = self.compute_predictions_accuracy(
                        validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

                    training_accuracy_list.append(self.compute_predictions_accuracy(predictions, minibatch_labels))
                    validation_accuracy_list.append(validation_accuracy)
                    steps_list.append(step)

                    print('Validation accuracy: %.1f%%' % validation_accuracy)

            test_feed_dictionary = self.create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_keep_probability,
                test_data, test_labels, 1.0)

            test_accuracy = self.compute_predictions_accuracy(
                test_predictions.eval(feed_dict=test_feed_dictionary), test_labels)

            print('Test accuracy: %.1f%%' % test_accuracy)

        return training_accuracy_list, validation_accuracy_list, steps_list, test_accuracy

    def initialize_weights_and_biases(self):
        """
        Initialize the weights for the neural network using He initialization and initialize the biases to zero
        :return: weights dictionary
        :return: biases dictionary
        """

        weights = dict()
        biases = dict()

        hidden_units_1 = self.hidden_units[0]
        hidden_units_2 = self.hidden_units[1]
        hidden_units_3 = self.hidden_units[2]
        hidden_units_4 = self.hidden_units[3]

        # weights for the input layer
        weights_input_layer = tf.Variable(
            tf.truncated_normal([self.input_data_size, hidden_units_1],
                                stddev=math.sqrt(2.0 / float(self.input_data_size))))
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
            tf.truncated_normal([hidden_units_4, self.output_size], stddev=math.sqrt(2.0 / float(hidden_units_4))))
        weights['weights_forth_hidden_layer'] = weights_forth_hidden_layer

        # biases for output layer
        biases_output_layer = tf.Variable(tf.zeros(self.output_size))
        biases['biases_output_layer'] = biases_output_layer

        return weights, biases

    def compute_predictions(self, input_data, weights, biases, keep_probability):
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
            self, placeholder_data, placeholder_labels, placeholder_keep_probability, data, labels, keep_probability):
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

    def compute_loss(self, logits, labels, weights, weight_decay):
        """
        :param logits:
        :param labels:
        :param weights
        :return:
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        L2_loss = tf.nn.l2_loss(weights['weights_input_layer']) + \
                  tf.nn.l2_loss(weights['weights_first_hidden_layer']) + \
                  tf.nn.l2_loss(weights['weights_second_hidden_layer']) + \
                  tf.nn.l2_loss(weights['weights_third_hidden_layer']) + \
                  tf.nn.l2_loss(weights['weights_forth_hidden_layer'])

        loss = tf.reduce_mean(cross_entropy + L2_loss * weight_decay)

        return loss

    def compute_predictions_accuracy(self, predictions, labels):

        """
        :param predictions: labels given by the feedforward neural network
        :param labels: correct labels for the input date
        :return: percentage of predictions that match the correct labels
        """
        num_correct_labels = 0
        for index in range(predictions.shape[0]):
            if np.argmax(predictions[index]) == np.argmax(labels[index]):
                num_correct_labels += 1

        return (100 * num_correct_labels) / predictions.shape[0]


    def compute_confussion_matrix(self, test_predictions, test_labels):

        confusion_matrix = np.zeros(shape=(self.output_size, self.output_size))
        for index in range(test_predictions.shape[0]):
            predicted_class_index = np.argmax(test_predictions[index])
            actual_class_index = np.argmax(test_labels[index])

            confusion_matrix[actual_class_index][predicted_class_index] +=1

        return confusion_matrix

    def compute_ROC_points(self, test_predictions, test_labels):

        ROC_points = dict()
        ROC_points['y_true'] = []
        ROC_points['y_score'] = []

        for index in range(test_predictions.shape[0]):
            true_class = np.argmax(test_labels[index])
            ROC_points['y_true'] += [true_class]

            score = test_predictions[index][1]
            ROC_points['y_score'] += [score]

        return ROC_points