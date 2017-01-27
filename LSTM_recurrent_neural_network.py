import math
import numpy as np
import tensorflow as tf

# Hyperparameters
# input_size = 16
# num_units_1 = 64
# num_units_2 = 128
# num_units_3 = 256

# hidden_units_1 = 512
# hidden_units_2 = 256
# hidden_units_3 = 64
# hidden_units_4 = 32

keep_probability = 0.5

# Training parameters
learning_rate = 0.0001
weight_decay = 0.05
batch_size = 16

epsilon = 1e-3

class RecurrentNeuralNetwork:

    def __init__(self, input_sequence_length, input_step_size, LSTMs_state_size, hidden_units, output_size):
        self.input_sequence_length = input_sequence_length
        self.input_step_size = input_step_size
        self.LSTMs_state_size = LSTMs_state_size
        self.hidden_units = hidden_units
        self.output_size = output_size

    def train_and_validate(self, training_dataset, validation_dataset):
        """
        Train the feed forward neural network using gradient descent by trying to minimize the loss.
        This function is used for cross validation.

        :param training_dataset: dictionary containing the training data and training labels
        :param validation_dataset: dictionary containing the validation data and validation labels
        :return: the validation accuracy of the model
        """

        training_data = training_dataset["training_data"]
        training_labels = training_dataset["training_labels"]

        training_data = np.reshape(
            training_data, (len(training_data), self.input_sequence_length, self.input_step_size))

        validation_data = validation_dataset["validation_data"]
        validation_labels = validation_dataset["validation_labels"]

        validation_data = np.reshape(
            validation_data, (len(validation_data), self.input_sequence_length, self.input_step_size))

        graph = tf.Graph()
        with graph.as_default():

            # create placeholders for input tensors
            tf_input_data = tf.placeholder(
                tf.float32, shape=(None, self.input_sequence_length, self.input_step_size))
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, self.output_size))

            # create placeholder for the keep probability
            # dropout is used during training, but not during testing
            tf_keep_probability = tf.placeholder(tf.float32)

            # initialize weights and biases for the LSTM cell and MLP network
            weights, biases = self.initialize_weights_and_biases(
                self.input_step_size, self.LSTMs_state_size, self.hidden_units, self.output_size)

            # initialize the output and cell_state for the LSTM cells for training
            initial_LSMT_outputs, initial_LSMT_cell_states = \
                self.initialize_outputs_and_cell_states_for_LSTMs(batch_size, self.LSTMs_state_size)

            # use the model to perform inference
            logits = self.inference(
                tf_input_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSMT_outputs, initial_LSMT_cell_states,
                tf_keep_probability)

            training_loss = self.compute_loss(logits, tf_input_labels, weights)

            # TODO: describe how the AdamOptimizer works
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(training_loss)

            print("Training RNN")
            print(learning_rate)
            print(weight_decay)
            training_predictions = tf.nn.softmax(logits)

            # initialize the output and cell_state for the LSTM cells for validation
            initial_LSMT_outputs, initial_LSMT_cell_states = \
                self.initialize_outputs_and_cell_states_for_LSTMs(len(validation_data), self.LSTMs_state_size)

            validation_logits = self.inference(
                validation_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSMT_outputs, initial_LSMT_cell_states,
                tf_keep_probability)

            validation_predictions = tf.nn.softmax(validation_logits)

        steps = 9000
        with tf.Session(graph=graph) as session:

            # initialize weights and biases
            tf.initialize_all_variables().run()

            for step in range(steps):

                offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

                # Create a training minibatch.
                minibatch_data = training_data[offset:(offset + batch_size), :]
                minibatch_data = minibatch_data.reshape(batch_size, self.input_sequence_length, self.input_step_size)

                minibatch_labels = training_labels[offset:(offset + batch_size), :]

                feed_dictionary = self.create_feed_dictionary(
                    tf_input_data, tf_input_labels, tf_keep_probability,
                    minibatch_data, minibatch_labels, keep_probability)

                _, loss, predictions = session.run(
                    [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

                if (step % 500 == 0):
                    print('Minibatch loss at step %d: %f' % (step, loss))
                    print('Minibatch accuracy: %.1f%%' % self.compute_predictions_accuracy(predictions, minibatch_labels))

            validation_feed_dictionary = self.create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_keep_probability,
                validation_data, validation_labels, 1.0)

            validation_accuracy = self.compute_predictions_accuracy(
                validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

            print('Validation accuracy: %.1f%%' % validation_accuracy)

        return validation_accuracy

    def initializa_weights_and_biases_for_LSTM_cell(self, input_size, num_units):
        weights = dict()
        biases = dict()

        # Input gate
        input_cW = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))
        input_pW = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_input_cW'] = input_cW
        weights['LSTM_input_pW'] = input_pW

        input_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_input_biases'] = input_biases

        # Forget gate
        forget_cW = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))
        forget_pW = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_forget_cW'] = forget_cW
        weights['LSTM_forget_pW'] = forget_pW

        forget_biases = tf.Variable(tf.ones(num_units))
        biases['LSTM_forget_biases'] = forget_biases

        """
        Memory cell

        Use orthogonal initialization for the weights
        """
        s, u, v = tf.svd(tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size))),
                         compute_uv=True, full_matrices=False)
        memory_cell_cW = tf.Variable(tf.reshape(v, [input_size, num_units]))

        s, u, v = tf.svd(tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units))),
                         compute_uv=True, full_matrices=False)
        memory_cell_pW = tf.Variable(u)

        weights['LSTM_memory_cell_cW'] = memory_cell_cW
        weights['LSTM_memory_cell_pW'] = memory_cell_pW

        memory_cell_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_memory_cell_biases'] = memory_cell_biases

        # Output gate
        output_cW = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))
        output_pW = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_output_cW'] = output_cW
        weights['LSTM_output_pW'] = output_pW

        output_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_output_biases'] = output_biases

        return weights, biases

    def initialize_weights_and_biases_for_MLP(self, input_size, hidden_units, output_size):

        """
        Feed forward neural network layer used after the LSTM Units. The input to the feed forward neural netowork is
        the mean value of the last output of the LSTM cell

        Use Glorot initialization of the weights

        :param input_size:
        :param num_units:
        :param output_size:
        :return:
        """

        weights = dict()
        biases = dict()

        hidden_units_1 = hidden_units[0]
        hidden_units_2 = hidden_units[1]
        hidden_units_3 = hidden_units[2]
        hidden_units_4 = hidden_units[3]

        # weights for the input layer
        weights_input_layer = tf.Variable(
            tf.truncated_normal([input_size, hidden_units_1],
                                stddev=math.sqrt(2.0 / float(input_size))))
        weights['MLP_input_layer'] = weights_input_layer

        # biases for the first hidden layer
        biases_first_hidden_layer = tf.Variable(tf.zeros(hidden_units_1))
        biases['MLP_first_hidden_layer'] = biases_first_hidden_layer

        # weights for first hidden layer
        weights_first_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_1, hidden_units_2],
                                stddev=math.sqrt(2.0 / float(hidden_units_1))))
        weights['MLP_first_hidden_layer'] = weights_first_hidden_layer

        # biases for the second hidden layer
        biases_second_hidden_layer = tf.Variable(tf.zeros(hidden_units_2))
        biases['MLP_second_hidden_layer'] = biases_second_hidden_layer

        # weights for the second hidden layer
        weights_second_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_2, hidden_units_3],
                                stddev=math.sqrt(2.0 / float(hidden_units_2))))
        weights['MLP_second_hidden_layer'] = weights_second_hidden_layer

        # biases for third hidden layer
        biases_third_hidden_layer = tf.Variable(tf.zeros(hidden_units_3))
        biases['MLP_third_hidden_layer'] = biases_third_hidden_layer

        # weights for third hidden layer
        weights_third_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_3, hidden_units_4],
                                stddev=math.sqrt(2.0 / float(hidden_units_3))))
        weights['MLP_third_hidden_layer'] = weights_third_hidden_layer

        # biases for forth layer
        biases_forth_hidden_layer = tf.Variable(tf.zeros(hidden_units_4))
        biases['MLP_forth_hidden_layer'] = biases_forth_hidden_layer

        # weights for forth layer
        weights_forth_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_4, output_size], stddev=math.sqrt(2.0 / float(hidden_units_4))))
        weights['MLP_forth_hidden_layer'] = weights_forth_hidden_layer

        # biases for output layer
        biases_output_layer = tf.Variable(tf.zeros(output_size))
        biases['MLP_output_layer'] = biases_output_layer

        return weights, biases

    def initialize_weights_and_biases(self, input_step_size, LSTMs_state_size, hidden_units, output_size):
        weights = dict()
        biases = dict()

        weights['LSMT_1'], biases['LSTM_1'] = \
            self.initializa_weights_and_biases_for_LSTM_cell(input_step_size, LSTMs_state_size[0])
        weights['LSMT_2'], biases['LSTM_2'] = \
            self.initializa_weights_and_biases_for_LSTM_cell(LSTMs_state_size[0], LSTMs_state_size[1])
        weights['LSMT_3'], biases['LSTM_3'] = \
            self.initializa_weights_and_biases_for_LSTM_cell(LSTMs_state_size[1], LSTMs_state_size[2])

        weights['MLP'], biases['MLP'] = self.initialize_weights_and_biases_for_MLP(
            LSTMs_state_size[2], hidden_units, output_size)

        return weights, biases

    def initialize_outputs_and_cell_states_for_LSTMs(self, batch_size, LSTMs_state_size):

        num_units_1 = LSTMs_state_size[0]
        num_units_2 = LSTMs_state_size[1]
        num_units_3 = LSTMs_state_size[2]

        initial_LSTM_outputs = dict()
        initial_LSTM_cell_states = dict()

        initial_LSTM_outputs['LSTM_1'] = tf.Variable(tf.zeros([batch_size, num_units_1]), trainable=False)
        initial_LSTM_cell_states['LSTM_1'] = tf.Variable(tf.zeros([batch_size, num_units_1]), trainable=False)

        initial_LSTM_outputs['LSTM_2'] = tf.Variable(tf.zeros([batch_size, num_units_2]), trainable=False)
        initial_LSTM_cell_states['LSTM_2'] = tf.Variable(tf.zeros([batch_size, num_units_2]), trainable=False)

        initial_LSTM_outputs['LSTM_3'] = tf.Variable(tf.zeros([batch_size, num_units_3]), trainable=False)
        initial_LSTM_cell_states['LSTM_3'] = tf.Variable(tf.zeros([batch_size, num_units_3]), trainable=False)

        return initial_LSTM_outputs, initial_LSTM_cell_states

    def lstm(self, input_data, previous_hidden_state, previous_cell_state, weights, biases):
        """

        :param input_data:
        :param previous_hidden_state: previous output of the LSTM cell
        :param previous_cell_state:
        :param weights:
        :param biases:
        :return:
        """
        input_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_input_cW'])) + \
                     tf.matmul(previous_hidden_state, weights['LSTM_input_pW']) + \
                     biases['LSTM_input_biases']

        candidate_cell_state = tf.tanh(tf.matmul(input_data, weights['LSTM_memory_cell_cW'])) + \
                               tf.matmul(previous_hidden_state, weights['LSTM_memory_cell_pW']) + \
                               biases['LSTM_memory_cell_biases']

        forget_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_forget_cW'])) + \
                      tf.matmul(previous_hidden_state, weights['LSTM_forget_pW']) + \
                      biases['LSTM_forget_biases']

        new_cell_state = input_gate * candidate_cell_state + forget_gate * previous_cell_state

        output_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_output_cW'])) + \
                      tf.matmul(previous_hidden_state, weights['LSTM_output_pW']) + \
                      biases['LSTM_output_biases']

        new_hidden_state = output_gate * tf.tanh(new_cell_state)

        return new_hidden_state, new_cell_state

    def MLP_inference(self, input_data, weights, biases, keep_probability):

        # first hidden layer
        input_to_first_hidden_layer = \
            tf.matmul(input_data, weights['MLP_input_layer']) + biases['MLP_first_hidden_layer']
        mean, variance = tf.nn.moments(input_to_first_hidden_layer, [0])

        first_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_first_hidden_layer, mean, variance, None, None, epsilon)),
            keep_probability)

        # second hidden layer
        input_to_second_hidden_layer = \
            tf.matmul(first_hidden_layer, weights['MLP_first_hidden_layer']) + biases['MLP_second_hidden_layer']
        mean, variance = tf.nn.moments(input_to_second_hidden_layer, [0])

        second_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_second_hidden_layer, mean, variance, None, None, epsilon)),
            keep_probability)

        # third hidden layer
        input_to_third_hidden_layer = \
            tf.matmul(second_hidden_layer, weights['MLP_second_hidden_layer']) + biases['MLP_third_hidden_layer']
        mean, variance = tf.nn.moments(input_to_third_hidden_layer, [0])
        third_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_third_hidden_layer, mean, variance, None, None, epsilon)),
            keep_probability)

        # forth_hidden_layer
        input_to_forth_hidden_layer = \
            tf.matmul(third_hidden_layer, weights['MLP_third_hidden_layer']) + biases['MLP_forth_hidden_layer']
        mean, variance = tf.nn.moments(input_to_forth_hidden_layer, [0])

        forth_hidden_layer = tf.nn.relu(
            tf.nn.batch_normalization(input_to_forth_hidden_layer, mean, variance, None, None, epsilon))

        # output layer
        logits = tf.matmul(forth_hidden_layer, weights['MLP_forth_hidden_layer']) + biases['MLP_output_layer']

        return logits

    def inference(self, input_data, input_sequence_length, input_step_size, weights, biases, outputs, cell_states,
                  keep_probability):
        """
        The recurrent neural network processes the epigenetics data as a sequence of gene expressions.

        :param input_data:
        :param weights:
        :param biases:
        :return:
        """

        # Modify input data shape for the recurrent neural network
        # Current data shape: (batch_size, sequence_length, input_size)
        # Required data shape: (sequence_length, batch_size, input_size)

        input_data = tf.transpose(input_data, [1, 0, 2])
        input_data = tf.reshape(input_data, [-1, input_step_size])
        input_data = tf.split(0, input_sequence_length, input_data)

        RNN_outputs = list()
        LSTM_1_output = outputs['LSTM_1']
        LSTM_1_cell_state = cell_states['LSTM_1']

        LSTM_2_output = outputs['LSTM_2']
        LSTM_2_cell_state = cell_states['LSTM_2']

        LSTM_3_output = outputs['LSTM_3']
        LSTM_3_cell_state = cell_states['LSTM_3']

        for current_input in input_data:
            LSTM_1_output, LSTM_1_cell_state = \
                self.lstm(current_input, LSTM_1_output, LSTM_1_cell_state, weights['LSMT_1'], biases['LSTM_1'])
            # LSTM_1_output = tf.nn.dropout(LSTM_1_output, keep_probability)

            # mean, variance = tf.nn.moments(LSTM_1_output, [0])
            # LSTM_1_output = tf.nn.batch_normalization(LSTM_1_output, mean, variance, None, None, epsilon)

            LSTM_2_output, LSTM_2_cell_state = \
                self.lstm(LSTM_1_output, LSTM_2_output, LSTM_2_cell_state, weights['LSMT_2'], biases['LSTM_2'])

            # LSTM_2_output = tf.nn.dropout(LSTM_2_output, keep_probability)

            # mean, variance = tf.nn.moments(LSTM_2_output, [0])
            # LSTM_2_output = tf.nn.batch_normalization(LSTM_2_output, mean, variance, None, None, epsilon)

            LSTM_3_output, LSTM_3_cell_state = \
                self.lstm(LSTM_2_output, LSTM_3_output, LSTM_3_cell_state, weights['LSMT_3'], biases['LSTM_3'])

            RNN_outputs.append(LSTM_3_output)

        MLP_input = RNN_outputs[-1]

        mean, variance = tf.nn.moments(MLP_input, [0])
        MLP_input = tf.nn.batch_normalization(MLP_input, mean, variance, None, None, epsilon)

        # Multilayer perceptron network
        logits = self.MLP_inference(MLP_input, weights['MLP'], biases['MLP'], keep_probability)

        return logits

    def create_feed_dictionary(self,
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

    def compute_loss(self, logits, labels, weights):
        """
        :param logits:
        :param labels:
        :return:
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        L2_loss = tf.nn.l2_loss(weights['MLP']['MLP_input_layer']) + \
                  tf.nn.l2_loss(weights['MLP']['MLP_first_hidden_layer']) + \
                  tf.nn.l2_loss(weights['MLP']['MLP_second_hidden_layer']) + \
                  tf.nn.l2_loss(weights['MLP']['MLP_third_hidden_layer']) + \
                  tf.nn.l2_loss(weights['MLP']['MLP_forth_hidden_layer'])

        loss = tf.reduce_mean(cross_entropy + weight_decay * L2_loss)

        return loss

    def compute_predictions_accuracy(self, predictions, labels):
        """
        :param predictions: labels given by the feedforward neural network
        :param labels: correct labels for the input data
        :return: percentage of predictions that match the correct labels
        """
        num_correct_labels = 0
        for index in range(predictions.shape[0]):
            if np.argmax(predictions[index]) == np.argmax(labels[index]):
                num_correct_labels += 1

        return (100 * num_correct_labels) / predictions.shape[0]
