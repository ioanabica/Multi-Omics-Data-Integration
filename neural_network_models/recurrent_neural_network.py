import math
import numpy as np
import tensorflow as tf

# FCL_keep_probability = 0.8
# LSTMs_keep_probability = 0.8

# Training parameters
# learning_rate = 0.0001
# weight_decay = 0.001
batch_size = 16

epsilon = 1e-3


class RecurrentNeuralNetwork(object):
    def __init__(self, input_sequence_length, input_step_size, LSTM_units_state_size, hidden_units, output_size):
        self.input_sequence_length = input_sequence_length
        self.input_step_size = input_step_size
        self.LSTMs_state_size = LSTM_units_state_size
        self.hidden_units = hidden_units
        self.output_size = output_size

        print input_sequence_length
        print input_step_size
        print LSTM_units_state_size
        print hidden_units
        print output_size

    def train_and_evaluate(
            self, training_dataset, test_dataset, learning_rate, weight_decay, keep_probability):
        """
        Train the feed forward neural network using gradient descent by trying to minimize the loss.
        This function is used for cross validation.

        :param training_dataset: dictionary containing the training data and training labels
        :param test_dataset: dictionary containing the validation data and validation labels
        :return: the validation accuracy of the model
        """

        training_data = training_dataset["training_data"]
        training_labels = training_dataset["training_labels"]

        training_data = np.reshape(
            training_data, (len(training_data), self.input_sequence_length, self.input_step_size))

        validation_data = test_dataset["validation_data"]
        validation_labels = test_dataset["validation_labels"]

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

            # initialize weights and biases for the LSTM cell and FCL network
            weights, biases = self.initialize_weights_and_biases(
                self.input_step_size, self.LSTMs_state_size, self.hidden_units, self.output_size)

            # initialize the output and cell_state for the LSTM cells for training
            initial_LSTM_outputs, initial_LSTM_cell_states = \
                self.initialize_LSTM_units_output_and_cell_state(batch_size, self.LSTMs_state_size)

            # use the model to perform inference
            logits = self.compute_predictions(
                tf_input_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSTM_outputs, initial_LSTM_cell_states, tf_keep_probability)

            training_loss = self.compute_loss(logits, tf_input_labels, weights, weight_decay)

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(training_loss)

            print("Training LSTM for Testing")
            print(learning_rate)
            print(weight_decay)
            print keep_probability
            training_predictions = tf.nn.softmax(logits)

            # initialize the output and cell_state for the LSTM cells for validation
            initial_LSTM_outputs, initial_LSTM_cell_states = \
                self.initialize_LSTM_units_output_and_cell_state(len(validation_data), self.LSTMs_state_size)

            validation_logits = self.compute_predictions(
                validation_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSTM_outputs, initial_LSTM_cell_states,
                keep_probability)

            validation_predictions = tf.nn.softmax(validation_logits)

        #steps = 12000
        steps = 12000
        training_accuracy = list()
        losses = list()
        with tf.Session(graph=graph) as session:

            # initialize weights and biases
            init = tf.global_variables_initializer()
            session.run(init)

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
                #training_accuracy.append(self.compute_predictions_accuracy(predictions, minibatch_labels))
                losses.append(loss)

                if (step % 500 == 0):
                    print('Minibatch loss at step %d: %f' % (step, loss))
                    print(
                    'Minibatch accuracy: %.1f%%' % self.compute_predictions_accuracy(predictions, minibatch_labels))

            validation_feed_dictionary = self.create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_keep_probability,
                validation_data, validation_labels, 1.0)

            validation_pred = validation_predictions.eval(feed_dict=validation_feed_dictionary)

            validation_accuracy = self.compute_predictions_accuracy(
                validation_pred, validation_labels)

            confussion_matrix = self.compute_confussion_matrix(
                            validation_pred, validation_labels)

            ROC_points = self.compute_ROC_points(
                validation_pred, validation_labels)

            print('Validation accuracy: %.1f%%' % validation_accuracy)

        return validation_accuracy, confussion_matrix, ROC_points


    def train_validate_test(
            self, training_dataset, validation_dataset, test_dataset,
            learning_rate, weight_decay, LSTMs_keep_probability, FCL_keep_probability):
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

        test_data = test_dataset["test_data"]
        test_labels = test_dataset["test_labels"]

        test_data = np.reshape(
            test_data, (len(test_data), self.input_sequence_length, self.input_step_size))

        graph = tf.Graph()
        with graph.as_default():

            # create placeholders for input tensors
            tf_input_data = tf.placeholder(
                tf.float32, shape=(None, self.input_sequence_length, self.input_step_size))
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, self.output_size))

            # create placeholder for the keep probability
            # dropout is used during training, but not during testing
            tf_FCL_keep_probability = tf.placeholder(tf.float32)
            tf_LSTMs_keep_probability = tf.placeholder(tf.float32)

            # initialize weights and biases for the LSTM cell and FCL network
            weights, biases = self.initialize_weights_and_biases(
                self.input_step_size, self.LSTMs_state_size, self.hidden_units, self.output_size)

            # initialize the output and cell_state for the LSTM cells for training
            initial_LSTM_outputs, initial_LSTM_cell_states = \
                self.initialize_LSTM_units_output_and_cell_state(batch_size, self.LSTMs_state_size)

            # use the model to perform inference
            logits = self.compute_predictions(
                tf_input_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSTM_outputs, initial_LSTM_cell_states,
                tf_LSTMs_keep_probability, tf_FCL_keep_probability)

            training_loss = self.compute_loss(logits, tf_input_labels, weights, weight_decay)

            # TODO: describe how the AdamOptimizer works
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(training_loss)

            """
            optimizer_function = tf.train.AdamOptimizer(learning_rate)

            # Gradient clipping

            gvs = optimizer_function.compute_gradients(training_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            optimizer = optimizer_function.apply_gradients(capped_gvs)
            """

            print("Training LSTM for Testing")
            print(learning_rate)
            print(weight_decay)
            print FCL_keep_probability
            print LSTMs_keep_probability
            training_predictions = tf.nn.softmax(logits)

            # initialize the output and cell_state for the LSTM cells for validation
            initial_LSTM_outputs, initial_LSTM_cell_states = \
                self.initialize_LSTM_units_output_and_cell_state(len(validation_data), self.LSTMs_state_size)

            validation_logits = self.compute_predictions(
                validation_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSTM_outputs, initial_LSTM_cell_states,
                tf_LSTMs_keep_probability, tf_FCL_keep_probability)

            validation_predictions = tf.nn.softmax(validation_logits)

            # initialize the output and cell_state for the LSTM cells for testing
            initial_LSTM_outputs, initial_LSTM_cell_states = \
                self.initialize_LSTM_units_output_and_cell_state(len(test_data), self.LSTMs_state_size)

            test_logits = self.compute_predictions(
                test_data, self.input_sequence_length, self.input_step_size,
                weights, biases,
                initial_LSTM_outputs, initial_LSTM_cell_states,
                tf_LSTMs_keep_probability, tf_FCL_keep_probability)

            test_predictions = tf.nn.softmax(test_logits)

        steps = 20000

        training_accuracy_list = list()
        validation_accuracy_list = list()
        steps_list = list()

        losses = list()
        with tf.Session(graph=graph) as session:

            # initialize weights and biases
            init = tf.global_variables_initializer()
            session.run(init)

            for step in range(steps):

                offset = (step * batch_size) % (training_labels.shape[0] - batch_size)

                # Create a training minibatch.
                minibatch_data = training_data[offset:(offset + batch_size), :]
                minibatch_data = minibatch_data.reshape(batch_size, self.input_sequence_length, self.input_step_size)

                minibatch_labels = training_labels[offset:(offset + batch_size), :]

                feed_dictionary = self.create_feed_dictionary(
                    tf_input_data, tf_input_labels, tf_LSTMs_keep_probability, tf_FCL_keep_probability,
                    minibatch_data, minibatch_labels, LSTMs_keep_probability, FCL_keep_probability)

                _, loss, predictions = session.run(
                    [optimizer, training_loss, training_predictions], feed_dict=feed_dictionary)

                losses.append(loss)

                if (step % 400 == 0):
                    print('Minibatch loss at step %d: %f' % (step, loss))
                    print(
                    'Minibatch accuracy: %.1f%%' % self.compute_predictions_accuracy(predictions, minibatch_labels))

                    validation_feed_dictionary = self.create_feed_dictionary(
                        tf_input_data, tf_input_labels, tf_LSTMs_keep_probability, tf_FCL_keep_probability,
                        validation_data, validation_labels, 1.0, 1.0)

                    validation_accuracy = self.compute_predictions_accuracy(
                        validation_predictions.eval(feed_dict=validation_feed_dictionary), validation_labels)

                    training_accuracy_list.append(self.compute_predictions_accuracy(predictions, minibatch_labels))
                    validation_accuracy_list.append(validation_accuracy)
                    steps_list.append(step)

                    print('Validation accuracy: %.1f%%' % validation_accuracy)

            """ After training, compute the accuracy the model gets on the test data"""

            test_feed_dictionary = self.create_feed_dictionary(
                tf_input_data, tf_input_labels, tf_LSTMs_keep_probability, tf_FCL_keep_probability,
                test_data, test_labels, 1.0, 1.0)

            test_accuracy = self.compute_predictions_accuracy(
                test_predictions.eval(feed_dict=test_feed_dictionary), test_labels)

            print('Test accuracy: %.1f%%' % test_accuracy)

        return training_accuracy_list, validation_accuracy_list, steps_list, test_accuracy

    def initializa_weights_and_biases_for_LSTM_cell(self, input_size, num_units):
        weights = dict()
        biases = dict()

        # Input gate
        input_W = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))

        s, u, v = tf.svd(tf.random_normal([num_units, num_units], mean=0, stddev=math.sqrt(1/num_units)),
                         compute_uv=True, full_matrices=False)

        input_U = tf.Variable(tf.reshape(v, [num_units, num_units]))
        input_U = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_input_W'] = input_W
        weights['LSTM_input_U'] = input_U

        input_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_input_biases'] = input_biases

        # Forget gate
        forget_W = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))

        s, u, v = tf.svd(tf.random_normal([num_units, num_units], mean=0, stddev=math.sqrt(1/num_units)),
                         compute_uv=True, full_matrices=False)

        forget_U = tf.Variable(tf.reshape(v, [num_units, num_units]))
        forget_U = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_forget_W'] = forget_W
        weights['LSTM_forget_U'] = forget_U

        forget_biases = tf.Variable(tf.ones(num_units))
        biases['LSTM_forget_biases'] = forget_biases

        #Memory Cell

        memory_cell_W = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))

        s, u, v = tf.svd(tf.random_normal([num_units, num_units], mean=0, stddev=math.sqrt(1/num_units)),
                         compute_uv=True, full_matrices=False)

        memory_cell_U = tf.Variable(tf.reshape(v, [num_units, num_units]))
        #memory_cell_U = tf.Variable(tf.diag(tf.ones(num_units)))
        memory_cell_U = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))

        weights['LSTM_memory_cell_W'] = memory_cell_W
        weights['LSTM_memory_cell_U'] = memory_cell_U

        memory_cell_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_memory_cell_biases'] = memory_cell_biases

        # Output gate
        output_W = tf.Variable(
            tf.truncated_normal([input_size, num_units], stddev=math.sqrt(2.0 / float(input_size + num_units))))

        s, u, v = tf.svd(tf.random_normal([num_units, num_units], mean=0, stddev=math.sqrt(1/num_units)),
                         compute_uv=True, full_matrices=False)

        output_U = tf.Variable(tf.reshape(v, [num_units, num_units]))
        output_U = tf.Variable(
            tf.truncated_normal([num_units, num_units], stddev=math.sqrt(2.0 / float(num_units + num_units))))
        weights['LSTM_output_W'] = output_W
        weights['LSTM_output_U'] = output_U

        output_biases = tf.Variable(tf.zeros(num_units))
        biases['LSTM_output_biases'] = output_biases

        return weights, biases

    def initialize_weights_and_biases_for_FCL(self, input_size, hidden_units, output_size):

        """
        Feed forward neural network layer used after the LSTM Units. The input to the feed forward neural netowork is
        the value of the last output of the LSTM cell.

        Use Glorot initialization of the weights

        :param input_size:
        :param hidden_units:
        :param output_size:
        :return:
        """

        weights = dict()
        biases = dict()

        hidden_units_1 = hidden_units[0]
        hidden_units_2 = hidden_units[1]

        # weights for the input layer
        weights_input_layer = tf.Variable(
            tf.truncated_normal([input_size, hidden_units_1],
                                stddev=math.sqrt(2.0 / float(input_size))))
        weights['FCL_input_layer'] = weights_input_layer

        # biases for the first hidden layer
        biases_first_hidden_layer = tf.Variable(tf.zeros(hidden_units_1))
        biases['FCL_first_hidden_layer'] = biases_first_hidden_layer

        # weights for first hidden layer
        weights_first_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_1, hidden_units_2],
                                stddev=math.sqrt(2.0 / float(hidden_units_1))))
        weights['FCL_first_hidden_layer'] = weights_first_hidden_layer

        # biases for the second hidden layer
        biases_second_hidden_layer = tf.Variable(tf.zeros(hidden_units_2))
        biases['FCL_second_hidden_layer'] = biases_second_hidden_layer

        # weights for the second hidden layer
        weights_second_hidden_layer = tf.Variable(
            tf.truncated_normal([hidden_units_2, output_size],
                                stddev=math.sqrt(2.0 / float(hidden_units_2))))
        weights['FCL_second_hidden_layer'] = weights_second_hidden_layer

        # biases for output layer
        biases_output_layer = tf.Variable(tf.zeros(output_size))
        biases['FCL_output_layer'] = biases_output_layer

        return weights, biases

    def initialize_weights_and_biases(self, input_step_size, LSTMs_state_size, hidden_units, output_size):
        weights = dict()
        biases = dict()

        weights['LSTM_1'], biases['LSTM_1'] = \
            self.initializa_weights_and_biases_for_LSTM_cell(input_step_size, LSTMs_state_size[0])
        weights['LSTM_2'], biases['LSTM_2'] = \
            self.initializa_weights_and_biases_for_LSTM_cell(LSTMs_state_size[0], LSTMs_state_size[1])

        weights['FCL'], biases['FCL'] = self.initialize_weights_and_biases_for_FCL(
            LSTMs_state_size[1], hidden_units, output_size)

        return weights, biases

    def initialize_LSTM_units_output_and_cell_state(self, batch_size, LSTMs_state_size):

        num_units_1 = LSTMs_state_size[0]
        num_units_2 = LSTMs_state_size[1]

        initial_LSTM_outputs = dict()
        initial_LSTM_cell_states = dict()

        initial_LSTM_outputs['LSTM_1'] = tf.Variable(tf.zeros([batch_size, num_units_1]), trainable=False)
        initial_LSTM_cell_states['LSTM_1'] = tf.Variable(tf.zeros([batch_size, num_units_1]), trainable=False)

        initial_LSTM_outputs['LSTM_2'] = tf.Variable(tf.zeros([batch_size, num_units_2]), trainable=False)
        initial_LSTM_cell_states['LSTM_2'] = tf.Variable(tf.zeros([batch_size, num_units_2]), trainable=False)

        return initial_LSTM_outputs, initial_LSTM_cell_states

    def perform_LSTM_unit_operations(self, input_data, previous_hidden_state, previous_cell_state, weights, biases):
        """

        :param input_data:
        :param previous_hidden_state: previous output of the LSTM cell
        :param previous_cell_state:
        :param weights:
        :param biases:
        :return:
        """
        input_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_input_W'])) + \
                     tf.matmul(previous_hidden_state, weights['LSTM_input_U']) + \
                     biases['LSTM_input_biases']

        candidate_cell_state = tf.tanh(tf.matmul(input_data, weights['LSTM_memory_cell_W'])) + \
                               tf.matmul(previous_hidden_state, weights['LSTM_memory_cell_U']) + \
                               biases['LSTM_memory_cell_biases']

        forget_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_forget_W'])) + \
                      tf.matmul(previous_hidden_state, weights['LSTM_forget_U']) + \
                      biases['LSTM_forget_biases']

        new_cell_state = input_gate * candidate_cell_state + forget_gate * previous_cell_state

        output_gate = tf.sigmoid(tf.matmul(input_data, weights['LSTM_output_W'])) + \
                      tf.matmul(previous_hidden_state, weights['LSTM_output_U']) + \
                      biases['LSTM_output_biases']

        new_hidden_state = output_gate * tf.tanh(new_cell_state)

        return new_hidden_state, new_cell_state

    def FCL_inference(self, input_data, weights, biases, keep_probability):

        # first hidden layer
        input_to_first_hidden_layer = \
            tf.matmul(input_data, weights['FCL_input_layer']) + biases['FCL_first_hidden_layer']
        mean, variance = tf.nn.moments(input_to_first_hidden_layer, [0])

        first_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_first_hidden_layer, mean, variance, None, None, epsilon)),
            keep_probability)

        # second hidden layer
        input_to_second_hidden_layer = \
            tf.matmul(first_hidden_layer, weights['FCL_first_hidden_layer']) + biases['FCL_second_hidden_layer']
        mean, variance = tf.nn.moments(input_to_second_hidden_layer, [0])

        second_hidden_layer = tf.nn.dropout(tf.nn.relu(
            tf.nn.batch_normalization(input_to_second_hidden_layer, mean, variance, None, None, epsilon)),
            keep_probability)

        # output layer
        logits = tf.matmul(second_hidden_layer, weights['FCL_second_hidden_layer']) + biases['FCL_output_layer']

        return logits

    def compute_predictions(self, input_data, input_sequence_length, input_step_size, weights, biases,
                            outputs, cell_states, keep_probability):
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
        input_data = tf.split(input_data, input_sequence_length, 0)

        RNN_outputs = list()
        LSTM_1_output = outputs['LSTM_1']
        LSTM_1_cell_state = cell_states['LSTM_1']

        LSTM_2_output = outputs['LSTM_2']
        LSTM_2_cell_state = cell_states['LSTM_2']

        for current_input in input_data:
            LSTM_1_output, LSTM_1_cell_state = \
                self.perform_LSTM_unit_operations(current_input,
                                                  LSTM_1_output,
                                                  LSTM_1_cell_state,
                                                  weights['LSTM_1'],
                                                  biases['LSTM_1'])

            LSTM_2_input = tf.nn.dropout(LSTM_1_output, keep_probability)
            LSTM_2_output, LSTM_2_cell_state = self.perform_LSTM_unit_operations(
                LSTM_2_input,
                LSTM_2_output,
                LSTM_2_cell_state,
                weights['LSTM_2'],
                biases['LSTM_2'])

            RNN_outputs.append(LSTM_2_output)

        FCL_input = RNN_outputs[-1]

        mean, variance = tf.nn.moments(FCL_input, [0])
        FCL_input = tf.nn.batch_normalization(FCL_input, mean, variance, None, None, epsilon)

        # Multilayer perceptron network
        logits = self.FCL_inference(FCL_input, weights['FCL'], biases['FCL'], keep_probability)

        return logits

    def create_feed_dictionary(self, placeholder_data, placeholder_labels,
                               placeholder_keep_probability,
                               data, labels, keep_probability):
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
        :return:
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        FCL_L2_loss = tf.nn.l2_loss(weights['FCL']['FCL_input_layer']) + \
                      tf.nn.l2_loss(weights['FCL']['FCL_first_hidden_layer']) + \
                      tf.nn.l2_loss(weights['FCL']['FCL_second_hidden_layer'])

        LSTM_1_L2_loss = tf.nn.l2_loss(weights['LSTM_1']['LSTM_input_W']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_input_U']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_forget_W']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_forget_U']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_memory_cell_W']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_memory_cell_U']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_output_W']) + \
                         tf.nn.l2_loss(weights['LSTM_1']['LSTM_output_U'])

        LSTM_2_L2_loss = tf.nn.l2_loss(weights['LSTM_2']['LSTM_input_W']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_input_U']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_forget_W']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_forget_U']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_memory_cell_W']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_memory_cell_U']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_output_W']) + \
                         tf.nn.l2_loss(weights['LSTM_2']['LSTM_output_U'])

        #loss = tf.reduce_mean(
            #cross_entropy + weight_decay * (
                #FCL_L2_loss + LSTM_1_L2_loss + LSTM_2_L2_loss))

        loss = tf.reduce_mean(cross_entropy + weight_decay * FCL_L2_loss)

        return loss

    def compute_predictions_accuracy(self, predictions, labels):
        """
        :param predictions: labels given by the feedforward neural network
        :param labels: correct labels for the input data
        :return: percentage of predictions that match the correct labels
        """
        num_correct_labels = 0
        for index in range(predictions.shape[0]):
            #print str(np.argmax(predictions[index])) + " " + str(np.argmax(labels[index]))
            if np.argmax(predictions[index]) == np.argmax(labels[index]):
                num_correct_labels += 1
        return (100 * num_correct_labels) / predictions.shape[0]

    def compute_confussion_matrix(self, predictions, labels):

        confusion_matrix = np.zeros(shape=(self.output_size, self.output_size))
        for index in range(predictions.shape[0]):
            predicted_class_index = np.argmax(predictions[index])
            #print str(predictions[index]) + " " + str(predicted_class_index)
            actual_class_index = np.argmax(labels[index])
            #print str(labels[index]) + " " + str(actual_class_index)
            confusion_matrix[actual_class_index][predicted_class_index] += 1

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