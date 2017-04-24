import tensorflow as tf
import unittest
import numpy as np
from multilayer_perceptron import MultilayerPerceptron
from recurrent_neural_network import RecurrentNeuralNetwork
from superlayered_neural_network import SuperlayeredNeuralNetwork

class TestEvaluationMetrics(unittest.TestCase):

    def test_evaluation_metrics_MLP(self):
        mlp = MultilayerPerceptron(16, [16, 8, 4, 2], 2)
        test_predictions = np.array([[0.9, 0.1], [0.01, 0.99]])
        labels = np.array([[1, 0], [0, 1]])
        test_acc = mlp.compute_predictions_accuracy(test_predictions, labels)
        confusion_matrix = mlp.compute_confussion_matrix(test_predictions, labels)
        self.assertEqual(test_acc, 100)
        np.testing.assert_array_equal(confusion_matrix, np.array([[1, 0], [0, 1]]))

    def test_evaluation_metrics_RNN(self):
        rnn = RecurrentNeuralNetwork(16, 4, [16, 8], [16, 8], 2)
        test_predictions = np.array([[0.9, 0.1], [0.01, 0.99]])
        labels = np.array([[1, 0], [0, 1]])
        test_acc = rnn.compute_predictions_accuracy(test_predictions, labels)
        confusion_matrix = rnn.compute_confussion_matrix(test_predictions, labels)
        self.assertEqual(test_acc, 100)
        np.testing.assert_array_equal(confusion_matrix, np.array([[1, 0], [0, 1]]))

    def test_evaluation_metrics_SNN(self):
        snn = SuperlayeredNeuralNetwork(7, [16, 8, 4, 2], [16, 8, 4, 2], 2)
        test_predictions = np.array([[0.9, 0.1], [0.01, 0.99]])
        labels = np.array([[1, 0], [0, 1]])
        test_acc = snn.compute_predictions_accuracy(test_predictions, labels)
        confusion_matrix = snn.compute_confussion_matrix(test_predictions, labels)
        self.assertEqual(test_acc, 100)
        np.testing.assert_array_equal(confusion_matrix, np.array([[1, 0], [0, 1]]))




if __name__ == '__main__':
    unittest.main()