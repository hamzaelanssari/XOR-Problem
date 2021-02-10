import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from leaky_relu import leaky_relu
from relu import relu
from sigmoid import sigmoid
from sign import sign
from softmax import softmax
from step import step
from tanh import tanh

''' MultiLayerNN Class '''


class MultiLayerNN:
    def __init__(self, epochs, lr, num_input_layers, num_hidden_layers, num_output_layers):
        self.losses = []

        # Epochs
        self.epochs = epochs

        # Learning rate
        self.lr = lr

        # Random weights and bias initialization
        self.hidden_weights = np.random.uniform(size=(num_input_layers, num_hidden_layers))
        self.hidden_bias = np.random.uniform(size=(1, num_hidden_layers))
        self.output_weights = np.random.uniform(size=(num_hidden_layers, num_output_layers))
        self.output_bias = np.random.uniform(size=(1, num_output_layers))

        # Activation function initialization
        self.hidden_function = None
        self.output_function = None

    def hidden_activation_function(self, hidden_activation_function):
        if hidden_activation_function == 'sigmoid':
            self.hidden_function = sigmoid
        elif hidden_activation_function == 'softmax':
            self.hidden_function = softmax
        elif hidden_activation_function == 'relu':
            self.hidden_function = relu
        elif hidden_activation_function == 'tanh':
            self.hidden_function = tanh
        elif hidden_activation_function == 'leaky_relu':
            self.hidden_function = leaky_relu
        elif hidden_activation_function == 'sign':
            self.hidden_function = sign
        elif hidden_activation_function == 'step':
            self.hidden_function = step
        else:
            self.hidden_function = relu

    def output_activation_function(self, output_activation_function):
        if output_activation_function == 'sigmoid':
            self.output_function = sigmoid
        elif output_activation_function == 'softmax':
            self.output_function = softmax
        elif output_activation_function == 'relu':
            self.output_function = relu
        elif output_activation_function == 'tanh':
            self.output_function = tanh
        elif output_activation_function == 'leaky_relu':
            self.output_function = leaky_relu
        elif output_activation_function == 'sign':
            self.output_function = sign
        elif output_activation_function == 'step':
            self.output_function = step
        else:
            self.output_function = softmax

    def activation_function(self, hidden_activation_function, output_activation_function):
        self.hidden_activation_function(hidden_activation_function)
        self.output_activation_function(output_activation_function)

    # Loss Function
    def loss(self, yp, y):
        return (1 / 2) * np.square(yp - y)

    def forward(self, inputs):
        # Hidden Layer
        hidden_layer_activation = np.dot(inputs, self.hidden_weights) + self.hidden_bias

        hidden_layer_output = self.hidden_function.function(hidden_layer_activation)

        # Output Layer
        output_layer_activation = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        predicted_output = self.output_function.function(output_layer_activation)
        return hidden_layer_output, predicted_output

    def backward(self, hidden_layer_output, predicted_output):
        # Output Layer
        error = self.expected_output - predicted_output
        d_predicted_output = error * self.output_function.derivative(predicted_output)

        # Hidden Layer
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)

        d_hidden_layer = error_hidden_layer * self.hidden_function.derivative(hidden_layer_output)
        return d_hidden_layer, d_predicted_output

    def fit(self, X, y):
        np.random.seed(0)

        # Input data
        self.inputs = X
        self.expected_output = y.reshape(len(y), 1)

        # Training algorithm
        for _ in range(self.epochs):
            # Forward Propagation
            hidden_layer_output, predicted_output = self.forward(self.inputs)

            # Backpropagation
            d_hidden_layer, d_predicted_output = self.backward(hidden_layer_output, predicted_output)

            # Updating Weights and Biases
            self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * self.lr
            self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.lr
            self.hidden_weights += self.inputs.T.dot(d_hidden_layer) * self.lr
            self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.lr

            # Loss
            loss_ = self.loss(self.expected_output, predicted_output)[0]
            self.losses.append(loss_)

    def predict(self, inputs):
        predicted_output = self.forward(inputs)[1]
        predicted_output = np.squeeze(predicted_output)
        return np.where(predicted_output >= 0.5, 1, 0)

    def info_of_classification(self):
        predicted_output = self.predict(self.inputs)
        print("Accuracy : ", self.accuracy(predicted_output, self.expected_output))
        print("F1 score : ", self.f1_score(predicted_output, self.expected_output))
        print("Recall score : ", self.recall_score(predicted_output, self.expected_output))
        print("Precision score: ", self.precision_score(predicted_output, self.expected_output))
        print("Confusion Matrix : ", self.confusion_matrix(predicted_output, self.expected_output))
        return self.accuracy(predicted_output, self.expected_output), self.f1_score(predicted_output,
                                                                                    self.expected_output), self.recall_score(
            predicted_output, self.expected_output), self.precision_score(predicted_output, self.expected_output)

    def accuracy(self, predicted_output, outputs):
        return accuracy_score(predicted_output, outputs)

    def confusion_matrix(self, predicted_output, outputs):
        return confusion_matrix(outputs, predicted_output)

    def f1_score(self, predicted_output, outputs):
        return f1_score(outputs, predicted_output)

    def recall_score(self, predicted_output, outputs):
        return recall_score(outputs, predicted_output)

    def precision_score(self, predicted_output, outputs):
        return precision_score(outputs, predicted_output)

    def draw_loss(self):
        """ plt.plot(self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()"""
        return self.losses


''' End MultiLayerNN Class'''
