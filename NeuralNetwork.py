#!/usr/bin/python3.8

import math
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.__w_hidden = np.array((0, 0))
        self.__w_output = np.array((0, 0))

    def fit(self, X_train, y_train, max_epochs, learning_rate):
        self.__w_hidden = np.random.uniform(low=0, high=1, size=(X_train.shape[1], X_train.shape[0]))
        self.__w_output = np.random.uniform(low=0, high=1, size=(X_train.shape[0], 1))

        epoch = 1
        mse = math.inf
        while epoch <= max_epochs and mse > 0.00001:
            mse = self.__compute_mse(X_train, y_train)

            layer_1 = self.__sigmoid_activation(np.dot(X_train, self.__w_hidden))
            layer_2 = self.__sigmoid_activation(np.dot(layer_1, self.__w_output))

            layer_2_error = y_train - layer_2
            layer_2_delta = np.multiply(layer_2_error, self.__sigmoid_derivative(layer_2))

            layer_1_error = np.dot(layer_2_delta, self.__w_output.T)
            layer_1_delta = np.multiply(layer_1_error, self.__sigmoid_derivative(layer_1))

            self.__w_output += np.dot(layer_1.T, layer_2_delta) * learning_rate
            self.__w_hidden += np.dot(X_train.T, layer_1_delta) * learning_rate

            epoch += 1

    def predict(self, X):
        layer_1_output = self.__sigmoid_activation(np.dot(X, self.__w_hidden))
        layer_2_output = self.__sigmoid_activation(np.dot(layer_1_output, self.__w_output))
        return 1 if layer_2_output > 0.5 else 0

    def compute_accuracy(self, X_test, y_test):
        correct_predictions = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                correct_predictions += 1
        return correct_predictions / len(X_test)

    def __sigmoid_activation(self, predicted_y):
        return 1 / (1 + np.exp(-predicted_y))

    def __sigmoid_derivative(self, predicted_y_derivative):
        return predicted_y_derivative * (1 - predicted_y_derivative)

    def __compute_mse(self, X_train, y_train):
        predictions = np.array([self.predict(X) for X in X_train])
        return (np.square(y_train - predictions)).mean(axis=None)
