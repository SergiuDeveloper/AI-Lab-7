#!/usr/bin/python3.8

import math
import random
import copy


class NeuralNetwork:
    def __init__(self):
        self.__w = None
        self.__b = None

    def fit(self, X_train, y_train, max_epochs, learning_rate):
        self.__w = [
            self.__generate_random_float_not_null(-0.1, 0.1)
            for X in X_train[0]
        ]
        self.__b = self.__generate_random_float_not_null(-0.1, 0.1)

        epoch = 1
        mse = math.inf
        while epoch <= max_epochs or mse < 0.0001:
            mse = self.__compute_mse(X_train, y_train)

            prev_w = copy.copy(self.__w)
            prev_b = self.__b
            self.__update_weights(X_train, y_train, learning_rate)
            if prev_w == self.__w and prev_b == self.__b:
                break

            epoch += 1

    def predict(self, X):
        return 1 if self.__get_computed_function_result(X) >= 0.5 else 0

    def compute_accuracy(self, X_test, y_test):
        correct_predictions = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                correct_predictions += 1
        return correct_predictions / len(X_test)

    def __generate_random_float_not_null(self, min, max):
        x = 0
        while x == 0:
            x = random.uniform(min, max)
        return x

    def __sigmoid_activation(self, predicted_y):
        return 1 / (1 + math.exp(-predicted_y))

    def __compute_mse(self, X_train, y_train):
        mse = 0
        for i in range(len(X_train)):
            val = self.__b
            for j in range(len(X_train[i])):
                val += X_train[i][j] * self.__w[j]
            val = self.__sigmoid_activation(val)

            expected_val = self.__sigmoid_activation(y_train[i])

            mse += (expected_val - val) ** 2
        mse /= len(X_train)

        return mse

    def __get_computed_function_result(self, X):
        val = self.__b
        for i in range(len(X)):
            val += X[i] * self.__w[i]
        return val

    def __update_weights(self, X_train, y_train, learning_rate):
        updated_w = copy.copy(self.__w)

        for i in range(len(updated_w)):
            derivative = -2 / len(X_train)

            sum = 0
            for j in range(len(X_train)):
                sigmoid_result = self.__sigmoid_activation(self.__get_computed_function_result(X_train[j]))
                error = self.__sigmoid_activation(y_train[j]) - sigmoid_result
                sigmoid_activation_derivative = sigmoid_result * (1 - sigmoid_result)
                computed_function_weight_derivative = X_train[j][i]

                sum += error * sigmoid_activation_derivative * computed_function_weight_derivative

            derivative *= sum

            updated_w[i] += - derivative * learning_rate

        updated_b = self.__b

        derivative = -2 / len(X_train)
        sum = 0
        for j in range(len(X_train)):
            sigmoid_result = self.__sigmoid_activation(self.__get_computed_function_result(X_train[j]))
            error = self.__sigmoid_activation(y_train[j]) - sigmoid_result
            sigmoid_activation_derivative = sigmoid_result * (1 - sigmoid_result)
            computed_function_bias_derivative = 1

            sum += error * sigmoid_activation_derivative * computed_function_bias_derivative

        derivative *= sum

        updated_b += - derivative * learning_rate

        self.__w = updated_w
        self.__b = updated_b