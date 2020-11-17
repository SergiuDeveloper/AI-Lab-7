#!/usr/bin/python3.8

from NeuralNetwork import NeuralNetwork
import itertools
import numpy as np


def train_on_all_boolean_functions(max_epochs=10000, learning_rate=0.1):
    print()

    neural_network = NeuralNetwork()

    X_train = list(itertools.product([0, 1], repeat=2))
    X_train = np.array(X_train)
    for y_train in itertools.product([0, 1], repeat=len(X_train)):
        y_train = np.array([[y] for y in y_train])
        neural_network.fit(X_train, y_train, max_epochs, learning_rate)
        accuracy = neural_network.compute_accuracy(X_train, y_train)

        print('Accuracy = {0}'.format(accuracy))

        for i in range(len(X_train)):
            print('x1={0} x2={1} y={2} predicted_y={3}'.format(X_train[i][0], X_train[i][1], y_train[i],
                                                               neural_network.predict(X_train[i])))
        print()


def train_on_specific_function(y_train, max_epochs=10000, learning_rate=0.1):
    print()

    neural_network = NeuralNetwork()

    X_train = list(itertools.product([0, 1], repeat=2))
    X_train = np.array(X_train)

    y_train = np.array(y_train)

    neural_network.fit(X_train, y_train, max_epochs, learning_rate)

    accuracy = neural_network.compute_accuracy(X_train, y_train)
    print('Accuracy = {0}'.format(accuracy))

    for i in range(len(X_train)):
        print('x1={0} x2={1} y={2} predicted_y={3}'.format(X_train[i][0], X_train[i][1], y_train[i],
                                                           neural_network.predict(X_train[i])))
    print()


if __name__ == '__main__':
    train_specific = True if input(
        'Do you want to train the model on a specific function? (Yes/No): ').lower() == 'yes' else False

    y_train = None
    if train_specific:
        print('Enter the values of the desired function:')

        y_train = [
            [int(input('x1={0} x2={1} y='.format(i, j)))]
            for i in range(2)
            for j in range(2)
        ]

    specify_hyperparams = True if input(
        'Do you want to use specific hyperparameters for the model? (Yes/No): ').lower() == 'yes' else False
    max_epochs = None
    learning_rate = None
    if specify_hyperparams:
        max_epochs = int(input('What is the maximum number of epochs that the model should train on? '))
        learning_rate = float(input('Which should the learning rate be? '))

    if train_specific:
        if specify_hyperparams:
            train_on_specific_function(y_train, max_epochs, learning_rate)
        else:
            train_on_specific_function(y_train)
    else:
        if specify_hyperparams:
            train_on_all_boolean_functions(max_epochs, learning_rate)
        else:
            train_on_all_boolean_functions()
