import numpy as np
from typing import List


class Layer:
    def __init__(self, nodes):
        self.nodes = nodes

    def activation(self, x):
        raise NotImplementedError("Please Implement this method")

    def activation_prime(self, x):
        raise NotImplementedError("Please Implement this method")


class LinearLayer(Layer):
    def activation(self, x):
        return x

    def activation_prime(self, x):
        return 1


class TanhLayer(Layer):
    def activation(self, x):
        return np.tanh(x)

    def activation_prime(self, x):
        return 1.0 - x ** 2


class SigmoidLayer(Layer):
    def activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def activation_prime(self, x):
        return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, layers: List[Layer]):

        self.layers = layers
        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1].nodes + 1, layers[i].nodes + 1)) - 1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i].nodes + 1, layers[i + 1].nodes)) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            #if k % 10000 == 0: print('epochs:', k)

            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.layers[l].activation(dot_value) # self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.layers[-1].activation_prime(a[-1])] # self.activation_prime(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.layers[l].activation_prime(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.layers[l].activation(np.dot(a, self.weights[l]))
        return a


if __name__ == '__main__':

    nn = NeuralNetwork([TanhLayer(2), TanhLayer(2), TanhLayer(3)])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]])

    nn.fit(X, y)

    #v = nn.predict(X)

    for e in X:
        print(e, nn.predict(e))