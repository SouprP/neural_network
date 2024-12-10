import numpy as np
import os
import matplotlib.pyplot as plt
from openpyxl import Workbook

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int,
                 activation_function='relu', filename=None, debug=False, iters_check=100):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.activation_function = activation_function
        self.filename = filename
        self.debug = debug
        self.iters_check = iters_check

        self.alpha = None
        self.iterations = None
        self.mse_history = []  # Przechowywanie MSE na potrzeby wizualizacji

        # Tworzenie katalogu na dane
        if not os.path.exists("./data/"):
            os.mkdir("./data/")

        # Wczytywanie wag lub inicjalizacja
        if self.filename and os.path.exists(f"./data/{self.filename}.npz"):
            self.load_data()
        else:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.init_params()

    def init_params(self):
        W1 = np.random.rand(self.hidden_size1, self.input_size) - 0.5
        b1 = np.random.rand(self.hidden_size1, 1) - 0.5
        W2 = np.random.rand(self.hidden_size2, self.hidden_size1) - 0.5
        b2 = np.random.rand(self.hidden_size2, 1) - 0.5
        W3 = np.random.rand(self.output_size, self.hidden_size2) - 0.5
        b3 = np.random.rand(self.output_size, 1) - 0.5
        return W1, b1, W2, b2, W3, b3

    def load_data(self):
        data = np.load(f"./data/{self.filename}.npz")
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        if self.debug:
            print(f"Model loaded from ./data/{self.filename}.npz")

    def save_data(self):
        if self.filename:
            np.savez(f"./data/{self.filename}", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
            if self.debug:
                print(f"Model saved to ./data/{self.filename}.npz")

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return x > 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.relu(Z1) if self.activation_function == 'relu' else self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.relu(Z2) if self.activation_function == 'relu' else self.sigmoid(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        return Z1, A1, Z2, A2, Z3

    def backward_prop(self, Z1, A1, Z2, A2, Z3, X, Y):
        dZ3 = Z3 - Y
        dW3 = 1 / X.shape[1] * dZ3.dot(A2.T)
        db3 = 1 / X.shape[1] * np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = self.W3.T.dot(dZ3) * (self.relu_deriv(Z2) if self.activation_function == 'relu' else self.sigmoid_deriv(Z2))
        dW2 = 1 / X.shape[1] * dZ2.dot(A1.T)
        db2 = 1 / X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * (self.relu_deriv(Z1) if self.activation_function == 'relu' else self.sigmoid_deriv(Z1))
        dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
        db1 = 1 / X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * db3

    def get_mse(self, predictions, actual):
        return np.mean((predictions - actual) ** 2)

    def train(self, X, Y, alpha, iterations, sheet=None):
        self.alpha = alpha
        self.iterations = iterations

        for i in range(iterations):
            Z1, A1, Z2, A2, Z3 = self.forward_prop(X)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(Z1, A1, Z2, A2, Z3, X, Y)
            self.update_params(dW1, db1, dW2, db2, dW3, db3)

            if i % self.iters_check == 0:
                mse = self.get_mse(Z3, Y)
                self.mse_history.append((i, mse))
                print(f"Iteration {i}: MSE {mse:.4f}")
                if sheet:
                    sheet.append([i, mse])

        self.save_data()

    def predict(self, X):
        _, _, _, _, Z3 = self.forward_prop(X)
        return Z3


    def plot_mse(self):
        iterations, mse_values = zip(*self.mse_history)
        plt.plot(iterations, mse_values)
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.show()
